#[cfg(feature = "clustering-linfa")]
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "clustering-linfa")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "clustering-linfa")]
use linfa::DatasetBase;
#[cfg(feature = "clustering-linfa")]
use linfa::prelude::{Fit, Predict};
#[cfg(feature = "clustering-linfa")]
use linfa_clustering::KMeans;
#[cfg(feature = "clustering-linfa")]
use ndarray::Array2;
#[cfg(feature = "clustering-linfa")]
use rand::{SeedableRng, rngs::SmallRng};
#[cfg(feature = "clustering-linfa")]
use serde_json::Value;

#[cfg(feature = "clustering-linfa")]
use crate::clustering::assignment::ClusterAssignment;
#[cfg(feature = "clustering-linfa")]
use crate::clustering::embedding::CaseEmbedding;
#[cfg(feature = "clustering-linfa")]
use crate::clustering::quality::{ClusterQuality, ClusterQualityReport};
#[cfg(feature = "clustering-linfa")]
use crate::model::EvalCase;
use crate::{Result, TraceEvalError};

use super::{ClusterDiscovery, ClusterDiscoveryInput, ClusterModel};
#[cfg(feature = "clustering-linfa")]
use super::{ClusterModelSource, DiscoveredCluster, DistanceMetric};

#[derive(Debug, Clone, PartialEq)]
pub struct KMeansClusterDiscovery {
    pub k: usize,
    pub max_iterations: usize,
    pub tolerance: f32,
    pub random_seed: u64,
}

impl ClusterDiscovery for KMeansClusterDiscovery {
    fn algorithm_name(&self) -> &'static str {
        "kmeans"
    }

    #[cfg(feature = "clustering-linfa")]
    fn fit(&self, input: ClusterDiscoveryInput<'_>) -> Result<ClusterModel> {
        self.fit_with_linfa(input)
    }

    #[cfg(not(feature = "clustering-linfa"))]
    fn fit(&self, _input: ClusterDiscoveryInput<'_>) -> Result<ClusterModel> {
        Err(self.discovery_error("K-Means discovery requires the clustering-linfa implementation"))
    }
}

impl KMeansClusterDiscovery {
    fn discovery_error(&self, message: impl Into<String>) -> TraceEvalError {
        TraceEvalError::ClusterDiscovery {
            algorithm: self.algorithm_name().to_string(),
            message: message.into(),
        }
    }
}

#[cfg(feature = "clustering-linfa")]
impl KMeansClusterDiscovery {
    fn fit_with_linfa(&self, input: ClusterDiscoveryInput<'_>) -> Result<ClusterModel> {
        if self.k == 0 {
            return Err(self.discovery_error("k must be greater than 0"));
        }
        if input.cases.len() < self.k {
            return Err(self.discovery_error(format!(
                "cannot discover {} clusters from {} cases",
                self.k,
                input.cases.len()
            )));
        }
        if input.options.representative_count == 0 {
            return Err(self.discovery_error("representative_count must be greater than 0"));
        }

        let embeddings = input
            .embeddings
            .ok_or_else(|| self.discovery_error("K-Means discovery requires embeddings"))?;
        let ordered_embeddings = ordered_embeddings_for_cases(input.cases, embeddings)?;
        let dimensions = embedding_dimensions(&ordered_embeddings)?;
        let distance_metric = input.options.distance_metric;
        let records_f32 = ordered_embeddings
            .iter()
            .map(|embedding| discovery_vector(distance_metric, &embedding.vector))
            .collect::<Vec<_>>();
        let records_f64 = records_f32
            .iter()
            .flat_map(|record| record.iter().map(|value| f64::from(*value)))
            .collect::<Vec<_>>();
        let records = Array2::from_shape_vec((input.cases.len(), dimensions), records_f64)
            .map_err(|error| {
                self.discovery_error(format!("failed to build embedding matrix: {error}"))
            })?;
        let dataset = DatasetBase::from(records);
        let rng = SmallRng::seed_from_u64(self.random_seed);
        let kmeans = KMeans::params_with_rng(self.k, rng)
            .max_n_iterations(self.max_iterations as u64)
            .tolerance(f64::from(self.tolerance))
            .fit(&dataset)
            .map_err(|error| self.discovery_error(error.to_string()))?;
        let labels = kmeans.predict(&dataset);
        let labels = labels.iter().copied().collect::<Vec<_>>();
        let silhouette = silhouette_scores(&records_f32, &labels, distance_metric)?;
        let centroids = kmeans
            .centroids()
            .outer_iter()
            .map(|centroid| {
                centroid
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut members_by_label = BTreeMap::<usize, Vec<(usize, f32)>>::new();

        for (case_index, label) in labels.iter().copied().enumerate() {
            let centroid = centroids.get(label).ok_or_else(|| {
                self.discovery_error(format!("missing centroid for predicted cluster {label}"))
            })?;
            let distance = distance_metric.distance(&records_f32[case_index], centroid)?;
            members_by_label
                .entry(label)
                .or_default()
                .push((case_index, distance));
        }

        let mut sorted_labels = members_by_label.keys().copied().collect::<Vec<_>>();
        sorted_labels.sort_by(|left, right| {
            members_by_label[right]
                .len()
                .cmp(&members_by_label[left].len())
                .then_with(|| left.cmp(right))
        });

        let mut assignments_by_case = vec![None; input.cases.len()];
        let mut clusters = Vec::new();
        let mut total_distance = 0.0f32;

        for (cluster_index, label) in sorted_labels.iter().copied().enumerate() {
            let cluster_id = format!("cluster-{number:04}", number = cluster_index + 1);
            let mut members = members_by_label.remove(&label).unwrap_or_default();
            members.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| input.cases[left.0].id.cmp(&input.cases[right.0].id))
            });
            let size = members.len();
            let distance_sum = members.iter().map(|(_, distance)| *distance).sum::<f32>();
            total_distance += distance_sum;
            let mean_distance = distance_sum / size as f32;
            let max_distance = members
                .iter()
                .map(|(_, distance)| *distance)
                .max_by(f32::total_cmp)
                .unwrap_or(0.0);
            let representative_case_ids = members
                .iter()
                .take(input.options.representative_count)
                .map(|(case_index, _)| input.cases[*case_index].id.clone())
                .collect::<Vec<_>>();
            let cluster_silhouette = silhouette
                .as_ref()
                .and_then(|(_, cluster_scores)| cluster_scores.get(&label).copied());
            let quality = ClusterQuality {
                cluster_id: cluster_id.clone(),
                size,
                mean_distance: Some(mean_distance),
                max_distance: Some(max_distance),
                silhouette_score: cluster_silhouette,
                representative_case_ids: representative_case_ids.clone(),
            };
            let mut metadata = BTreeMap::new();
            metadata.insert("linfa_cluster_index".to_string(), Value::from(label as u64));

            let cluster = DiscoveredCluster {
                id: cluster_id.clone(),
                size,
                centroid: centroids.get(label).cloned(),
                representative_case_ids,
                radius: Some(max_distance),
                mean_distance: Some(mean_distance),
                quality,
                label: None,
                metadata,
            };

            for (case_index, distance) in members {
                assignments_by_case[case_index] = Some(
                    ClusterAssignment::new(
                        &input.cases[case_index],
                        cluster_id.clone(),
                        distance_metric.confidence(distance),
                        "kmeans",
                    )
                    .with_distance(distance),
                );
            }

            clusters.push(cluster);
        }

        let assignments = assignments_by_case
            .into_iter()
            .map(|assignment| {
                assignment.ok_or_else(|| self.discovery_error("missing assignment after K-Means"))
            })
            .collect::<Result<Vec<_>>>()?;
        let quality = ClusterQualityReport {
            cluster_count: clusters.len(),
            assigned_case_count: assignments.len(),
            mean_distance: (!assignments.is_empty())
                .then(|| total_distance / assignments.len() as f32),
            silhouette_score: silhouette.map(|(overall, _)| overall),
            clusters: clusters
                .iter()
                .map(|cluster| cluster.quality.clone())
                .collect(),
        };
        let source = ClusterModelSource {
            case_count: input.cases.len(),
            embedding_provider: common_embedding_field(&ordered_embeddings, |embedding| {
                embedding.provider.as_str()
            }),
            embedding_model: common_embedding_field(&ordered_embeddings, |embedding| {
                embedding.model.as_str()
            }),
            embedding_dimensions: Some(dimensions),
            projection_version: common_embedding_field(&ordered_embeddings, |embedding| {
                embedding.projection_version.as_str()
            }),
            algorithm: self.algorithm_name().to_string(),
            distance_metric: distance_metric.name().to_string(),
            random_seed: self.random_seed,
        };
        let created_at = unix_seconds().to_string();
        let model_id = input
            .options
            .model_id
            .clone()
            .unwrap_or_else(|| format!("cluster-model-{created_at}"));
        let mut model = ClusterModel::new_with_project(
            &input.options.project_name,
            model_id,
            created_at,
            source,
            clusters,
            assignments,
            quality,
        );
        model.metadata = input.options.metadata.clone();
        model.validate()?;
        Ok(model)
    }
}

#[cfg(feature = "clustering-linfa")]
fn ordered_embeddings_for_cases<'a>(
    cases: &[EvalCase],
    embeddings: &'a [CaseEmbedding],
) -> Result<Vec<&'a CaseEmbedding>> {
    let case_ids = cases
        .iter()
        .map(|case| case.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut embeddings_by_case = BTreeMap::new();

    for embedding in embeddings {
        embedding.validate()?;
        if embeddings_by_case
            .insert(embedding.case_id.as_str(), embedding)
            .is_some()
        {
            return Err(TraceEvalError::InvalidEmbedding {
                case_id: embedding.case_id.clone(),
                message: "duplicate embedding for case".to_string(),
            });
        }
    }

    for embedding in embeddings {
        if !case_ids.contains(embedding.case_id.as_str()) {
            return Err(TraceEvalError::InvalidEmbedding {
                case_id: embedding.case_id.clone(),
                message: "embedding does not match an input case".to_string(),
            });
        }
    }

    cases
        .iter()
        .map(|case| {
            embeddings_by_case
                .get(case.id.as_str())
                .copied()
                .ok_or_else(|| TraceEvalError::InvalidEmbedding {
                    case_id: case.id.clone(),
                    message: "missing embedding for case".to_string(),
                })
        })
        .collect()
}

#[cfg(feature = "clustering-linfa")]
fn embedding_dimensions(embeddings: &[&CaseEmbedding]) -> Result<usize> {
    let Some(first) = embeddings.first() else {
        return Err(TraceEvalError::InvalidEmbedding {
            case_id: "<dataset>".to_string(),
            message: "embedding dataset is empty".to_string(),
        });
    };
    let dimensions = first.dimensions;

    for embedding in embeddings {
        if embedding.dimensions != dimensions {
            return Err(TraceEvalError::InvalidEmbedding {
                case_id: embedding.case_id.clone(),
                message: format!(
                    "embedding dimensions {} do not match expected {}",
                    embedding.dimensions, dimensions
                ),
            });
        }
    }

    Ok(dimensions)
}

#[cfg(feature = "clustering-linfa")]
fn discovery_vector(distance_metric: DistanceMetric, vector: &[f32]) -> Vec<f32> {
    match distance_metric {
        DistanceMetric::Cosine => normalized_vector(vector),
        DistanceMetric::Euclidean => vector.to_vec(),
    }
}

#[cfg(feature = "clustering-linfa")]
fn normalized_vector(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vector.to_vec();
    }

    vector.iter().map(|value| value / norm).collect()
}

#[cfg(feature = "clustering-linfa")]
fn common_embedding_field<F>(embeddings: &[&CaseEmbedding], field: F) -> Option<String>
where
    F: Fn(&CaseEmbedding) -> &str,
{
    let first = field(embeddings.first().copied()?);
    embeddings
        .iter()
        .all(|embedding| field(embedding) == first)
        .then(|| first.to_string())
}

#[cfg(feature = "clustering-linfa")]
fn silhouette_scores(
    records: &[Vec<f32>],
    labels: &[usize],
    distance_metric: DistanceMetric,
) -> Result<Option<(f32, BTreeMap<usize, f32>)>> {
    if records.len() <= 1 || records.len() > 10_000 {
        return Ok(None);
    }

    let cluster_labels = labels.iter().copied().collect::<BTreeSet<_>>();
    if cluster_labels.len() <= 1 {
        return Ok(None);
    }

    let mut silhouette_sum = 0.0f32;
    let mut cluster_sums = BTreeMap::<usize, (f32, usize)>::new();

    for (index, own_label) in labels.iter().copied().enumerate() {
        let mut same_total = 0.0f32;
        let mut same_count = 0usize;
        let mut other_totals = BTreeMap::<usize, (f32, usize)>::new();

        for (other_index, other_label) in labels.iter().copied().enumerate() {
            if index == other_index {
                continue;
            }
            let distance = distance_metric.distance(&records[index], &records[other_index])?;
            if other_label == own_label {
                same_total += distance;
                same_count += 1;
            } else {
                let entry = other_totals.entry(other_label).or_default();
                entry.0 += distance;
                entry.1 += 1;
            }
        }

        let a = if same_count == 0 {
            0.0
        } else {
            same_total / same_count as f32
        };
        let b = other_totals
            .values()
            .filter_map(|(total, count)| (*count > 0).then_some(total / *count as f32))
            .min_by(f32::total_cmp)
            .unwrap_or(0.0);
        let denominator = a.max(b);
        let silhouette = if denominator > 0.0 {
            (b - a) / denominator
        } else {
            0.0
        };

        silhouette_sum += silhouette;
        let entry = cluster_sums.entry(own_label).or_default();
        entry.0 += silhouette;
        entry.1 += 1;
    }

    let cluster_scores = cluster_sums
        .into_iter()
        .filter_map(|(label, (sum, count))| (count > 0).then_some((label, sum / count as f32)))
        .collect::<BTreeMap<_, _>>();

    Ok(Some((
        silhouette_sum / records.len() as f32,
        cluster_scores,
    )))
}

#[cfg(feature = "clustering-linfa")]
fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}
