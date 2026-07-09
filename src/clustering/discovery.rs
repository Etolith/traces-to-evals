use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::calibration::HumanRating;
use crate::evaluation::EvaluationResult;
use crate::model::EvalCase;
use crate::{Result, TraceEvalError};

use super::embedding::CaseEmbedding;
use super::labeling::ClusterLabel;
use super::quality::{ClusterQuality, ClusterQualityReport};
use super::{ClusterAssignment, EvalCluster, UNCLUSTERED};

pub const CLUSTER_MODEL_SCHEMA_VERSION: &str = "traceeval.cluster_model.v1";

#[derive(Debug, Clone, Copy)]
pub struct ClusterDiscoveryInput<'a> {
    pub cases: &'a [EvalCase],
    pub embeddings: Option<&'a [CaseEmbedding]>,
    pub human_ratings: Option<&'a [HumanRating]>,
    pub previous_results: Option<&'a [EvaluationResult]>,
    pub options: &'a ClusterDiscoveryOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterDiscoveryOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    pub algorithm: ClusterAlgorithm,
    pub distance_metric: DistanceMetric,
    pub representative_count: usize,
    pub random_seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub novelty_distance_threshold: Option<f32>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl Default for ClusterDiscoveryOptions {
    fn default() -> Self {
        Self {
            model_id: None,
            algorithm: ClusterAlgorithm::KMeans {
                k: 2,
                max_iterations: 100,
                tolerance: 0.0001,
            },
            distance_metric: DistanceMetric::Cosine,
            representative_count: 5,
            random_seed: 42,
            novelty_distance_threshold: None,
            metadata: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterAlgorithm {
    KMeans {
        k: usize,
        max_iterations: usize,
        tolerance: f32,
    },
    Dbscan {
        min_points: usize,
        epsilon: f32,
    },
}

impl ClusterAlgorithm {
    pub fn name(self) -> &'static str {
        match self {
            Self::KMeans { .. } => "kmeans",
            Self::Dbscan { .. } => "dbscan",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
}

impl DistanceMetric {
    pub fn distance(self, left: &[f32], right: &[f32]) -> Result<f32> {
        if left.len() != right.len() {
            return Err(TraceEvalError::ClusterAssignment {
                case_id: "<embedding>".to_string(),
                message: format!(
                    "embedding dimensions differ: {} vs {}",
                    left.len(),
                    right.len()
                ),
            });
        }

        match self {
            Self::Cosine => cosine_distance(left, right),
            Self::Euclidean => euclidean_distance(left, right),
        }
    }

    fn confidence(self, distance: f32) -> f32 {
        match self {
            Self::Cosine => (1.0 - (distance / 2.0)).clamp(0.0, 1.0),
            Self::Euclidean => (1.0 - (distance / (1.0 + distance))).clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterModel {
    pub schema_version: String,
    pub model_id: String,
    pub created_at: String,
    pub source: ClusterModelSource,
    pub clusters: Vec<DiscoveredCluster>,
    pub assignments: Vec<ClusterAssignment>,
    pub quality: ClusterQualityReport,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl ClusterModel {
    pub fn new(
        model_id: impl Into<String>,
        created_at: impl Into<String>,
        source: ClusterModelSource,
        clusters: Vec<DiscoveredCluster>,
        assignments: Vec<ClusterAssignment>,
        quality: ClusterQualityReport,
    ) -> Self {
        Self {
            schema_version: CLUSTER_MODEL_SCHEMA_VERSION.to_string(),
            model_id: model_id.into(),
            created_at: created_at.into(),
            source,
            clusters,
            assignments,
            quality,
            metadata: BTreeMap::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CLUSTER_MODEL_SCHEMA_VERSION {
            return Err(self.validation_error(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }

        if self.clusters.is_empty() {
            return Err(self.validation_error("model has no clusters"));
        }

        let cluster_ids = self
            .clusters
            .iter()
            .map(|cluster| cluster.id.as_str())
            .collect::<BTreeSet<_>>();

        for cluster in &self.clusters {
            if cluster.size == 0 {
                return Err(self.validation_error(format!("cluster {} has size 0", cluster.id)));
            }
            if cluster.representative_case_ids.is_empty() {
                return Err(self.validation_error(format!(
                    "cluster {} has no representative cases",
                    cluster.id
                )));
            }
        }

        for assignment in &self.assignments {
            if assignment.cluster_id != UNCLUSTERED
                && !cluster_ids.contains(assignment.cluster_id.as_str())
            {
                return Err(self.validation_error(format!(
                    "assignment for case {} references unknown cluster {}",
                    assignment.case_id, assignment.cluster_id
                )));
            }
        }

        Ok(())
    }

    pub fn to_eval_clusters(&self) -> Vec<EvalCluster> {
        self.clusters
            .iter()
            .map(|cluster| {
                let mut metadata = cluster.metadata.clone();
                metadata.insert(
                    "source_model_id".to_string(),
                    Value::String(self.model_id.clone()),
                );
                metadata.insert("cluster_size".to_string(), Value::from(cluster.size));
                if let Some(radius) = cluster.radius {
                    metadata.insert("cluster_radius".to_string(), Value::from(radius));
                }

                EvalCluster {
                    id: cluster.id.clone(),
                    label: cluster
                        .label
                        .as_ref()
                        .map(|label| label.label.clone())
                        .unwrap_or_else(|| cluster.id.clone()),
                    description: cluster
                        .label
                        .as_ref()
                        .map(|label| label.description.clone()),
                    weight: 1.0,
                    metadata,
                }
            })
            .collect()
    }

    fn validation_error(&self, message: impl Into<String>) -> TraceEvalError {
        TraceEvalError::ClusterModelValidation {
            model_id: self.model_id.clone(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterModelSource {
    pub case_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_dimensions: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_version: Option<String>,
    pub algorithm: String,
    pub distance_metric: String,
    pub random_seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredCluster {
    pub id: String,
    pub size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub centroid: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub representative_case_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_distance: Option<f32>,
    pub quality: ClusterQuality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<ClusterLabel>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl DiscoveredCluster {
    pub fn new(id: impl Into<String>, size: usize, representative_case_ids: Vec<String>) -> Self {
        let id = id.into();
        Self {
            quality: ClusterQuality {
                representative_case_ids: representative_case_ids.clone(),
                ..ClusterQuality::new(id.clone(), size)
            },
            id,
            size,
            centroid: None,
            representative_case_ids,
            radius: None,
            mean_distance: None,
            label: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_centroid(mut self, centroid: Vec<f32>) -> Self {
        self.centroid = Some(centroid);
        self
    }
}

pub trait ClusterDiscovery {
    fn algorithm_name(&self) -> &'static str;
    fn fit(&self, input: ClusterDiscoveryInput<'_>) -> Result<ClusterModel>;
}

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

    fn fit(&self, _input: ClusterDiscoveryInput<'_>) -> Result<ClusterModel> {
        Err(TraceEvalError::ClusterDiscovery {
            algorithm: self.algorithm_name().to_string(),
            message: "K-Means discovery requires the clustering-linfa implementation".to_string(),
        })
    }
}

pub trait EmbeddingClusterAssigner {
    fn assign_case_embedding(
        &self,
        case: &EvalCase,
        embedding: &[f32],
    ) -> Result<ClusterAssignment>;

    fn assign_case_embeddings(
        &self,
        cases: &[EvalCase],
        embeddings: &[CaseEmbedding],
    ) -> Result<Vec<ClusterAssignment>> {
        let embeddings_by_case = embeddings
            .iter()
            .map(|embedding| (embedding.case_id.as_str(), embedding))
            .collect::<BTreeMap<_, _>>();

        cases
            .iter()
            .map(|case| {
                let embedding = embeddings_by_case.get(case.id.as_str()).ok_or_else(|| {
                    TraceEvalError::InvalidEmbedding {
                        case_id: case.id.clone(),
                        message: "missing embedding for case".to_string(),
                    }
                })?;
                self.assign_case_embedding(case, &embedding.vector)
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterModelAssigner {
    pub model: ClusterModel,
    pub distance_metric: DistanceMetric,
    pub novelty_distance_threshold: Option<f32>,
}

impl ClusterModelAssigner {
    pub fn new(model: ClusterModel) -> Self {
        Self {
            model,
            distance_metric: DistanceMetric::Cosine,
            novelty_distance_threshold: None,
        }
    }

    pub fn with_distance_metric(mut self, distance_metric: DistanceMetric) -> Self {
        self.distance_metric = distance_metric;
        self
    }

    pub fn with_novelty_distance_threshold(mut self, threshold: f32) -> Self {
        self.novelty_distance_threshold = Some(threshold);
        self
    }
}

impl EmbeddingClusterAssigner for ClusterModelAssigner {
    fn assign_case_embedding(
        &self,
        case: &EvalCase,
        embedding: &[f32],
    ) -> Result<ClusterAssignment> {
        let mut nearest: Option<(&DiscoveredCluster, f32)> = None;

        for cluster in &self.model.clusters {
            let Some(centroid) = cluster.centroid.as_deref() else {
                continue;
            };
            let distance = self.distance_metric.distance(embedding, centroid)?;
            if nearest.is_none_or(|(_, best_distance)| distance < best_distance) {
                nearest = Some((cluster, distance));
            }
        }

        let Some((cluster, distance)) = nearest else {
            return Ok(ClusterAssignment::new(
                case,
                UNCLUSTERED,
                0.0,
                "embedding_nearest_centroid",
            )
            .with_novelty(true));
        };

        let novelty = self
            .novelty_distance_threshold
            .is_some_and(|threshold| distance > threshold);
        let assigned_cluster_id = if novelty {
            UNCLUSTERED
        } else {
            cluster.id.as_str()
        };

        let mut assignment = ClusterAssignment::new(
            case,
            assigned_cluster_id,
            if novelty {
                0.0
            } else {
                self.distance_metric.confidence(distance)
            },
            "embedding_nearest_centroid",
        )
        .with_distance(distance)
        .with_novelty(novelty);

        if novelty {
            assignment.metadata.insert(
                "nearest_cluster_id".to_string(),
                Value::String(cluster.id.clone()),
            );
        }

        Ok(assignment)
    }
}

fn cosine_distance(left: &[f32], right: &[f32]) -> Result<f32> {
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;

    for (left, right) in left.iter().zip(right) {
        if !left.is_finite() || !right.is_finite() {
            return Err(TraceEvalError::ClusterAssignment {
                case_id: "<embedding>".to_string(),
                message: "embedding contains non-finite value".to_string(),
            });
        }
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        return Ok(1.0);
    }

    Ok(1.0 - (dot / (left_norm.sqrt() * right_norm.sqrt())))
}

fn euclidean_distance(left: &[f32], right: &[f32]) -> Result<f32> {
    let mut sum = 0.0f32;

    for (left, right) in left.iter().zip(right) {
        if !left.is_finite() || !right.is_finite() {
            return Err(TraceEvalError::ClusterAssignment {
                case_id: "<embedding>".to_string(),
                message: "embedding contains non-finite value".to_string(),
            });
        }
        let delta = left - right;
        sum += delta * delta;
    }

    Ok(sum.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ClusterModel {
        let clusters = vec![
            DiscoveredCluster::new("cluster-0001", 2, vec!["case-a".to_string()])
                .with_centroid(vec![1.0, 0.0]),
            DiscoveredCluster::new("cluster-0002", 2, vec!["case-b".to_string()])
                .with_centroid(vec![0.0, 1.0]),
        ];
        let quality = ClusterQualityReport {
            cluster_count: 2,
            assigned_case_count: 4,
            mean_distance: None,
            silhouette_score: None,
            clusters: clusters
                .iter()
                .map(|cluster| cluster.quality.clone())
                .collect(),
        };

        ClusterModel::new(
            "model-1",
            "2026-01-01T00:00:00Z",
            ClusterModelSource {
                case_count: 4,
                embedding_provider: Some("test".to_string()),
                embedding_model: Some("test".to_string()),
                embedding_dimensions: Some(2),
                projection_version: Some("projection".to_string()),
                algorithm: "manual".to_string(),
                distance_metric: "cosine".to_string(),
                random_seed: 42,
            },
            clusters,
            Vec::new(),
            quality,
        )
    }

    #[test]
    fn nearest_centroid_assignment_sets_distance_confidence_and_method() {
        let assigner = ClusterModelAssigner::new(model());
        let case = EvalCase::new("case-new", "trace-new", "input");

        let assignment = assigner.assign_case_embedding(&case, &[0.9, 0.1]).unwrap();

        assert_eq!(assignment.cluster_id, "cluster-0001");
        assert_eq!(assignment.method, "embedding_nearest_centroid");
        assert!(assignment.distance.unwrap() < 0.1);
        assert!(assignment.confidence > 0.95);
        assert!(!assignment.novelty);
    }

    #[test]
    fn nearest_centroid_assignment_marks_novelty() {
        let assigner = ClusterModelAssigner::new(model()).with_novelty_distance_threshold(0.01);
        let case = EvalCase::new("case-new", "trace-new", "input");

        let assignment = assigner.assign_case_embedding(&case, &[0.7, 0.7]).unwrap();

        assert_eq!(assignment.cluster_id, UNCLUSTERED);
        assert!(assignment.novelty);
        assert_eq!(
            assignment.metadata.get("nearest_cluster_id"),
            Some(&Value::String("cluster-0001".to_string()))
        );
    }

    #[test]
    fn cluster_model_validation_rejects_unknown_assignment_cluster() {
        let mut model = model();
        let case = EvalCase::new("case-new", "trace-new", "input");
        model.assignments.push(ClusterAssignment::new(
            &case,
            "missing-cluster",
            1.0,
            "test",
        ));

        assert!(model.validate().is_err());
    }
}
