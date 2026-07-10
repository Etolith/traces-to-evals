use std::collections::BTreeMap;

use serde_json::Value;

use crate::clustering::assignment::{ClusterAssignment, UNCLUSTERED};
use crate::clustering::embedding::CaseEmbedding;
use crate::model::EvalCase;
use crate::{Result, TraceEvalError};

use super::{ClusterModel, DiscoveredCluster, DistanceMetric};

pub trait EmbeddingClusterAssigner {
    fn assign_case_embedding(
        &mut self,
        case: &EvalCase,
        embedding: &[f32],
    ) -> Result<ClusterAssignment>;

    fn assign_case_embeddings(
        &mut self,
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
        &mut self,
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
