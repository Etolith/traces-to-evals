use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::clustering::assignment::{ClusterAssignment, UNCLUSTERED};
use crate::clustering::discovery::{ClusterModel, DistanceMetric, EmbeddingClusterAssigner};
use crate::model::EvalCase;
use crate::{Result, TraceEvalError};

use super::{
    VectorIndex, VectorIndexRowMap, VectorMetric, VectorRowId, VectorSearchOptions,
    vector_index_error,
};

#[derive(Debug, Clone, PartialEq)]
pub struct VectorIndexClusterAssigner<I> {
    pub model: ClusterModel,
    pub index: I,
    pub row_map: BTreeMap<VectorRowId, String>,
    pub distance_metric: DistanceMetric,
    pub search_options: VectorSearchOptions,
    pub novelty_distance_threshold: Option<f32>,
}

impl<I> VectorIndexClusterAssigner<I>
where
    I: VectorIndex,
{
    pub fn new(model: ClusterModel, index: I, row_map: VectorIndexRowMap) -> Result<Self> {
        Self::new_with_distance_metric(model, index, row_map, DistanceMetric::Cosine)
    }

    pub fn new_with_distance_metric(
        model: ClusterModel,
        index: I,
        row_map: VectorIndexRowMap,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        let row_map = row_map.external_ids_by_row_id()?;
        validate_cluster_row_map(&model, &row_map)?;
        let metric = VectorMetric::from(distance_metric);
        if index.metric() != metric {
            return Err(TraceEvalError::VectorIndex {
                backend: "cluster-assignment".to_string(),
                message: format!(
                    "index metric {} does not match assignment metric {}",
                    index.metric().name(),
                    metric.name()
                ),
            });
        }

        Ok(Self {
            model,
            index,
            row_map,
            distance_metric,
            search_options: VectorSearchOptions::new(1, metric),
            novelty_distance_threshold: None,
        })
    }

    pub fn with_novelty_distance_threshold(mut self, threshold: f32) -> Self {
        self.novelty_distance_threshold = Some(threshold);
        self
    }

    pub fn with_search_options(mut self, options: VectorSearchOptions) -> Result<Self> {
        let metric = VectorMetric::from(self.distance_metric);
        if options.top_k != 1 {
            return Err(vector_index_error(
                "cluster-assignment",
                "cluster assignment requires vector search top_k = 1",
            ));
        }
        if options.metric != metric {
            return Err(vector_index_error(
                "cluster-assignment",
                format!(
                    "search metric {} does not match assignment metric {}",
                    options.metric.name(),
                    metric.name()
                ),
            ));
        }
        self.search_options = options;
        Ok(self)
    }
}

impl<I> EmbeddingClusterAssigner for VectorIndexClusterAssigner<I>
where
    I: VectorIndex,
{
    fn assign_case_embedding(
        &mut self,
        case: &EvalCase,
        embedding: &[f32],
    ) -> Result<ClusterAssignment> {
        let hits = self.index.search(embedding, &self.search_options)?;
        let Some(hit) = hits.first() else {
            return Ok(
                ClusterAssignment::new(case, UNCLUSTERED, 0.0, "embedding_vector_index")
                    .with_novelty(true),
            );
        };

        let cluster_id =
            self.row_map
                .get(&hit.row_id)
                .ok_or_else(|| TraceEvalError::VectorIndex {
                    backend: "cluster-assignment".to_string(),
                    message: format!("row_id {} is missing from row map", hit.row_id),
                })?;

        let novelty = self
            .novelty_distance_threshold
            .is_some_and(|threshold| hit.distance > threshold);
        let assigned_cluster_id = if novelty {
            UNCLUSTERED
        } else {
            cluster_id.as_str()
        };

        let mut assignment = ClusterAssignment::new(
            case,
            assigned_cluster_id,
            if novelty {
                0.0
            } else {
                self.distance_metric.confidence(hit.distance)
            },
            "embedding_vector_index",
        )
        .with_distance(hit.distance)
        .with_novelty(novelty);

        if novelty {
            assignment.metadata.insert(
                "nearest_cluster_id".to_string(),
                Value::String(cluster_id.clone()),
            );
        }

        Ok(assignment)
    }
}

fn validate_cluster_row_map(
    model: &ClusterModel,
    row_map: &BTreeMap<VectorRowId, String>,
) -> Result<()> {
    let cluster_ids = model
        .clusters
        .iter()
        .map(|cluster| cluster.id.as_str())
        .collect::<BTreeSet<_>>();

    for cluster_id in row_map.values() {
        if !cluster_ids.contains(cluster_id.as_str()) {
            return Err(vector_index_error(
                "cluster-assignment",
                format!("row map references unknown cluster {}", cluster_id),
            ));
        }
    }

    Ok(())
}
