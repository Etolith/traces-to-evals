mod assigner;
mod brute_force;
#[cfg(feature = "ann-paimon")]
mod paimon;
mod records;
mod row_map;
#[cfg(test)]
mod tests;

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::clustering::discovery::DistanceMetric;
use crate::{Result, TraceEvalError};

pub use assigner::VectorIndexClusterAssigner;
pub use brute_force::{BruteForceVectorIndex, BruteForceVectorIndexBuilder};
#[cfg(feature = "ann-paimon")]
pub use paimon::{
    PaimonHnswOptions, PaimonVectorIndex, PaimonVectorIndexBuilder, PaimonVectorIndexConfig,
    PaimonVectorIndexKind,
};
pub use records::{
    OwnedVectorRecord, borrowed_records, case_embedding_records, cluster_centroid_records,
};
pub use row_map::{VECTOR_INDEX_ROW_MAP_SCHEMA_KIND, VectorIndexRow, VectorIndexRowMap};

pub type VectorRowId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorMetric {
    Cosine,
    Euclidean,
    InnerProduct,
}

impl VectorMetric {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::Euclidean => "euclidean",
            Self::InnerProduct => "inner_product",
        }
    }

    pub(crate) fn distance(self, left: &[f32], right: &[f32]) -> Result<f32> {
        validate_same_dimensions(left, right)?;

        match self {
            Self::Cosine => cosine_distance(left, right),
            Self::Euclidean => euclidean_distance(left, right),
            Self::InnerProduct => negative_inner_product(left, right),
        }
    }
}

impl From<DistanceMetric> for VectorMetric {
    fn from(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Cosine => Self::Cosine,
            DistanceMetric::Euclidean => Self::Euclidean,
        }
    }
}

impl TryFrom<VectorMetric> for DistanceMetric {
    type Error = TraceEvalError;

    fn try_from(metric: VectorMetric) -> Result<Self> {
        match metric {
            VectorMetric::Cosine => Ok(Self::Cosine),
            VectorMetric::Euclidean => Ok(Self::Euclidean),
            VectorMetric::InnerProduct => Err(TraceEvalError::VectorIndex {
                backend: "cluster-assignment".to_string(),
                message: "inner_product does not have a cluster-assignment confidence policy"
                    .to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorRecord<'a> {
    pub row_id: VectorRowId,
    pub external_id: &'a str,
    pub vector: &'a [f32],
}

impl<'a> VectorRecord<'a> {
    pub fn new(row_id: VectorRowId, external_id: &'a str, vector: &'a [f32]) -> Self {
        Self {
            row_id,
            external_id,
            vector,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorSearchOptions {
    pub top_k: usize,
    pub metric: VectorMetric,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nprobe: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ef_search: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_row_ids: Option<Vec<VectorRowId>>,
}

impl VectorSearchOptions {
    pub fn new(top_k: usize, metric: VectorMetric) -> Self {
        Self {
            top_k,
            metric,
            nprobe: None,
            ef_search: None,
            allowed_row_ids: None,
        }
    }

    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = Some(nprobe);
        self
    }

    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = Some(ef_search);
        self
    }

    pub fn with_allowed_row_ids(mut self, row_ids: Vec<VectorRowId>) -> Self {
        self.allowed_row_ids = Some(row_ids);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorSearchHit {
    pub row_id: VectorRowId,
    pub distance: f32,
}

pub trait VectorIndex: Send + Sync {
    fn metric(&self) -> VectorMetric;
    fn dimensions(&self) -> usize;

    fn optimize_for_search(&mut self) -> Result<()> {
        Ok(())
    }

    fn search(
        &mut self,
        query: &[f32],
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchHit>>;

    fn search_batch(
        &mut self,
        queries: &[Vec<f32>],
        options: &VectorSearchOptions,
    ) -> Result<Vec<Vec<VectorSearchHit>>> {
        queries
            .iter()
            .map(|query| self.search(query, options))
            .collect()
    }
}

pub trait VectorIndexBuilder: Send + Sync {
    type Index: VectorIndex;

    fn build(&self, records: &[VectorRecord<'_>]) -> Result<Self::Index>;
}

pub(crate) fn validate_search_options(
    backend: &str,
    metric: VectorMetric,
    dimensions: usize,
    query: &[f32],
    options: &VectorSearchOptions,
) -> Result<()> {
    if options.top_k == 0 {
        return Err(vector_index_error(backend, "top_k must be greater than 0"));
    }
    if options.metric != metric {
        return Err(vector_index_error(
            backend,
            format!(
                "search metric {} does not match index metric {}",
                options.metric.name(),
                metric.name()
            ),
        ));
    }
    if query.len() != dimensions {
        return Err(vector_index_error(
            backend,
            format!(
                "query dimensions {} do not match index dimensions {}",
                query.len(),
                dimensions
            ),
        ));
    }
    if query.iter().any(|value| !value.is_finite()) {
        return Err(vector_index_error(
            backend,
            "query contains non-finite value",
        ));
    }

    Ok(())
}

pub(crate) fn vector_index_error(
    backend: impl Into<String>,
    message: impl Into<String>,
) -> TraceEvalError {
    TraceEvalError::VectorIndex {
        backend: backend.into(),
        message: message.into(),
    }
}

pub(crate) fn compare_hits(left: &VectorSearchHit, right: &VectorSearchHit) -> Ordering {
    left.distance
        .partial_cmp(&right.distance)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.row_id.cmp(&right.row_id))
}

fn validate_same_dimensions(left: &[f32], right: &[f32]) -> Result<()> {
    if left.len() != right.len() {
        return Err(vector_index_error(
            "brute-force",
            format!(
                "vector dimensions differ: {} vs {}",
                left.len(),
                right.len()
            ),
        ));
    }
    Ok(())
}

fn cosine_distance(left: &[f32], right: &[f32]) -> Result<f32> {
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;

    for (left, right) in left.iter().zip(right) {
        validate_finite(*left)?;
        validate_finite(*right)?;
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
        validate_finite(*left)?;
        validate_finite(*right)?;
        let delta = left - right;
        sum += delta * delta;
    }

    Ok(sum.sqrt())
}

fn negative_inner_product(left: &[f32], right: &[f32]) -> Result<f32> {
    let mut dot = 0.0f32;

    for (left, right) in left.iter().zip(right) {
        validate_finite(*left)?;
        validate_finite(*right)?;
        dot += left * right;
    }

    Ok(-dot)
}

fn validate_finite(value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(vector_index_error(
            "brute-force",
            "vector contains non-finite value",
        ))
    }
}
