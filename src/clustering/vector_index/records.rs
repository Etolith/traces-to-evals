use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::Result;
use crate::clustering::discovery::ClusterModel;
use crate::clustering::embedding::CaseEmbedding;

use super::{VectorRecord, VectorRowId, vector_index_error};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OwnedVectorRecord {
    pub row_id: VectorRowId,
    pub external_id: String,
    pub vector: Vec<f32>,
}

impl OwnedVectorRecord {
    pub fn borrowed(&self) -> VectorRecord<'_> {
        VectorRecord::new(self.row_id, &self.external_id, &self.vector)
    }
}

pub fn cluster_centroid_records(model: &ClusterModel) -> Vec<OwnedVectorRecord> {
    model
        .clusters
        .iter()
        .enumerate()
        .filter_map(|(index, cluster)| {
            cluster.centroid.as_ref().map(|centroid| OwnedVectorRecord {
                row_id: index as VectorRowId,
                external_id: cluster.id.clone(),
                vector: centroid.clone(),
            })
        })
        .collect()
}

pub fn case_embedding_records(embeddings: &[CaseEmbedding]) -> Vec<OwnedVectorRecord> {
    embeddings
        .iter()
        .enumerate()
        .map(|(index, embedding)| OwnedVectorRecord {
            row_id: index as VectorRowId,
            external_id: embedding.case_id.clone(),
            vector: embedding.vector.clone(),
        })
        .collect()
}

pub fn borrowed_records(records: &[OwnedVectorRecord]) -> Vec<VectorRecord<'_>> {
    records.iter().map(OwnedVectorRecord::borrowed).collect()
}

pub(crate) fn owned_records(records: &[VectorRecord<'_>]) -> Vec<OwnedVectorRecord> {
    records
        .iter()
        .map(|record| OwnedVectorRecord {
            row_id: record.row_id,
            external_id: record.external_id.to_string(),
            vector: record.vector.to_vec(),
        })
        .collect()
}

pub(crate) fn validate_records(backend: &str, records: &[OwnedVectorRecord]) -> Result<usize> {
    if records.is_empty() {
        return Err(vector_index_error(backend, "index has no records"));
    }

    let dimensions = records[0].vector.len();
    if dimensions == 0 {
        return Err(vector_index_error(
            backend,
            "vectors must have at least one dimension",
        ));
    }

    let mut row_ids = BTreeSet::new();
    for record in records {
        if record.external_id.trim().is_empty() {
            return Err(vector_index_error(
                backend,
                format!("row {} external_id is empty", record.row_id),
            ));
        }
        if !row_ids.insert(record.row_id) {
            return Err(vector_index_error(
                backend,
                format!("duplicate row_id {}", record.row_id),
            ));
        }
        if record.vector.len() != dimensions {
            return Err(vector_index_error(
                backend,
                format!(
                    "record {} dimensions {} do not match expected {}",
                    record.external_id,
                    record.vector.len(),
                    dimensions
                ),
            ));
        }
        if record.vector.iter().any(|value| !value.is_finite()) {
            return Err(vector_index_error(
                backend,
                format!("record {} contains non-finite value", record.external_id),
            ));
        }
    }

    Ok(dimensions)
}
