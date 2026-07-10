use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::Result;

use super::records::{OwnedVectorRecord, owned_records, validate_records};
use super::{
    VectorIndex, VectorIndexBuilder, VectorMetric, VectorRecord, VectorSearchHit,
    VectorSearchOptions, compare_hits, validate_search_options,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BruteForceVectorIndex {
    metric: VectorMetric,
    dimensions: usize,
    records: Vec<OwnedVectorRecord>,
}

impl BruteForceVectorIndex {
    pub fn new(metric: VectorMetric, records: Vec<OwnedVectorRecord>) -> Result<Self> {
        let dimensions = validate_records("brute-force", &records)?;
        Ok(Self {
            metric,
            dimensions,
            records,
        })
    }

    pub fn records(&self) -> &[OwnedVectorRecord] {
        &self.records
    }
}

impl VectorIndex for BruteForceVectorIndex {
    fn metric(&self) -> VectorMetric {
        self.metric
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn search(
        &mut self,
        query: &[f32],
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchHit>> {
        validate_search_options("brute-force", self.metric, self.dimensions, query, options)?;

        let allowed = options
            .allowed_row_ids
            .as_ref()
            .map(|row_ids| row_ids.iter().copied().collect::<BTreeSet<_>>());
        let mut hits = Vec::new();

        for record in &self.records {
            if allowed
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(&record.row_id))
            {
                continue;
            }

            hits.push(VectorSearchHit {
                row_id: record.row_id,
                distance: self.metric.distance(query, &record.vector)?,
            });
        }

        hits.sort_by(compare_hits);
        hits.truncate(options.top_k);
        Ok(hits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BruteForceVectorIndexBuilder {
    metric: VectorMetric,
}

impl BruteForceVectorIndexBuilder {
    pub fn new(metric: VectorMetric) -> Self {
        Self { metric }
    }
}

impl VectorIndexBuilder for BruteForceVectorIndexBuilder {
    type Index = BruteForceVectorIndex;

    fn build(&self, records: &[VectorRecord<'_>]) -> Result<Self::Index> {
        BruteForceVectorIndex::new(self.metric, owned_records(records))
    }
}
