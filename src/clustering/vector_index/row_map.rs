use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::project::ProjectName;
use crate::{Result, TraceEvalError};

use super::{VectorRecord, VectorRowId};

pub const VECTOR_INDEX_ROW_MAP_SCHEMA_KIND: &str = "vector_index_row_map";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorIndexRowMap {
    pub schema_version: String,
    pub index_id: String,
    pub rows: Vec<VectorIndexRow>,
}

impl VectorIndexRowMap {
    pub fn new(index_id: impl Into<String>, rows: Vec<VectorIndexRow>) -> Self {
        Self::new_with_project(&ProjectName::default(), index_id, rows)
    }

    pub fn new_with_project(
        project_name: &ProjectName,
        index_id: impl Into<String>,
        rows: Vec<VectorIndexRow>,
    ) -> Self {
        Self {
            schema_version: project_name.schema_version(VECTOR_INDEX_ROW_MAP_SCHEMA_KIND, 1),
            index_id: index_id.into(),
            rows,
        }
    }

    pub fn from_records(index_id: impl Into<String>, records: &[VectorRecord<'_>]) -> Self {
        Self::from_records_with_project(&ProjectName::default(), index_id, records)
    }

    pub fn from_records_with_project(
        project_name: &ProjectName,
        index_id: impl Into<String>,
        records: &[VectorRecord<'_>],
    ) -> Self {
        Self::new_with_project(
            project_name,
            index_id,
            records
                .iter()
                .map(|record| VectorIndexRow {
                    row_id: record.row_id,
                    external_id: record.external_id.to_string(),
                })
                .collect(),
        )
    }

    pub fn validate(&self) -> Result<()> {
        if !ProjectName::matches_schema_version(
            &self.schema_version,
            VECTOR_INDEX_ROW_MAP_SCHEMA_KIND,
            1,
        ) {
            return Err(self.error(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }

        if self.index_id.trim().is_empty() {
            return Err(self.error("index_id must not be empty"));
        }
        if self.rows.is_empty() {
            return Err(self.error("row map has no rows"));
        }

        let mut row_ids = BTreeSet::new();
        let mut external_ids = BTreeSet::new();
        for row in &self.rows {
            if row.external_id.trim().is_empty() {
                return Err(self.error(format!("row {} external_id is empty", row.row_id)));
            }
            if !row_ids.insert(row.row_id) {
                return Err(self.error(format!("duplicate row_id {}", row.row_id)));
            }
            if !external_ids.insert(row.external_id.as_str()) {
                return Err(self.error(format!("duplicate external_id {}", row.external_id)));
            }
        }

        Ok(())
    }

    pub fn external_ids_by_row_id(&self) -> Result<BTreeMap<VectorRowId, String>> {
        self.validate()?;
        Ok(self
            .rows
            .iter()
            .map(|row| (row.row_id, row.external_id.clone()))
            .collect())
    }

    pub fn external_id_for(&self, row_id: VectorRowId) -> Result<&str> {
        self.validate()?;
        self.rows
            .iter()
            .find(|row| row.row_id == row_id)
            .map(|row| row.external_id.as_str())
            .ok_or_else(|| self.error(format!("row_id {} is missing from row map", row_id)))
    }

    fn error(&self, message: impl Into<String>) -> TraceEvalError {
        TraceEvalError::VectorIndex {
            backend: "row-map".to_string(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorIndexRow {
    pub row_id: VectorRowId,
    pub external_id: String,
}
