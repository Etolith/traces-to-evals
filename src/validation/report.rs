use serde::{Deserialize, Serialize};

use crate::{Result, TraceEvalError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationProfile {
    DraftCases,
    RunnableCases,
    EvaluationResults,
    CalibrationDataset,
    EmbeddingDataset,
    ClusterModel,
    ClusterAssignments,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationReport {
    pub checked_cases: usize,
    pub checked_results: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub checked_embeddings: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub checked_cluster_models: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub checked_cluster_assignments: usize,
    pub errors: Vec<ValidationIssue>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }

    pub fn ensure_valid(&self) -> Result<()> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(TraceEvalError::ValidationFailed {
                error_count: self.error_count(),
                warning_count: self.warning_count(),
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub case_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}
