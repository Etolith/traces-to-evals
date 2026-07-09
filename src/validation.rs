use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::clustering::{CaseEmbedding, ClusterAssignment, ClusterModel};
use crate::evaluation::{EvaluationResult, ScoreScale};
use crate::model::EvalCase;
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

#[derive(Debug, Default)]
pub struct ValidationReportBuilder {
    checked_cases: usize,
    checked_results: usize,
    checked_embeddings: usize,
    checked_cluster_models: usize,
    checked_cluster_assignments: usize,
    errors: Vec<ValidationIssue>,
    warnings: Vec<ValidationIssue>,
}

impl ValidationReportBuilder {
    pub fn check_cases(self, cases: &[EvalCase]) -> Self {
        self.check_cases_with_profile(cases, ValidationProfile::RunnableCases)
    }

    pub fn check_cases_with_profile(
        mut self,
        cases: &[EvalCase],
        profile: ValidationProfile,
    ) -> Self {
        self.checked_cases += cases.len();
        let mut seen_case_ids = BTreeSet::<&str>::new();

        for case in cases {
            if case.id.trim().is_empty() {
                self.push_error("missing_case_id", "case id is empty", None, None);
            } else if !seen_case_ids.insert(case.id.as_str()) {
                self.push_error(
                    "duplicate_case_id",
                    "case id appears more than once",
                    Some(case.id.clone()),
                    Some(case.trace_id.clone()),
                );
            }

            if case.trace_id.trim().is_empty() {
                self.push_error(
                    "missing_trace_id",
                    "trace id is empty",
                    Some(case.id.clone()),
                    None,
                );
            }

            if case.input.trim().is_empty() {
                self.push_error(
                    "missing_input",
                    "case input is empty",
                    Some(case.id.clone()),
                    Some(case.trace_id.clone()),
                );
            }

            if case
                .actual_output
                .as_deref()
                .is_none_or(|output| output.trim().is_empty())
            {
                match profile {
                    ValidationProfile::DraftCases
                    | ValidationProfile::EmbeddingDataset
                    | ValidationProfile::ClusterModel
                    | ValidationProfile::ClusterAssignments => {
                        self.push_warning(
                            "missing_actual_output",
                            "case actual_output is missing or empty",
                            Some(case.id.clone()),
                            Some(case.trace_id.clone()),
                        );
                    }
                    ValidationProfile::RunnableCases | ValidationProfile::CalibrationDataset => {
                        self.push_error(
                            "missing_actual_output",
                            "case actual_output is missing or empty",
                            Some(case.id.clone()),
                            Some(case.trace_id.clone()),
                        );
                    }
                    ValidationProfile::EvaluationResults => {}
                }
            }
        }

        self
    }

    pub fn check_results(self, results: &[EvaluationResult]) -> Self {
        self.check_results_with_profile(results, ValidationProfile::EvaluationResults)
    }

    pub fn check_results_with_profile(
        mut self,
        results: &[EvaluationResult],
        _profile: ValidationProfile,
    ) -> Self {
        self.checked_results += results.len();
        let mut seen_result_keys = BTreeSet::<(&str, &str)>::new();

        for result in results {
            if result.case_id.trim().is_empty() {
                self.push_error("missing_case_id", "result case_id is empty", None, None);
            }

            if result.trace_id.trim().is_empty() {
                self.push_error(
                    "missing_trace_id",
                    "result trace_id is empty",
                    Some(result.case_id.clone()),
                    None,
                );
            }

            if result.evaluator_name.trim().is_empty() {
                self.push_error(
                    "missing_evaluator_name",
                    "result evaluator_name is empty",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            } else if !seen_result_keys
                .insert((result.case_id.as_str(), result.evaluator_name.as_str()))
            {
                self.push_error(
                    "duplicate_result",
                    "case has more than one result for the same evaluator",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            }

            if !result.raw_score.is_finite() {
                self.push_error(
                    "invalid_raw_score",
                    "raw_score must be finite",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            } else if !raw_score_in_scale(result.raw_score, result.score_scale) {
                self.push_error(
                    "invalid_raw_score",
                    "raw_score is outside its score_scale",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            }

            if !unit_interval(result.normalized_score) {
                self.push_error(
                    "invalid_normalized_score",
                    "normalized_score must be between 0.0 and 1.0",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            }

            if result
                .calibrated_score
                .is_some_and(|score| !unit_interval(score))
            {
                self.push_error(
                    "invalid_calibrated_score",
                    "calibrated_score must be between 0.0 and 1.0",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            }
        }

        self
    }

    pub fn check_overlap(mut self, cases: &[EvalCase], results: &[EvaluationResult]) -> Self {
        let case_ids = cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<BTreeSet<_>>();

        for result in results {
            if !case_ids.contains(result.case_id.as_str()) {
                self.push_error(
                    "unknown_result_case",
                    "result case_id is not present in cases",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            }
        }

        self
    }

    pub fn check_embeddings(mut self, embeddings: &[CaseEmbedding]) -> Self {
        self.checked_embeddings += embeddings.len();
        let mut seen_case_ids = BTreeSet::<&str>::new();
        let mut expected_dimensions: Option<usize> = None;

        for embedding in embeddings {
            if embedding.case_id.trim().is_empty() {
                self.push_error("missing_case_id", "embedding case_id is empty", None, None);
            } else if !seen_case_ids.insert(embedding.case_id.as_str()) {
                self.push_error(
                    "duplicate_embedding",
                    "case has more than one embedding",
                    Some(embedding.case_id.clone()),
                    Some(embedding.trace_id.clone()),
                );
            }

            if let Err(error) = embedding.validate() {
                self.push_error(
                    "invalid_embedding",
                    error.to_string(),
                    Some(embedding.case_id.clone()),
                    Some(embedding.trace_id.clone()),
                );
            }

            match expected_dimensions {
                Some(dimensions) if dimensions != embedding.dimensions => self.push_error(
                    "mixed_embedding_dimensions",
                    "all embeddings in one dataset must have identical dimensions",
                    Some(embedding.case_id.clone()),
                    Some(embedding.trace_id.clone()),
                ),
                None => expected_dimensions = Some(embedding.dimensions),
                _ => {}
            }
        }

        self
    }

    pub fn check_embedding_overlap(
        mut self,
        cases: &[EvalCase],
        embeddings: &[CaseEmbedding],
        allow_extra_embeddings: bool,
    ) -> Self {
        let case_ids = cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<BTreeSet<_>>();
        let embedding_case_ids = embeddings
            .iter()
            .map(|embedding| embedding.case_id.as_str())
            .collect::<BTreeSet<_>>();

        for case in cases {
            if !embedding_case_ids.contains(case.id.as_str()) {
                self.push_error(
                    "missing_embedding",
                    "case has no embedding row",
                    Some(case.id.clone()),
                    Some(case.trace_id.clone()),
                );
            }
        }

        if !allow_extra_embeddings {
            for embedding in embeddings {
                if !case_ids.contains(embedding.case_id.as_str()) {
                    self.push_error(
                        "unknown_embedding_case",
                        "embedding case_id is not present in cases",
                        Some(embedding.case_id.clone()),
                        Some(embedding.trace_id.clone()),
                    );
                }
            }
        }

        self
    }

    pub fn check_cluster_model(mut self, model: &ClusterModel) -> Self {
        self.checked_cluster_models += 1;

        if let Err(error) = model.validate() {
            self.push_error("invalid_cluster_model", error.to_string(), None, None);
        }

        for cluster in &model.clusters {
            if let Some(label) = &cluster.label {
                if label.label.trim().is_empty() {
                    self.push_error(
                        "invalid_cluster_label",
                        format!("cluster {} label is empty", cluster.id),
                        None,
                        None,
                    );
                }
                if label.label.chars().count() > 80 {
                    self.push_error(
                        "invalid_cluster_label",
                        format!("cluster {} label exceeds 80 characters", cluster.id),
                        None,
                        None,
                    );
                }
                if !unit_interval(label.confidence) {
                    self.push_error(
                        "invalid_cluster_label_confidence",
                        format!("cluster {} label confidence must be 0.0..=1.0", cluster.id),
                        None,
                        None,
                    );
                }
            }
        }

        self
    }

    pub fn check_cluster_assignments(mut self, assignments: &[ClusterAssignment]) -> Self {
        self.checked_cluster_assignments += assignments.len();
        let mut seen_case_ids = BTreeSet::<&str>::new();

        for assignment in assignments {
            if assignment.case_id.trim().is_empty() {
                self.push_error("missing_case_id", "assignment case_id is empty", None, None);
            } else if !seen_case_ids.insert(assignment.case_id.as_str()) {
                self.push_error(
                    "duplicate_cluster_assignment",
                    "case has more than one cluster assignment",
                    Some(assignment.case_id.clone()),
                    Some(assignment.trace_id.clone()),
                );
            }

            if assignment.trace_id.trim().is_empty() {
                self.push_error(
                    "missing_trace_id",
                    "assignment trace_id is empty",
                    Some(assignment.case_id.clone()),
                    None,
                );
            }

            if assignment.cluster_id.trim().is_empty() {
                self.push_error(
                    "missing_cluster_id",
                    "assignment cluster_id is empty",
                    Some(assignment.case_id.clone()),
                    Some(assignment.trace_id.clone()),
                );
            }

            if !unit_interval(assignment.confidence) {
                self.push_error(
                    "invalid_cluster_confidence",
                    "assignment confidence must be between 0.0 and 1.0",
                    Some(assignment.case_id.clone()),
                    Some(assignment.trace_id.clone()),
                );
            }

            if assignment
                .distance
                .is_some_and(|distance| !distance.is_finite())
            {
                self.push_error(
                    "invalid_cluster_distance",
                    "assignment distance must be finite",
                    Some(assignment.case_id.clone()),
                    Some(assignment.trace_id.clone()),
                );
            }
        }

        self
    }

    pub fn check_assignment_overlap(
        mut self,
        cases: &[EvalCase],
        assignments: &[ClusterAssignment],
    ) -> Self {
        let case_ids = cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<BTreeSet<_>>();

        for assignment in assignments {
            if !case_ids.contains(assignment.case_id.as_str()) {
                self.push_error(
                    "unknown_assignment_case",
                    "assignment case_id is not present in cases",
                    Some(assignment.case_id.clone()),
                    Some(assignment.trace_id.clone()),
                );
            }
        }

        self
    }

    pub fn finish(self) -> ValidationReport {
        ValidationReport {
            checked_cases: self.checked_cases,
            checked_results: self.checked_results,
            checked_embeddings: self.checked_embeddings,
            checked_cluster_models: self.checked_cluster_models,
            checked_cluster_assignments: self.checked_cluster_assignments,
            errors: self.errors,
            warnings: self.warnings,
        }
    }

    fn push_error(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
        case_id: Option<String>,
        trace_id: Option<String>,
    ) {
        self.push(ValidationSeverity::Error, code, message, case_id, trace_id);
    }

    fn push_warning(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
        case_id: Option<String>,
        trace_id: Option<String>,
    ) {
        self.push(
            ValidationSeverity::Warning,
            code,
            message,
            case_id,
            trace_id,
        );
    }

    fn push(
        &mut self,
        severity: ValidationSeverity,
        code: impl Into<String>,
        message: impl Into<String>,
        case_id: Option<String>,
        trace_id: Option<String>,
    ) {
        let issue = ValidationIssue {
            severity,
            code: code.into(),
            message: message.into(),
            case_id,
            trace_id,
        };

        match issue.severity {
            ValidationSeverity::Error => self.errors.push(issue),
            ValidationSeverity::Warning => self.warnings.push(issue),
        }
    }
}

pub fn validate_cases(cases: &[EvalCase]) -> ValidationReport {
    validate_cases_with_profile(cases, ValidationProfile::RunnableCases)
}

pub fn validate_cases_with_profile(
    cases: &[EvalCase],
    profile: ValidationProfile,
) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_cases_with_profile(cases, profile)
        .finish()
}

pub fn validate_results(results: &[EvaluationResult]) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_results(results)
        .finish()
}

pub fn validate_embeddings(embeddings: &[CaseEmbedding]) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_embeddings(embeddings)
        .finish()
}

pub fn validate_cases_and_embeddings(
    cases: &[EvalCase],
    embeddings: &[CaseEmbedding],
) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_cases_with_profile(cases, ValidationProfile::EmbeddingDataset)
        .check_embeddings(embeddings)
        .check_embedding_overlap(cases, embeddings, false)
        .finish()
}

pub fn validate_cluster_model(model: &ClusterModel) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_cluster_model(model)
        .finish()
}

pub fn validate_cluster_assignments(assignments: &[ClusterAssignment]) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_cluster_assignments(assignments)
        .finish()
}

pub fn validate_cases_and_cluster_assignments(
    cases: &[EvalCase],
    assignments: &[ClusterAssignment],
) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_cases_with_profile(cases, ValidationProfile::ClusterAssignments)
        .check_cluster_assignments(assignments)
        .check_assignment_overlap(cases, assignments)
        .finish()
}

pub fn validate_cases_and_results(
    cases: &[EvalCase],
    results: &[EvaluationResult],
) -> ValidationReport {
    validate_cases_and_results_with_profile(cases, results, ValidationProfile::RunnableCases)
}

pub fn validate_cases_and_results_with_profile(
    cases: &[EvalCase],
    results: &[EvaluationResult],
    case_profile: ValidationProfile,
) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_cases_with_profile(cases, case_profile)
        .check_results(results)
        .check_overlap(cases, results)
        .finish()
}

fn raw_score_in_scale(raw_score: f32, score_scale: ScoreScale) -> bool {
    match score_scale {
        ScoreScale::Binary | ScoreScale::Unit => unit_interval(raw_score),
        ScoreScale::FourPoint => (1.0..=4.0).contains(&raw_score),
    }
}

fn unit_interval(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::ClusterTextProjector;

    #[test]
    fn catches_missing_case_output_and_duplicate_ids() {
        let cases = vec![
            EvalCase::new("case-1", "trace-1", "input"),
            EvalCase::new("case-1", "trace-2", "input").with_actual_output("ok"),
        ];

        let report = validate_cases(&cases);

        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|issue| issue.code == "missing_actual_output")
        );
        assert!(
            report
                .errors
                .iter()
                .any(|issue| issue.code == "duplicate_case_id")
        );
    }

    #[test]
    fn draft_cases_warn_but_do_not_fail_for_missing_actual_output() {
        let cases = vec![EvalCase::new("case-1", "trace-1", "input")];

        let report = validate_cases_with_profile(&cases, ValidationProfile::DraftCases);

        assert!(report.is_valid());
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 1);
        assert_eq!(report.warnings[0].severity, ValidationSeverity::Warning);
        assert_eq!(report.warnings[0].code, "missing_actual_output");
    }

    #[test]
    fn runnable_cases_fail_for_missing_actual_output() {
        let cases = vec![EvalCase::new("case-1", "trace-1", "input")];

        let report = validate_cases_with_profile(&cases, ValidationProfile::RunnableCases);

        assert!(!report.is_valid());
        assert_eq!(report.error_count(), 1);
        assert_eq!(report.errors[0].severity, ValidationSeverity::Error);
        assert_eq!(report.errors[0].code, "missing_actual_output");
    }

    #[test]
    fn validates_result_score_ranges() {
        let result = EvaluationResult::from_ids(
            "case-1",
            "trace-1",
            "judge",
            5.0,
            ScoreScale::FourPoint,
            true,
            "bad score",
        );

        let report = validate_results(&[result]);

        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|issue| issue.code == "invalid_raw_score")
        );
    }

    #[test]
    fn validates_embedding_dataset_overlap_and_dimensions() {
        let case = EvalCase::new("case-1", "trace-1", "input");
        let projected = crate::clustering::DefaultClusterTextProjector::new().project_case(&case);
        let mut embedding =
            crate::clustering::CaseEmbedding::new(&projected, "test", "model", vec![0.1], "p");
        embedding.dimensions = 2;

        let report = validate_cases_and_embeddings(&[case], &[embedding]);

        assert!(!report.is_valid());
        assert_eq!(report.checked_embeddings, 1);
        assert!(
            report
                .errors
                .iter()
                .any(|issue| issue.code == "invalid_embedding")
        );
    }

    #[test]
    fn validates_cluster_assignment_overlap() {
        let known_case = EvalCase::new("case-1", "trace-1", "input");
        let unknown_case = EvalCase::new("case-2", "trace-2", "input");
        let assignment = ClusterAssignment::new(&unknown_case, "cluster-1", 1.0, "test");

        let report = validate_cases_and_cluster_assignments(&[known_case], &[assignment]);

        assert!(!report.is_valid());
        assert_eq!(report.checked_cluster_assignments, 1);
        assert!(
            report
                .errors
                .iter()
                .any(|issue| issue.code == "unknown_assignment_case")
        );
    }
}
