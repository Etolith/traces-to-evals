use std::collections::BTreeSet;

use crate::clustering::{CaseEmbedding, ClusterAssignment, ClusterModel};
use crate::evaluation::EvaluationResult;
use crate::model::EvalCase;

use super::scalars::{raw_score_in_scale, unit_interval};
use super::{ValidationIssue, ValidationProfile, ValidationReport, ValidationSeverity};

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
                if label.description.trim().is_empty() {
                    self.push_error(
                        "invalid_cluster_label",
                        format!("cluster {} description is empty", cluster.id),
                        None,
                        None,
                    );
                }
                if label.description.chars().count() > 600 {
                    self.push_error(
                        "invalid_cluster_label",
                        format!("cluster {} description exceeds 600 characters", cluster.id),
                        None,
                        None,
                    );
                }
                if label
                    .known_failure_modes
                    .iter()
                    .any(|mode| mode.trim().is_empty())
                {
                    self.push_error(
                        "invalid_cluster_label",
                        format!("cluster {} has an empty failure mode", cluster.id),
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
