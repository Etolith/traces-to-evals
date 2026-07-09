mod builder;
mod report;
mod scalars;

use crate::clustering::{CaseEmbedding, ClusterAssignment, ClusterModel};
use crate::evaluation::EvaluationResult;
use crate::model::EvalCase;

pub use builder::ValidationReportBuilder;
pub use report::{ValidationIssue, ValidationProfile, ValidationReport, ValidationSeverity};

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
            crate::evaluation::ScoreScale::FourPoint,
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

    #[test]
    fn validates_cluster_label_constraints() {
        let cluster = crate::clustering::DiscoveredCluster {
            label: Some(crate::clustering::ClusterLabel {
                label: "x".repeat(81),
                description: String::new(),
                suggested_rubric: None,
                known_failure_modes: vec![" ".to_string()],
                confidence: 1.2,
                metadata: Default::default(),
                needs_review: true,
            }),
            ..crate::clustering::DiscoveredCluster::new("cluster-1", 1, vec!["case-1".to_string()])
        };
        let model = crate::clustering::ClusterModel::new(
            "model-1",
            "2026-01-01T00:00:00Z",
            crate::clustering::ClusterModelSource {
                case_count: 1,
                embedding_provider: None,
                embedding_model: None,
                embedding_dimensions: None,
                projection_version: None,
                algorithm: "manual".to_string(),
                distance_metric: "cosine".to_string(),
                random_seed: 42,
            },
            vec![cluster],
            Vec::new(),
            crate::clustering::ClusterQualityReport {
                cluster_count: 1,
                assigned_case_count: 0,
                mean_distance: None,
                silhouette_score: None,
                clusters: Vec::new(),
            },
        );

        let report = validate_cluster_model(&model);

        assert!(
            report
                .errors
                .iter()
                .filter(|issue| issue.code == "invalid_cluster_label")
                .count()
                >= 3
        );
        assert!(
            report
                .errors
                .iter()
                .any(|issue| issue.code == "invalid_cluster_label_confidence")
        );
    }
}
