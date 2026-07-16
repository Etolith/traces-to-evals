use std::collections::BTreeMap;

use super::*;
use crate::behavior::{BEHAVIOR_FINDING_SCHEMA_VERSION, RecoveryStatus};

fn finding(id: &str, trace: &str, signature: &str, severity: FindingSeverity) -> BehaviorFinding {
    BehaviorFinding {
        schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
        finding_id: id.to_string(),
        detector_id: "detector".to_string(),
        detector_version: "2".to_string(),
        trace_id: trace.to_string(),
        kind: "test".to_string(),
        severity,
        recovery: RecoveryStatus::Unrecovered,
        confidence: Some(1.0),
        certainty: crate::behavior::FindingCertaintyV1::default(),
        failure_signature: signature.to_string(),
        evidence: Vec::new(),
        created_at: "2026-07-10T12:00:00Z".to_string(),
        metadata: BTreeMap::new(),
    }
}

#[test]
fn compares_affected_trace_rates_without_claiming_sampled_prevalence() {
    let baseline = vec![
        finding("b1", "trace-1", "target", FindingSeverity::High),
        finding("b2", "trace-2", "target", FindingSeverity::High),
    ];
    let candidate = vec![finding("c1", "trace-3", "target", FindingSeverity::High)];

    let report = FindingRecurrenceComparator::default()
        .compare(
            ["target".to_string()],
            FindingWindow {
                window_id: "baseline".to_string(),
                observed_trace_count: 10,
                population_basis: PopulationBasis::Sampled,
            },
            &baseline,
            FindingWindow {
                window_id: "canary".to_string(),
                observed_trace_count: 10,
                population_basis: PopulationBasis::Sampled,
            },
            &candidate,
        )
        .unwrap();

    assert_eq!(report.baseline_recurrence_rate, Some(0.2));
    assert_eq!(report.candidate_recurrence_rate, Some(0.1));
    assert_eq!(report.recurrence_rate_ratio, Some(0.5));
    assert!(report.interpretation.contains("sampled traces"));
}

#[test]
fn reports_severe_novel_candidate_findings() {
    let baseline = vec![finding("b1", "trace-1", "target", FindingSeverity::High)];
    let candidate = vec![finding("c1", "trace-2", "novel", FindingSeverity::Critical)];

    let report = FindingRecurrenceComparator::default()
        .compare(
            ["target".to_string()],
            FindingWindow {
                window_id: "baseline".to_string(),
                observed_trace_count: 1,
                population_basis: PopulationBasis::Exact,
            },
            &baseline,
            FindingWindow {
                window_id: "canary".to_string(),
                observed_trace_count: 1,
                population_basis: PopulationBasis::Exact,
            },
            &candidate,
        )
        .unwrap();

    assert_eq!(report.severe_novel_finding_ids, ["c1"]);
}

#[test]
fn rejects_more_affected_traces_than_the_window_observed() {
    let baseline = vec![
        finding("b1", "trace-1", "target", FindingSeverity::High),
        finding("b2", "trace-2", "target", FindingSeverity::High),
    ];

    let error = FindingRecurrenceComparator::new()
        .compare(
            ["target".to_string()],
            FindingWindow {
                window_id: "baseline".to_string(),
                observed_trace_count: 1,
                population_basis: PopulationBasis::Exact,
            },
            &baseline,
            FindingWindow {
                window_id: "candidate".to_string(),
                observed_trace_count: 1,
                population_basis: PopulationBasis::Exact,
            },
            &[],
        )
        .unwrap_err();

    assert!(matches!(
        error,
        TraceEvalError::InvalidFindingRecurrenceRequest { .. }
    ));
}

#[test]
fn missing_population_evidence_is_explicit_and_not_success() {
    let report = FindingRecurrenceComparator::new()
        .compare(
            ["target".to_string()],
            FindingWindow {
                window_id: "baseline".to_string(),
                observed_trace_count: 0,
                population_basis: PopulationBasis::Unknown,
            },
            &[],
            FindingWindow {
                window_id: "candidate".to_string(),
                observed_trace_count: 0,
                population_basis: PopulationBasis::Unknown,
            },
            &[],
        )
        .unwrap();

    assert!(!report.evidence_complete);
    assert_eq!(report.evidence_gaps.len(), 4);
    assert_eq!(report.baseline_recurrence_rate, None);
    assert!(report.interpretation.contains("must pause"));
}

#[test]
fn comparison_identity_includes_window_population_and_counts() {
    let baseline = vec![finding("b1", "trace-1", "target", FindingSeverity::High)];
    let compare = |observed_trace_count| {
        FindingRecurrenceComparator::new()
            .compare(
                ["target".to_string()],
                FindingWindow {
                    window_id: "baseline".to_string(),
                    observed_trace_count,
                    population_basis: PopulationBasis::Sampled,
                },
                &baseline,
                FindingWindow {
                    window_id: "candidate".to_string(),
                    observed_trace_count,
                    population_basis: PopulationBasis::Sampled,
                },
                &[],
            )
            .unwrap()
            .comparison_id
    };

    assert_ne!(compare(1), compare(2));
}

#[test]
fn duplicate_delivery_is_deduplicated_but_conflicting_records_fail() {
    let baseline_finding = finding("b1", "trace-1", "target", FindingSeverity::High);
    let duplicate = baseline_finding.clone();
    let report = FindingRecurrenceComparator::new()
        .compare(
            ["target".to_string()],
            FindingWindow {
                window_id: "baseline".to_string(),
                observed_trace_count: 1,
                population_basis: PopulationBasis::Exact,
            },
            &[baseline_finding.clone(), duplicate],
            FindingWindow {
                window_id: "candidate".to_string(),
                observed_trace_count: 1,
                population_basis: PopulationBasis::Exact,
            },
            &[],
        )
        .unwrap();
    assert_eq!(report.baseline_occurrence_count, 1);
    assert_eq!(report.baseline_finding_rates_by_kind[0].occurrence_count, 1);

    let mut conflicting = baseline_finding.clone();
    conflicting.trace_id = "different-trace".to_string();
    let error = FindingRecurrenceComparator::new()
        .compare(
            ["target".to_string()],
            FindingWindow {
                window_id: "baseline".to_string(),
                observed_trace_count: 2,
                population_basis: PopulationBasis::Exact,
            },
            &[baseline_finding, conflicting],
            FindingWindow {
                window_id: "candidate".to_string(),
                observed_trace_count: 1,
                population_basis: PopulationBasis::Exact,
            },
            &[],
        )
        .unwrap_err();
    assert!(matches!(
        error,
        TraceEvalError::InvalidFindingRecurrenceRequest { .. }
    ));
}
