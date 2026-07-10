use std::collections::BTreeMap;

use super::*;
use crate::TraceEvalError;
use crate::behavior::{
    BEHAVIOR_FINDING_SCHEMA_VERSION, BehaviorFinding, EvidenceRef, FindingSeverity, RecoveryStatus,
};
use crate::evaluation::ScoreScale;

fn finding(id: &str, signature: &str, severity: FindingSeverity) -> BehaviorFinding {
    BehaviorFinding {
        schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
        finding_id: id.to_string(),
        detector_id: "detector".to_string(),
        detector_version: "2".to_string(),
        trace_id: "trace".to_string(),
        kind: "test".to_string(),
        severity,
        recovery: RecoveryStatus::Unrecovered,
        confidence: Some(1.0),
        failure_signature: signature.to_string(),
        evidence: Vec::new(),
        created_at: "2026-07-10T12:00:00Z".to_string(),
        metadata: BTreeMap::new(),
    }
}

fn result(
    case_id: &str,
    evaluator_name: &str,
    evaluator_version: Option<&str>,
    score: f32,
    passed: bool,
) -> EvaluationResult {
    let mut result = EvaluationResult::from_ids(
        case_id,
        format!("trace-{case_id}"),
        evaluator_name,
        score,
        ScoreScale::Unit,
        passed,
        "test evaluation",
    );
    if let Some(version) = evaluator_version {
        result.metadata.insert(
            "evaluator_spec_hash".to_string(),
            serde_json::Value::String(version.to_string()),
        );
    }
    result
}

fn request() -> RemediationVerificationRequest {
    RemediationVerificationRequest {
        schema_version: REMEDIATION_VERIFICATION_REQUEST_SCHEMA_VERSION.to_string(),
        case_id: "finding-case".to_string(),
        target_failure_signatures: vec!["target".to_string()],
        incident_case_id: "incident-case".to_string(),
        suite_case_ids: vec!["suite-case".to_string()],
        severe_threshold: FindingSeverity::High,
        policy_gate: VerificationGate::passed(vec![EvidenceRef::new(
            "policy_report",
            "sha256:policy",
        )]),
        approval_gate: VerificationGate::passed(vec![EvidenceRef::new(
            "approval_record",
            "approval:1",
        )]),
        baseline_budget: ExecutionBudget {
            tool_call_count: 3,
            latency_ms: 100,
            cost_microunits: 20,
        },
        candidate_budget: ExecutionBudget {
            tool_call_count: 3,
            latency_ms: 100,
            cost_microunits: 20,
        },
        input_artifacts: Some(input_artifacts()),
        policy: RemediationVerificationPolicy::strict(),
    }
}

fn input_artifacts() -> RemediationInputArtifacts {
    let digest = |record_count| VerificationArtifactDigest {
        content_hash: format!("sha256:{}", "0".repeat(64)),
        byte_count: 10,
        record_count,
    };
    RemediationInputArtifacts {
        baseline_findings: digest(1),
        candidate_findings: digest(0),
        baseline_results: digest(2),
        candidate_results: digest(2),
    }
}

fn passing_results() -> (Vec<EvaluationResult>, Vec<EvaluationResult>) {
    (
        vec![
            result("incident-case", "incident-grader", Some("v1"), 0.0, false),
            result("suite-case", "suite-grader", Some("v1"), 1.0, true),
        ],
        vec![
            result("incident-case", "incident-grader", Some("v1"), 1.0, true),
            result("suite-case", "suite-grader", Some("v1"), 1.0, true),
        ],
    )
}

#[test]
fn passes_only_when_baseline_reproduces_and_candidate_resolves_target() {
    let baseline = vec![finding("baseline", "target", FindingSeverity::High)];
    let candidate = vec![finding("candidate-low", "novel-low", FindingSeverity::Low)];

    let report = PairedFindingVerifier::default().verify(
        "case-1",
        ["target".to_string()],
        &baseline,
        &candidate,
    );

    assert!(report.baseline_reproduced);
    assert!(report.candidate_resolved);
    assert!(report.no_severe_novel_findings);
    assert!(report.finding_gate_passed);
    assert!(report.verification_id.starts_with("sha256:"));
}

#[test]
fn fails_for_recurring_target_or_severe_novel_finding() {
    let baseline = vec![finding("baseline", "target", FindingSeverity::High)];
    let candidate = vec![
        finding("recurring", "target", FindingSeverity::High),
        finding("novel", "novel-high", FindingSeverity::Critical),
    ];

    let report = PairedFindingVerifier::default().verify(
        "case-1",
        ["target".to_string()],
        &baseline,
        &candidate,
    );

    assert!(!report.finding_gate_passed);
    assert_eq!(report.recurring_target_signatures, ["target"]);
    assert_eq!(report.severe_novel_finding_ids, ["novel"]);
    assert_eq!(report.reasons.len(), 2);
}

#[test]
fn combined_verification_passes_only_with_every_stage_ten_gate() {
    let baseline_findings = vec![finding("baseline", "target", FindingSeverity::High)];
    let (baseline_results, candidate_results) = passing_results();

    let report = RemediationVerifier::new()
        .verify_request(
            request(),
            &baseline_findings,
            &[],
            &baseline_results,
            &candidate_results,
        )
        .unwrap();

    assert!(report.passed);
    assert!(report.finding_gate.finding_gate_passed);
    assert!(report.incident_regression_gate.passed);
    assert!(report.suite_regression_gate.passed);
    assert!(report.budget_gate.passed);
    assert_eq!(report.incident_regression_gate.paired_result_count, 1);
    assert_eq!(report.suite_regression_gate.paired_result_count, 1);
    assert!(report.verification_id.starts_with("sha256:"));
}

#[test]
fn evaluator_version_changes_are_not_treated_as_paired_results() {
    let baseline_findings = vec![finding("baseline", "target", FindingSeverity::High)];
    let (baseline_results, mut candidate_results) = passing_results();
    candidate_results[0].metadata.insert(
        "evaluator_spec_hash".to_string(),
        serde_json::Value::String("v2".to_string()),
    );

    let report = RemediationVerifier::new()
        .verify_request(
            request(),
            &baseline_findings,
            &[],
            &baseline_results,
            &candidate_results,
        )
        .unwrap();

    assert!(!report.passed);
    assert!(!report.incident_regression_gate.passed);
    assert_eq!(
        report.incident_regression_gate.missing_candidate_results[0]
            .evaluator_version
            .as_deref(),
        Some("v1")
    );
    assert_eq!(
        report.incident_regression_gate.unexpected_candidate_results[0]
            .evaluator_version
            .as_deref(),
        Some("v2")
    );
}

#[test]
fn equal_aggregate_pass_rate_cannot_hide_a_new_suite_failure() {
    let baseline_findings = vec![finding("baseline", "target", FindingSeverity::High)];
    let mut request = request();
    request.suite_case_ids = vec!["suite-a".to_string(), "suite-b".to_string()];
    request.policy.max_suite_score_drop = 1.0;
    request
        .input_artifacts
        .as_mut()
        .unwrap()
        .baseline_results
        .record_count = 3;
    request
        .input_artifacts
        .as_mut()
        .unwrap()
        .candidate_results
        .record_count = 3;
    let baseline_results = vec![
        result("incident-case", "incident-grader", Some("v1"), 0.0, false),
        result("suite-a", "suite-grader", Some("v1"), 1.0, true),
        result("suite-b", "suite-grader", Some("v1"), 0.0, false),
    ];
    let candidate_results = vec![
        result("incident-case", "incident-grader", Some("v1"), 1.0, true),
        result("suite-a", "suite-grader", Some("v1"), 0.0, false),
        result("suite-b", "suite-grader", Some("v1"), 1.0, true),
    ];

    let report = RemediationVerifier::new()
        .verify_request(
            request,
            &baseline_findings,
            &[],
            &baseline_results,
            &candidate_results,
        )
        .unwrap();

    assert!(!report.passed);
    assert_eq!(report.suite_regression_gate.new_failure_results.len(), 1);
    assert_eq!(
        report.suite_regression_gate.new_failure_results[0].case_id,
        "suite-a"
    );
}

#[test]
fn missing_evidence_and_budget_regression_fail_closed() {
    let baseline_findings = vec![finding("baseline", "target", FindingSeverity::High)];
    let (baseline_results, candidate_results) = passing_results();
    let mut request = request();
    request.policy_gate = VerificationGate::passed(Vec::new());
    request.approval_gate = VerificationGate::missing("approval not recorded");
    request.candidate_budget.tool_call_count = 4;

    let report = RemediationVerifier::new()
        .verify_request(
            request,
            &baseline_findings,
            &[],
            &baseline_results,
            &candidate_results,
        )
        .unwrap();

    assert!(!report.passed);
    assert!(!report.policy_gate.passed_gate());
    assert!(!report.approval_gate.passed_gate());
    assert!(!report.budget_gate.passed);
    assert_eq!(report.reasons.len(), 3);
}

#[test]
fn request_validation_rejects_unknown_schema_and_non_finite_tolerance() {
    let mut invalid_schema = request();
    invalid_schema.schema_version = "unknown".to_string();
    assert!(matches!(
        invalid_schema.validate(),
        Err(TraceEvalError::InvalidRemediationVerificationRequest { .. })
    ));

    let mut invalid_score = request();
    invalid_score.policy.max_suite_score_drop = f32::NAN;
    assert!(matches!(
        invalid_score.validate(),
        Err(TraceEvalError::InvalidRemediationVerificationRequest { .. })
    ));

    let mut missing_artifacts = request();
    missing_artifacts.input_artifacts = None;
    assert!(matches!(
        missing_artifacts.validate(),
        Err(TraceEvalError::InvalidRemediationVerificationRequest { .. })
    ));
}

#[test]
fn request_record_counts_must_match_supplied_artifacts() {
    let baseline_findings = vec![finding("baseline", "target", FindingSeverity::High)];
    let (baseline_results, candidate_results) = passing_results();
    let mut request = request();
    request
        .input_artifacts
        .as_mut()
        .unwrap()
        .baseline_findings
        .record_count = 2;

    let result = RemediationVerifier::new().verify_request(
        request,
        &baseline_findings,
        &[],
        &baseline_results,
        &candidate_results,
    );

    assert!(matches!(
        result,
        Err(TraceEvalError::InvalidRemediationVerificationRequest { .. })
    ));
}
