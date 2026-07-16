use super::*;
use crate::behavior::ToolUsageBudgetV1;
use crate::behavior::model::{FinalOutcome, StateChangeRef};

fn call(name: &str, status: ToolCallStatus, mutating: bool) -> ToolCallFact {
    ToolCallFact {
        call_id: format!("{name}-call"),
        tool_name: name.to_string(),
        tool_name_source_quality: FactQuality::Explicit,
        operation: Some(if mutating { "cancel" } else { "lookup" }.to_string()),
        operation_source_quality: crate::model::FactQuality::Explicit,
        invocation_fingerprint: Some(format!("sha256:{:064x}", 1)),
        invocation_fingerprint_quality: crate::model::FactQuality::Explicit,
        result_fingerprint: None,
        result_fingerprint_quality: crate::model::FactQuality::Missing,
        effect: if mutating {
            OperationEffect::Mutating
        } else {
            OperationEffect::ReadOnly
        },
        retry_safety: if mutating {
            RetrySafety::NonIdempotent
        } else {
            RetrySafety::Idempotent
        },
        requirement: ToolRequirement::Required,
        attempt: 1,
        started_at: "2026-07-10T12:00:00Z".to_string(),
        started_at_unix_nano: Some(1),
        duration_ms: 1,
        duration_nano: Some(1_000_000),
        status,
        status_quality: crate::model::FactQuality::Explicit,
        error: status
            .is_failure()
            .then(|| super::super::model::NormalizedToolError {
                kind: "timeout".to_string(),
                code: None,
                retryable: Some(true),
                redacted_message_hash: None,
            }),
        approval_required: false,
        approval_outcome: None,
        state_change: None,
        evidence: vec![EvidenceRef::span(format!("{name}-span"))],
    }
}

fn trace_with(calls: Vec<ToolCallFact>) -> AgentBehaviorTrace {
    AgentBehaviorTrace {
        tool_calls: calls,
        final_outcome: FinalOutcome {
            status: FinalOutcomeStatus::Incomplete,
            ..FinalOutcome::default()
        },
        ..AgentBehaviorTrace::new("trace-1")
    }
}

#[test]
fn coverage_aware_report_does_not_promote_missing_facts() {
    let mut trace = trace_with(vec![call("lookup", ToolCallStatus::Failed, false)]);
    trace.tool_calls[0].invocation_fingerprint = None;
    trace.tool_calls[0].invocation_fingerprint_quality = crate::model::FactQuality::Missing;

    let legacy_findings = DeterministicDetectorSet::default().detect(&trace);
    let report = DeterministicDetectorSet::default().detect_report(&trace);

    assert!(!legacy_findings.is_empty());
    assert!(report.findings.is_empty());
    assert_eq!(report.schema_version, DETECTION_REPORT_SCHEMA_VERSION);
    assert!(report.telemetry_diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "missing_invocation_fingerprint"
            || diagnostic.code == "detector_inconclusive_missing_facts"
    }));
}

#[test]
fn unrelated_success_without_requiredness_does_not_hide_terminal_failure() {
    let failed = call("required_lookup", ToolCallStatus::Failed, false);
    let mut unrelated_success = call("optional_cleanup", ToolCallStatus::Succeeded, false);
    unrelated_success.requirement = ToolRequirement::Unknown;
    let mut trace = trace_with(vec![failed, unrelated_success]);
    trace.coverage.final_outcome = FactQuality::Explicit;

    let report = DeterministicDetectorSet::default().detect_report(&trace);
    let coverage = &report.detector_coverage["terminal_tool_failure"];

    assert_eq!(coverage.status, DetectorEvaluationStatusV1::Evaluated);
    assert!(!coverage.missing_facts.contains("tool_requirement"));
    assert!(
        report
            .findings
            .iter()
            .any(|finding| finding.detector_id == "terminal_tool_failure")
    );
}

#[test]
fn recovered_idempotent_retry_does_not_fire_terminal_failure() {
    let failed = call("lookup", ToolCallStatus::Failed, false);
    let mut succeeded = call("lookup", ToolCallStatus::Succeeded, false);
    succeeded.attempt = 2;
    let trace = trace_with(vec![failed, succeeded]);

    assert_eq!(
        RecoveryAnalyzer.recovery_for_call(&trace, 0),
        RecoveryStatus::Recovered
    );
    assert!(TerminalToolFailureDetector.detect(&trace).is_empty());
}

#[test]
fn uncertain_mutation_with_verification_is_recovered() {
    let mut uncertain = call("cancel_card", ToolCallStatus::TimedOut, true);
    uncertain.state_change = Some(StateChangeRef {
        predicate: Some("card_cancelled".to_string()),
        observation: StateObservation::Ambiguous,
        artifact: EvidenceRef::new("state_change", "state:uncertain"),
    });
    let mut verify = call("verify_card", ToolCallStatus::Succeeded, false);
    verify.operation = Some("verify".to_string());
    verify.effect = OperationEffect::Verifying;
    verify.state_change = Some(StateChangeRef {
        predicate: Some("card_cancelled".to_string()),
        observation: StateObservation::VerifiedChanged,
        artifact: EvidenceRef::new("state_change", "state:verified"),
    });
    let mut trace = trace_with(vec![uncertain, verify]);
    trace.final_outcome.claims.push(OutcomeClaim {
        operation: Some("cancel".to_string()),
        call_id: None,
        status: ClaimedOutcomeStatus::Succeeded,
        evidence: vec![EvidenceRef::new("outcome_claim", "claim:cancel")],
    });

    assert_eq!(
        RecoveryAnalyzer.recovery_for_call(&trace, 0),
        RecoveryStatus::Recovered
    );
    assert!(UncertainMutationStateDetector.detect(&trace).is_empty());
    assert!(FalseSuccessClaimDetector.detect(&trace).is_empty());
}

#[test]
fn alternate_tool_reaching_same_state_predicate_recovers_failure() {
    let mut failed = call("primary", ToolCallStatus::Failed, true);
    failed.operation = Some("primary_update".to_string());
    failed.state_change = Some(StateChangeRef {
        predicate: Some("desired_state".to_string()),
        observation: StateObservation::Ambiguous,
        artifact: EvidenceRef::new("state_change", "state:ambiguous"),
    });
    let mut alternate = call("alternate", ToolCallStatus::Succeeded, true);
    alternate.operation = Some("alternate_update".to_string());
    alternate.state_change = Some(StateChangeRef {
        predicate: Some("desired_state".to_string()),
        observation: StateObservation::VerifiedChanged,
        artifact: EvidenceRef::new("state_change", "state:changed"),
    });
    let trace = trace_with(vec![failed, alternate]);

    assert_eq!(
        RecoveryAnalyzer.recovery_for_call(&trace, 0),
        RecoveryStatus::Recovered
    );
}

#[test]
fn finding_identity_is_stable() {
    let trace = trace_with(vec![call("cancel_card", ToolCallStatus::TimedOut, true)]);

    let first = TerminalToolFailureDetector.detect(&trace);
    let second = TerminalToolFailureDetector.detect(&trace);

    assert_eq!(first[0].finding_id, second[0].finding_id);
    assert_eq!(first[0].failure_signature, second[0].failure_signature);
}

#[test]
fn repeated_failures_and_call_loops_are_detected() {
    let calls = (0..4)
        .map(|attempt| {
            let mut call = call("lookup", ToolCallStatus::Failed, false);
            call.call_id = format!("lookup-{attempt}");
            call.attempt = attempt + 1;
            call.evidence = vec![EvidenceRef::span(format!("lookup-span-{attempt}"))];
            call
        })
        .collect();
    let trace = trace_with(calls);

    assert_eq!(
        RepeatedToolFailureDetector::default().detect(&trace).len(),
        1
    );
    assert_eq!(ToolCallLoopDetector::default().detect(&trace).len(), 1);
}

#[test]
fn distinct_invocations_do_not_form_a_repeated_failure_or_loop() {
    let calls = (0..100)
        .map(|attempt| {
            let mut call = call("bash", ToolCallStatus::Failed, false);
            call.call_id = format!("bash-{attempt}");
            call.attempt = attempt + 1;
            call.invocation_fingerprint = Some(format!("sha256:{attempt:064x}"));
            call.evidence = vec![EvidenceRef::span(format!("bash-span-{attempt}"))];
            call
        })
        .collect();
    let trace = trace_with(calls);

    assert!(
        RepeatedToolFailureDetector::default()
            .detect(&trace)
            .is_empty()
    );
    assert!(ToolCallLoopDetector::default().detect(&trace).is_empty());
}

#[test]
fn progress_and_changed_errors_break_repeated_failure_episodes() {
    let mut calls = Vec::new();
    for attempt in 0..2 {
        let mut failed = call("lookup", ToolCallStatus::Failed, false);
        failed.call_id = format!("before-progress-{attempt}");
        calls.push(failed);
    }
    let mut progress = call("inspect", ToolCallStatus::Succeeded, false);
    progress.operation = Some("inspect".into());
    progress.invocation_fingerprint = Some(format!("sha256:{:064x}", 2));
    calls.push(progress);
    for attempt in 0..2 {
        let mut failed = call("lookup", ToolCallStatus::Failed, false);
        failed.call_id = format!("after-progress-{attempt}");
        calls.push(failed);
    }
    let trace = trace_with(calls);
    assert!(
        RepeatedToolFailureDetector::default()
            .detect(&trace)
            .is_empty()
    );

    let changed_errors = (0..4)
        .map(|attempt| {
            let mut failed = call("lookup", ToolCallStatus::Failed, false);
            failed.call_id = format!("changed-error-{attempt}");
            failed.error.as_mut().unwrap().kind = format!("error-{attempt}");
            failed
        })
        .collect();
    assert!(
        RepeatedToolFailureDetector::default()
            .detect(&trace_with(changed_errors))
            .is_empty()
    );
}

#[test]
fn repeated_failure_episode_reports_later_compatible_recovery() {
    let mut calls = (0..3)
        .map(|attempt| {
            let mut failed = call("lookup", ToolCallStatus::Failed, false);
            failed.call_id = format!("failed-{attempt}");
            failed.attempt = attempt + 1;
            failed
        })
        .collect::<Vec<_>>();
    let mut recovered = call("lookup", ToolCallStatus::Succeeded, false);
    recovered.call_id = "recovered".into();
    recovered.attempt = 4;
    calls.push(recovered);

    let findings = RepeatedToolFailureDetector::default().detect(&trace_with(calls));
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].recovery, RecoveryStatus::Recovered);
}

#[test]
fn default_profile_treats_missing_terminal_outcome_as_diagnostic_only() {
    let trace = AgentBehaviorTrace::new("missing-terminal");
    let report = DeterministicDetectorSet::default().detect_report(&trace);

    assert!(report.findings.is_empty());
    assert!(
        report
            .telemetry_diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "missing_final_outcome")
    );
    assert!(!report.detector_versions.contains_key("missing_resolution"));
    assert!(
        !report
            .detector_versions
            .contains_key("repeated_tool_failure")
    );
    assert!(!report.detector_versions.contains_key("tool_call_loop"));
    assert!(
        !report
            .detector_versions
            .contains_key("excessive_tool_usage")
    );
}

#[test]
fn versioned_profiles_require_protocols_and_explicit_tool_budgets() {
    let mut invalid_missing = DetectorProfileV1::conservative();
    invalid_missing
        .enabled_detectors
        .insert("missing_resolution".into());
    assert!(DeterministicDetectorSet::from_profile(invalid_missing).is_err());

    let coding = DetectorProfileV1::coding_agent();
    let coding_set = DeterministicDetectorSet::from_profile(coding.clone()).unwrap();
    assert_eq!(coding_set.profile(), &coding.identity);
    assert!(
        coding_set
            .detector_versions()
            .contains_key("missing_resolution")
    );

    let mut invalid_budget = DetectorProfileV1::conservative();
    invalid_budget
        .enabled_detectors
        .insert("excessive_tool_usage".into());
    assert!(DeterministicDetectorSet::from_profile(invalid_budget).is_err());

    let mut budgeted = DetectorProfileV1::conservative();
    budgeted.identity.profile_id = "test.budgeted".into();
    budgeted
        .enabled_detectors
        .insert("excessive_tool_usage".into());
    budgeted.tool_usage_budget = Some(ToolUsageBudgetV1 {
        maximum_calls: 10,
        maximum_total_duration_ms: 1_000,
    });
    let budgeted_set = DeterministicDetectorSet::from_profile(budgeted).unwrap();
    assert!(
        budgeted_set
            .detector_versions()
            .contains_key("excessive_tool_usage")
    );
}

#[test]
fn repeated_failure_labeled_fixture_matrix_has_no_false_positives_or_false_negatives() {
    struct Case {
        name: &'static str,
        calls: Vec<ToolCallFact>,
        expected: bool,
    }

    let identical_failures = |count: u32| {
        (0..count)
            .map(|attempt| {
                let mut call = call("bash", ToolCallStatus::Failed, false);
                call.call_id = format!("same-{attempt}");
                call.attempt = attempt + 1;
                call
            })
            .collect::<Vec<_>>()
    };
    let mut distinct = identical_failures(4);
    for (index, call) in distinct.iter_mut().enumerate() {
        call.invocation_fingerprint = Some(format!("sha256:{index:064x}"));
    }
    let mut missing_identity = identical_failures(4);
    for call in &mut missing_identity {
        call.invocation_fingerprint = None;
        call.invocation_fingerprint_quality = FactQuality::Missing;
    }
    let mut recovered = identical_failures(3);
    recovered.push(call("bash", ToolCallStatus::Succeeded, false));
    let cases = vec![
        Case {
            name: "three identical failures",
            calls: identical_failures(3),
            expected: true,
        },
        Case {
            name: "four identical failures",
            calls: identical_failures(4),
            expected: true,
        },
        Case {
            name: "successful recovery after episode",
            calls: recovered,
            expected: true,
        },
        Case {
            name: "below threshold",
            calls: identical_failures(2),
            expected: false,
        },
        Case {
            name: "distinct invocations",
            calls: distinct,
            expected: false,
        },
        Case {
            name: "missing invocation identity",
            calls: missing_identity,
            expected: false,
        },
        Case {
            name: "ordinary success",
            calls: vec![call("bash", ToolCallStatus::Succeeded, false)],
            expected: false,
        },
        Case {
            name: "empty trace",
            calls: Vec::new(),
            expected: false,
        },
    ];
    let detector = RepeatedToolFailureDetector::default();
    let mut true_positive = 0usize;
    let mut false_positive = 0usize;
    let mut false_negative = 0usize;
    for case in cases {
        let observed = !detector.detect(&trace_with(case.calls)).is_empty();
        match (case.expected, observed) {
            (true, true) => true_positive += 1,
            (false, true) => false_positive += 1,
            (true, false) => false_negative += 1,
            (false, false) => {}
        }
        assert_eq!(observed, case.expected, "fixture: {}", case.name);
    }
    let precision = true_positive as f32 / (true_positive + false_positive) as f32;
    let recall = true_positive as f32 / (true_positive + false_negative) as f32;
    assert_eq!(precision, 1.0);
    assert_eq!(recall, 1.0);
}

#[test]
fn call_loop_detects_a_repeated_multi_call_cycle() {
    let calls = (0..7)
        .map(|index| {
            let name = if index % 2 == 0 {
                "lookup_a"
            } else {
                "lookup_b"
            };
            let mut call = call(name, ToolCallStatus::Failed, false);
            call.operation = Some(name.to_string());
            call.call_id = format!("{name}-{index}");
            call.evidence = vec![EvidenceRef::span(format!("{name}-span-{index}"))];
            call
        })
        .collect();
    let trace = trace_with(calls);

    let findings = ToolCallLoopDetector::default().detect(&trace);

    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].metadata["operation"], "lookup_a");
    assert_eq!(findings[0].metadata["cycle_length"], 2);
}

#[test]
fn call_loop_does_not_merge_retries_with_distinct_intervening_work() {
    let mut calls = Vec::new();
    for index in 0..4 {
        let mut retry = call("bash", ToolCallStatus::Failed, false);
        retry.call_id = format!("bash-{index}");
        calls.push(retry);
        if index < 3 {
            let mut edit = call("editor", ToolCallStatus::Failed, false);
            edit.call_id = format!("editor-{index}");
            edit.operation = Some("edit".into());
            edit.invocation_fingerprint = Some(format!("sha256:{:064x}", index + 10));
            calls.push(edit);
        }
    }

    assert!(
        ToolCallLoopDetector::default()
            .detect(&trace_with(calls))
            .is_empty()
    );
}

#[test]
fn conservative_v2_detector_v6_agent_handoff_breaks_a_call_loop_episode() {
    let mut calls = Vec::new();
    for index in 0..2 {
        let mut retry = call("browser", ToolCallStatus::Failed, false);
        retry.call_id = format!("before-handoff-{index}");
        calls.push(retry);
    }
    let mut handoff = call("handoff", ToolCallStatus::Succeeded, false);
    handoff.call_id = "handoff-to-verifier".into();
    handoff.operation = Some("delegate_to_verifier".into());
    handoff.effect = OperationEffect::Escalating;
    handoff.invocation_fingerprint = Some(format!("sha256:{:064x}", 99));
    calls.push(handoff);
    for index in 0..2 {
        let mut retry = call("browser", ToolCallStatus::Failed, false);
        retry.call_id = format!("after-handoff-{index}");
        calls.push(retry);
    }

    let detector_set = DeterministicDetectorSet::default();
    assert_eq!(detector_set.profile().profile_id, "traceeval.conservative");
    assert_eq!(detector_set.profile().profile_version, "2");
    assert_eq!(ToolCallLoopDetector::default().version(), "6");
    assert!(
        ToolCallLoopDetector::default()
            .detect(&trace_with(calls))
            .is_empty()
    );
}

#[test]
fn unrelated_success_does_not_support_a_failed_mutation_claim() {
    let failed = call("cancel_card", ToolCallStatus::TimedOut, true);
    let mut unrelated = call("update_preferences", ToolCallStatus::Succeeded, true);
    unrelated.operation = Some("update".to_string());
    unrelated.state_change = Some(StateChangeRef {
        predicate: Some("preferences_updated".to_string()),
        observation: StateObservation::VerifiedChanged,
        artifact: EvidenceRef::new("state_change", "state:preferences"),
    });
    let mut trace = trace_with(vec![failed, unrelated]);
    trace.final_outcome.claims.push(OutcomeClaim {
        operation: Some("cancel".to_string()),
        call_id: None,
        status: ClaimedOutcomeStatus::Succeeded,
        evidence: vec![EvidenceRef::new("outcome_claim", "claim:cancel")],
    });

    assert_eq!(FalseSuccessClaimDetector.detect(&trace).len(), 1);
}

#[test]
fn authorization_policy_budget_and_escalation_detectors_fire() {
    let mut protected_call = call("update_account", ToolCallStatus::Succeeded, true);
    protected_call.approval_required = true;
    protected_call.duration_ms = 100;
    let mut trace = trace_with(vec![protected_call]);
    trace
        .policy_decisions
        .push(super::super::model::PolicyDecision {
            decision_id: "decision-1".to_string(),
            policy_id: Some("policy-1".to_string()),
            action: Some("update_account".to_string()),
            outcome: PolicyDecisionOutcome::Denied,
            reason_code: Some("not_allowed".to_string()),
            evidence: vec![EvidenceRef::span("policy-span")],
        });
    trace.final_outcome.escalation = EscalationStatus::RequiredAndMissing;

    assert_eq!(ApprovalBypassDetector.detect(&trace).len(), 1);
    assert_eq!(PolicyViolationDetector.detect(&trace).len(), 1);
    assert_eq!(
        ExcessiveToolUsageDetector::new(10, 50).detect(&trace).len(),
        1
    );
    assert_eq!(UnresolvedEscalationDetector.detect(&trace).len(), 1);
}

#[test]
fn denied_policy_without_matching_execution_is_not_a_violation() {
    let mut trace = trace_with(Vec::new());
    trace
        .policy_decisions
        .push(super::super::model::PolicyDecision {
            decision_id: "decision-blocked".into(),
            policy_id: Some("policy-1".into()),
            action: Some("delete_account".into()),
            outcome: PolicyDecisionOutcome::Denied,
            reason_code: Some("not_allowed".into()),
            evidence: vec![EvidenceRef::span("policy-denied")],
        });

    assert!(PolicyViolationDetector.detect(&trace).is_empty());

    let mut unrelated = call("update_account", ToolCallStatus::Succeeded, true);
    unrelated.operation = Some("update_preferences".into());
    trace.tool_calls.push(unrelated);
    assert!(PolicyViolationDetector.detect(&trace).is_empty());
}

#[test]
fn denied_policy_with_matching_successful_execution_is_a_violation() {
    let mut executed = call("delete_account", ToolCallStatus::Succeeded, true);
    executed.operation = Some("delete_account".into());
    let mut trace = trace_with(vec![executed]);
    trace
        .policy_decisions
        .push(super::super::model::PolicyDecision {
            decision_id: "decision-bypassed".into(),
            policy_id: Some("policy-1".into()),
            action: Some("delete_account".into()),
            outcome: PolicyDecisionOutcome::Denied,
            reason_code: Some("not_allowed".into()),
            evidence: vec![EvidenceRef::span("policy-denied")],
        });

    let findings = PolicyViolationDetector.detect(&trace);
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].evidence.len(), 2);
    assert_eq!(
        findings[0].metadata["executed_call_ids"],
        json!(["delete_account-call"])
    );
}
