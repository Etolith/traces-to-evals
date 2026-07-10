use super::*;
use crate::behavior::model::{FinalOutcome, StateChangeRef};

fn call(name: &str, status: ToolCallStatus, mutating: bool) -> ToolCallFact {
    ToolCallFact {
        call_id: format!("{name}-call"),
        tool_name: name.to_string(),
        operation: Some(if mutating { "cancel" } else { "lookup" }.to_string()),
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
        duration_ms: 1,
        status,
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
    let uncertain = call("cancel_card", ToolCallStatus::TimedOut, true);
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
fn call_loop_detects_interleaved_equivalent_calls_without_progress() {
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
