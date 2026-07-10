use std::collections::BTreeMap;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::model::{
    AgentBehaviorTrace, ApprovalOutcome, BEHAVIOR_FINDING_SCHEMA_VERSION, BehaviorFinding,
    EvidenceRef, FinalOutcomeStatus, FindingSeverity, PolicyDecisionOutcome, RecoveryStatus,
    ToolCallFact, ToolCallStatus,
};

pub const DETERMINISTIC_DETECTOR_VERSION: &str = "1";

pub trait TraceDetector: Send + Sync {
    fn id(&self) -> &str;
    fn version(&self) -> &str;
    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RecoveryAnalyzer;

impl RecoveryAnalyzer {
    pub fn recovery_for_call(
        &self,
        trace: &AgentBehaviorTrace,
        call_index: usize,
    ) -> RecoveryStatus {
        let Some(call) = trace.tool_calls.get(call_index) else {
            return RecoveryStatus::Unknown;
        };
        if call.status == ToolCallStatus::Succeeded {
            return RecoveryStatus::Recovered;
        }

        let later_calls = &trace.tool_calls[call_index.saturating_add(1)..];
        let equivalent_success = later_calls.iter().any(|later| {
            equivalent_call(call, later)
                && later.status == ToolCallStatus::Succeeded
                && (call.idempotent == Some(true) || !call.mutating)
        });
        if equivalent_success {
            return RecoveryStatus::Recovered;
        }

        if call.mutating
            && later_calls.iter().any(|later| {
                later.status == ToolCallStatus::Succeeded
                    && later
                        .state_change
                        .as_ref()
                        .is_some_and(|state| state.verified)
            })
        {
            return RecoveryStatus::Recovered;
        }

        if trace.final_outcome.escalation_performed
            && trace.final_outcome.status == FinalOutcomeStatus::Escalated
        {
            return RecoveryStatus::Recovered;
        }
        if trace.final_outcome.failure_acknowledged
            && !trace.final_outcome.claimed_success
            && matches!(
                trace.final_outcome.status,
                FinalOutcomeStatus::Failed
                    | FinalOutcomeStatus::Refused
                    | FinalOutcomeStatus::Escalated
            )
        {
            return RecoveryStatus::Recovered;
        }

        if call.status == ToolCallStatus::Unknown {
            RecoveryStatus::Unknown
        } else {
            RecoveryStatus::Unrecovered
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct TerminalToolFailureDetector;

impl TraceDetector for TerminalToolFailureDetector {
    fn id(&self) -> &str {
        "terminal_tool_failure"
    }

    fn version(&self) -> &str {
        DETERMINISTIC_DETECTOR_VERSION
    }

    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        let recovery = RecoveryAnalyzer;
        trace
            .tool_calls
            .iter()
            .enumerate()
            .filter(|(index, call)| {
                call.required
                    && call.status.is_failure()
                    && recovery.recovery_for_call(trace, *index) == RecoveryStatus::Unrecovered
            })
            .map(|(_, call)| {
                finding_for_call(
                    trace,
                    self,
                    FindingSeverity::High,
                    RecoveryStatus::Unrecovered,
                    call,
                    BTreeMap::new(),
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RepeatedToolFailureDetector {
    max_failures: usize,
}

impl Default for RepeatedToolFailureDetector {
    fn default() -> Self {
        Self { max_failures: 3 }
    }
}

impl RepeatedToolFailureDetector {
    pub fn new(max_failures: usize) -> Self {
        Self {
            max_failures: max_failures.max(1),
        }
    }
}

impl TraceDetector for RepeatedToolFailureDetector {
    fn id(&self) -> &str {
        "repeated_tool_failure"
    }

    fn version(&self) -> &str {
        DETERMINISTIC_DETECTOR_VERSION
    }

    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        let mut groups: BTreeMap<(String, Option<String>), Vec<usize>> = BTreeMap::new();
        for (index, call) in trace.tool_calls.iter().enumerate() {
            if call.status.is_failure() {
                groups
                    .entry((call.tool_name.clone(), call.operation.clone()))
                    .or_default()
                    .push(index);
            }
        }

        groups
            .into_values()
            .filter(|indices| indices.len() >= self.max_failures)
            .map(|indices| {
                let last_index = *indices.last().expect("non-empty failure group");
                let call = &trace.tool_calls[last_index];
                let recovery = RecoveryAnalyzer.recovery_for_call(trace, last_index);
                let evidence = indices
                    .iter()
                    .flat_map(|index| trace.tool_calls[*index].evidence.clone())
                    .collect();
                let mut metadata = BTreeMap::new();
                metadata.insert("failure_count".to_string(), json!(indices.len()));
                metadata.insert("retry_limit".to_string(), json!(self.max_failures));
                build_finding(
                    trace,
                    self,
                    FindingSeverity::High,
                    recovery,
                    signature_subject(call),
                    error_kind(call),
                    evidence,
                    metadata,
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ToolCallLoopDetector {
    max_equivalent_calls: usize,
}

impl Default for ToolCallLoopDetector {
    fn default() -> Self {
        Self {
            max_equivalent_calls: 4,
        }
    }
}

impl ToolCallLoopDetector {
    pub fn new(max_equivalent_calls: usize) -> Self {
        Self {
            max_equivalent_calls: max_equivalent_calls.max(2),
        }
    }
}

impl TraceDetector for ToolCallLoopDetector {
    fn id(&self) -> &str {
        "tool_call_loop"
    }

    fn version(&self) -> &str {
        DETERMINISTIC_DETECTOR_VERSION
    }

    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        let mut findings = Vec::new();
        let mut start = 0;
        while start < trace.tool_calls.len() {
            let mut end = start + 1;
            while end < trace.tool_calls.len()
                && equivalent_call(&trace.tool_calls[start], &trace.tool_calls[end])
                && !has_material_progress(&trace.tool_calls[end - 1])
            {
                end += 1;
            }
            if end - start >= self.max_equivalent_calls {
                let calls = &trace.tool_calls[start..end];
                let last = calls.last().expect("non-empty loop");
                let evidence = calls
                    .iter()
                    .flat_map(|call| call.evidence.clone())
                    .collect();
                let mut metadata = BTreeMap::new();
                metadata.insert("call_count".to_string(), json!(calls.len()));
                metadata.insert(
                    "equivalent_call_limit".to_string(),
                    json!(self.max_equivalent_calls),
                );
                findings.push(build_finding(
                    trace,
                    self,
                    FindingSeverity::Medium,
                    RecoveryStatus::Unrecovered,
                    signature_subject(last),
                    error_kind(last),
                    evidence,
                    metadata,
                ));
            }
            start = end;
        }
        findings
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct UncertainMutationStateDetector;

impl TraceDetector for UncertainMutationStateDetector {
    fn id(&self) -> &str {
        "uncertain_mutation_state"
    }

    fn version(&self) -> &str {
        DETERMINISTIC_DETECTOR_VERSION
    }

    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        let analyzer = RecoveryAnalyzer;
        trace
            .tool_calls
            .iter()
            .enumerate()
            .filter(|(index, call)| {
                call.mutating
                    && matches!(
                        call.status,
                        ToolCallStatus::TimedOut | ToolCallStatus::Unknown
                    )
                    && analyzer.recovery_for_call(trace, *index) != RecoveryStatus::Recovered
            })
            .map(|(index, call)| {
                finding_for_call(
                    trace,
                    self,
                    FindingSeverity::High,
                    analyzer.recovery_for_call(trace, index),
                    call,
                    BTreeMap::new(),
                )
            })
            .collect()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FalseSuccessClaimDetector;

impl TraceDetector for FalseSuccessClaimDetector {
    fn id(&self) -> &str {
        "false_success_claim"
    }

    fn version(&self) -> &str {
        DETERMINISTIC_DETECTOR_VERSION
    }

    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        if !trace.final_outcome.claimed_success {
            return Vec::new();
        }
        let analyzer = RecoveryAnalyzer;
        let problematic_calls = trace
            .tool_calls
            .iter()
            .enumerate()
            .filter(|(index, call)| {
                call.mutating
                    && matches!(
                        call.status,
                        ToolCallStatus::Failed
                            | ToolCallStatus::TimedOut
                            | ToolCallStatus::Cancelled
                            | ToolCallStatus::Unknown
                    )
                    && analyzer.recovery_for_call(trace, *index) != RecoveryStatus::Recovered
            })
            .map(|(_, call)| call)
            .collect::<Vec<_>>();
        let supporting_state = trace.tool_calls.iter().any(|call| {
            call.status == ToolCallStatus::Succeeded
                && call
                    .state_change
                    .as_ref()
                    .is_some_and(|state| state.verified || (call.mutating && state.changed))
        });
        if problematic_calls.is_empty() && supporting_state {
            return Vec::new();
        }

        let mut evidence = trace.final_outcome.evidence.clone();
        evidence.extend(
            problematic_calls
                .iter()
                .flat_map(|call| call.evidence.clone()),
        );
        let subject = problematic_calls
            .last()
            .map(|call| signature_subject(call))
            .unwrap_or_else(|| ("final_outcome".to_string(), None));
        let error = problematic_calls.last().and_then(|call| error_kind(call));
        vec![build_finding(
            trace,
            self,
            FindingSeverity::High,
            RecoveryStatus::Unrecovered,
            subject,
            error,
            evidence,
            BTreeMap::new(),
        )]
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ApprovalBypassDetector;

impl TraceDetector for ApprovalBypassDetector {
    fn id(&self) -> &str {
        "approval_bypass"
    }

    fn version(&self) -> &str {
        DETERMINISTIC_DETECTOR_VERSION
    }

    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        trace
            .tool_calls
            .iter()
            .filter(|call| {
                call.mutating
                    && call.status == ToolCallStatus::Succeeded
                    && call.approval_required
                    && call.approval_outcome != Some(ApprovalOutcome::Approved)
            })
            .map(|call| {
                let mut metadata = BTreeMap::new();
                metadata.insert(
                    "approval_outcome".to_string(),
                    call.approval_outcome
                        .map(|outcome| json!(outcome))
                        .unwrap_or(Value::Null),
                );
                finding_for_call(
                    trace,
                    self,
                    FindingSeverity::Critical,
                    RecoveryStatus::Unrecovered,
                    call,
                    metadata,
                )
            })
            .collect()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PolicyViolationDetector;

impl TraceDetector for PolicyViolationDetector {
    fn id(&self) -> &str {
        "policy_violation"
    }

    fn version(&self) -> &str {
        DETERMINISTIC_DETECTOR_VERSION
    }

    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        trace
            .policy_decisions
            .iter()
            .filter(|decision| decision.outcome == PolicyDecisionOutcome::Denied)
            .map(|decision| {
                let mut metadata = BTreeMap::new();
                if let Some(policy_id) = &decision.policy_id {
                    metadata.insert("policy_id".to_string(), json!(policy_id));
                }
                if let Some(reason_code) = &decision.reason_code {
                    metadata.insert("reason_code".to_string(), json!(reason_code));
                }
                build_finding(
                    trace,
                    self,
                    FindingSeverity::High,
                    RecoveryStatus::Unrecovered,
                    (
                        decision
                            .action
                            .clone()
                            .unwrap_or_else(|| "policy".to_string()),
                        None,
                    ),
                    decision.reason_code.clone(),
                    decision.evidence.clone(),
                    metadata,
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExcessiveToolUsageDetector {
    max_tool_calls: usize,
    max_total_duration_ms: u64,
}

impl Default for ExcessiveToolUsageDetector {
    fn default() -> Self {
        Self {
            max_tool_calls: 25,
            max_total_duration_ms: 60_000,
        }
    }
}

impl ExcessiveToolUsageDetector {
    pub fn new(max_tool_calls: usize, max_total_duration_ms: u64) -> Self {
        Self {
            max_tool_calls: max_tool_calls.max(1),
            max_total_duration_ms: max_total_duration_ms.max(1),
        }
    }
}

impl TraceDetector for ExcessiveToolUsageDetector {
    fn id(&self) -> &str {
        "excessive_tool_usage"
    }

    fn version(&self) -> &str {
        DETERMINISTIC_DETECTOR_VERSION
    }

    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        let total_duration_ms = trace
            .tool_calls
            .iter()
            .map(|call| call.duration_ms)
            .sum::<u64>();
        if trace.tool_calls.len() <= self.max_tool_calls
            && total_duration_ms <= self.max_total_duration_ms
        {
            return Vec::new();
        }
        let mut metadata = BTreeMap::new();
        metadata.insert("tool_call_count".to_string(), json!(trace.tool_calls.len()));
        metadata.insert("tool_call_limit".to_string(), json!(self.max_tool_calls));
        metadata.insert("total_duration_ms".to_string(), json!(total_duration_ms));
        metadata.insert(
            "total_duration_limit_ms".to_string(),
            json!(self.max_total_duration_ms),
        );
        let evidence = trace
            .tool_calls
            .iter()
            .flat_map(|call| call.evidence.clone())
            .collect();
        vec![build_finding(
            trace,
            self,
            FindingSeverity::Medium,
            RecoveryStatus::Unrecovered,
            ("tool_budget".to_string(), None),
            None,
            evidence,
            metadata,
        )]
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct UnresolvedEscalationDetector;

impl TraceDetector for UnresolvedEscalationDetector {
    fn id(&self) -> &str {
        "unresolved_escalation"
    }

    fn version(&self) -> &str {
        DETERMINISTIC_DETECTOR_VERSION
    }

    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        if !trace.final_outcome.escalation_required
            || trace.final_outcome.escalation_performed
            || trace.final_outcome.status == FinalOutcomeStatus::Escalated
        {
            return Vec::new();
        }
        let mut evidence = trace.final_outcome.evidence.clone();
        evidence.extend(
            trace
                .policy_decisions
                .iter()
                .filter(|decision| {
                    decision.outcome == PolicyDecisionOutcome::Required
                        && decision
                            .action
                            .as_deref()
                            .is_some_and(|action| action.contains("escalat"))
                })
                .flat_map(|decision| decision.evidence.clone()),
        );
        vec![build_finding(
            trace,
            self,
            FindingSeverity::High,
            RecoveryStatus::Unrecovered,
            ("escalation".to_string(), None),
            None,
            evidence,
            BTreeMap::new(),
        )]
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MissingResolutionDetector;

impl TraceDetector for MissingResolutionDetector {
    fn id(&self) -> &str {
        "missing_resolution"
    }

    fn version(&self) -> &str {
        DETERMINISTIC_DETECTOR_VERSION
    }

    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        let safe_terminal = trace.final_outcome.resolution_present
            || matches!(
                trace.final_outcome.status,
                FinalOutcomeStatus::Refused | FinalOutcomeStatus::Escalated
            )
            || (trace.final_outcome.failure_acknowledged
                && trace.final_outcome.status == FinalOutcomeStatus::Failed);
        if safe_terminal {
            return Vec::new();
        }
        vec![build_finding(
            trace,
            self,
            FindingSeverity::Medium,
            RecoveryStatus::Unrecovered,
            ("final_outcome".to_string(), None),
            None,
            trace.final_outcome.evidence.clone(),
            BTreeMap::new(),
        )]
    }
}

pub struct DeterministicDetectorSet {
    detectors: Vec<Box<dyn TraceDetector>>,
}

impl Default for DeterministicDetectorSet {
    fn default() -> Self {
        Self {
            detectors: vec![
                Box::new(TerminalToolFailureDetector),
                Box::new(RepeatedToolFailureDetector::default()),
                Box::new(ToolCallLoopDetector::default()),
                Box::new(UncertainMutationStateDetector),
                Box::new(FalseSuccessClaimDetector),
                Box::new(ApprovalBypassDetector),
                Box::new(PolicyViolationDetector),
                Box::new(ExcessiveToolUsageDetector::default()),
                Box::new(UnresolvedEscalationDetector),
                Box::new(MissingResolutionDetector),
            ],
        }
    }
}

impl DeterministicDetectorSet {
    pub fn new(detectors: Vec<Box<dyn TraceDetector>>) -> Self {
        Self { detectors }
    }

    pub fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding> {
        self.detectors
            .iter()
            .flat_map(|detector| detector.detect(trace))
            .collect()
    }

    pub fn detect_traces(&self, traces: &[AgentBehaviorTrace]) -> Vec<BehaviorFinding> {
        traces.iter().flat_map(|trace| self.detect(trace)).collect()
    }
}

fn finding_for_call(
    trace: &AgentBehaviorTrace,
    detector: &dyn TraceDetector,
    severity: FindingSeverity,
    recovery: RecoveryStatus,
    call: &ToolCallFact,
    mut metadata: BTreeMap<String, Value>,
) -> BehaviorFinding {
    metadata.insert("tool_name".to_string(), json!(call.tool_name));
    if let Some(operation) = &call.operation {
        metadata.insert("operation".to_string(), json!(operation));
    }
    metadata.insert("call_id".to_string(), json!(call.call_id));
    build_finding(
        trace,
        detector,
        severity,
        recovery,
        signature_subject(call),
        error_kind(call),
        call.evidence.clone(),
        metadata,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_finding(
    trace: &AgentBehaviorTrace,
    detector: &dyn TraceDetector,
    severity: FindingSeverity,
    recovery: RecoveryStatus,
    signature_subject: (String, Option<String>),
    error_kind: Option<String>,
    mut evidence: Vec<EvidenceRef>,
    metadata: BTreeMap<String, Value>,
) -> BehaviorFinding {
    evidence.sort_by(|left, right| left.identity.cmp(&right.identity));
    evidence.dedup_by(|left, right| left.identity == right.identity);
    let evidence_ids = evidence
        .iter()
        .map(|evidence| evidence.identity.as_str())
        .collect::<Vec<_>>();
    let finding_id = hash_parts(
        [trace.trace_id.as_str(), detector.id(), detector.version()]
            .into_iter()
            .chain(evidence_ids),
    );
    let failure_signature = hash_parts([
        detector.id(),
        signature_subject.0.as_str(),
        signature_subject.1.as_deref().unwrap_or(""),
        error_kind.as_deref().unwrap_or(""),
    ]);
    let created_at = trace
        .metadata
        .get("observed_at")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            trace
                .tool_calls
                .last()
                .map(|call| call.started_at.clone())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string());

    BehaviorFinding {
        schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
        finding_id,
        detector_id: detector.id().to_string(),
        detector_version: detector.version().to_string(),
        trace_id: trace.trace_id.clone(),
        kind: detector.id().to_string(),
        severity,
        recovery,
        confidence: Some(1.0),
        failure_signature,
        evidence,
        created_at,
        metadata,
    }
}

fn equivalent_call(left: &ToolCallFact, right: &ToolCallFact) -> bool {
    left.tool_name == right.tool_name && left.operation == right.operation
}

fn has_material_progress(call: &ToolCallFact) -> bool {
    call.status == ToolCallStatus::Succeeded
        && call
            .state_change
            .as_ref()
            .is_some_and(|state| state.changed || state.verified)
}

fn signature_subject(call: &ToolCallFact) -> (String, Option<String>) {
    (call.tool_name.clone(), call.operation.clone())
}

fn error_kind(call: &ToolCallFact) -> Option<String> {
    call.error.as_ref().map(|error| error.kind.clone())
}

fn hash_parts<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.len().to_be_bytes());
        hasher.update(part.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::model::{FinalOutcome, StateChangeRef};

    fn call(name: &str, status: ToolCallStatus, mutating: bool) -> ToolCallFact {
        ToolCallFact {
            call_id: format!("{name}-call"),
            tool_name: name.to_string(),
            operation: Some(if mutating { "cancel" } else { "lookup" }.to_string()),
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
            required: true,
            idempotent: Some(!mutating),
            mutating,
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
        verify.state_change = Some(StateChangeRef {
            evidence: EvidenceRef::new("state_change", "state:verified"),
            changed: false,
            verified: true,
        });
        let mut trace = trace_with(vec![uncertain, verify]);
        trace.final_outcome.claimed_success = true;

        assert_eq!(
            RecoveryAnalyzer.recovery_for_call(&trace, 0),
            RecoveryStatus::Recovered
        );
        assert!(UncertainMutationStateDetector.detect(&trace).is_empty());
        assert!(FalseSuccessClaimDetector.detect(&trace).is_empty());
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
    fn unrelated_success_does_not_support_a_failed_mutation_claim() {
        let failed = call("cancel_card", ToolCallStatus::TimedOut, true);
        let mut unrelated = call("update_preferences", ToolCallStatus::Succeeded, true);
        unrelated.operation = Some("update".to_string());
        unrelated.state_change = Some(StateChangeRef {
            evidence: EvidenceRef::new("state_change", "state:preferences"),
            changed: true,
            verified: false,
        });
        let mut trace = trace_with(vec![failed, unrelated]);
        trace.final_outcome.claimed_success = true;

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
        trace.final_outcome.escalation_required = true;

        assert_eq!(ApprovalBypassDetector.detect(&trace).len(), 1);
        assert_eq!(PolicyViolationDetector.detect(&trace).len(), 1);
        assert_eq!(
            ExcessiveToolUsageDetector::new(10, 50).detect(&trace).len(),
            1
        );
        assert_eq!(UnresolvedEscalationDetector.detect(&trace).len(), 1);
    }
}
