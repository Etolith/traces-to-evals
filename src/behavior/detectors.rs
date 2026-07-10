use std::collections::BTreeMap;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::model::{
    AgentBehaviorTrace, ApprovalOutcome, BEHAVIOR_FINDING_SCHEMA_VERSION, BehaviorFinding,
    ClaimedOutcomeStatus, EscalationStatus, EvidenceRef, FinalOutcomeStatus, FindingSeverity,
    OperationEffect, OutcomeClaim, PolicyDecisionOutcome, RecoveryStatus, RetrySafety,
    StateObservation, ToolCallFact, ToolCallStatus, ToolRequirement,
};

mod policy;
mod recovery;
mod support;

pub use policy::{ApprovalBypassDetector, ExcessiveToolUsageDetector, PolicyViolationDetector};
pub use recovery::RecoveryAnalyzer;

use recovery::{
    claim_has_success_evidence, claim_matches_call, equivalent_call_key, has_material_progress,
};
use support::{build_finding, error_kind, finding_for_call, signature_subject};

pub const DETERMINISTIC_DETECTOR_VERSION: &str = "2";

pub trait TraceDetector: Send + Sync {
    fn id(&self) -> &str;
    fn version(&self) -> &str;
    fn detect(&self, trace: &AgentBehaviorTrace) -> Vec<BehaviorFinding>;
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
                call.requirement == ToolRequirement::Required
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
        for segment in trace.tool_calls.split(has_material_progress) {
            let mut groups: BTreeMap<(String, String), Vec<&ToolCallFact>> = BTreeMap::new();
            for call in segment {
                groups
                    .entry(equivalent_call_key(call))
                    .or_default()
                    .push(call);
            }
            for calls in groups
                .into_values()
                .filter(|calls| calls.len() >= self.max_equivalent_calls)
            {
                let last = calls.last().expect("non-empty loop group");
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
                call.effect == OperationEffect::Mutating
                    && (matches!(
                        call.status,
                        ToolCallStatus::TimedOut | ToolCallStatus::Unknown
                    ) || call.state_change.as_ref().is_some_and(|state| {
                        matches!(
                            state.observation,
                            StateObservation::Ambiguous | StateObservation::Conflicting
                        )
                    }))
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
        trace
            .final_outcome
            .claims
            .iter()
            .filter(|claim| claim.status == ClaimedOutcomeStatus::Succeeded)
            .filter(|claim| !claim_has_success_evidence(trace, claim))
            .map(|claim| {
                let matching_calls = trace
                    .tool_calls
                    .iter()
                    .filter(|call| claim_matches_call(claim, call))
                    .collect::<Vec<_>>();
                let mut evidence = claim.evidence.clone();
                evidence.extend(matching_calls.iter().flat_map(|call| call.evidence.clone()));
                let subject = matching_calls
                    .last()
                    .map(|call| signature_subject(call))
                    .unwrap_or_else(|| {
                        (
                            claim
                                .operation
                                .clone()
                                .unwrap_or_else(|| "final_outcome".to_string()),
                            claim.operation.clone(),
                        )
                    });
                let error = matching_calls.last().and_then(|call| error_kind(call));
                build_finding(
                    trace,
                    self,
                    FindingSeverity::High,
                    RecoveryStatus::Unrecovered,
                    subject,
                    error,
                    evidence,
                    BTreeMap::new(),
                )
            })
            .collect()
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
        if trace.final_outcome.escalation != EscalationStatus::RequiredAndMissing {
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
        let safe_terminal = matches!(
            trace.final_outcome.status,
            FinalOutcomeStatus::Completed
                | FinalOutcomeStatus::Failed
                | FinalOutcomeStatus::SafelyRefused
                | FinalOutcomeStatus::Escalated
        );
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

    pub fn detector_versions(&self) -> BTreeMap<String, String> {
        self.detectors
            .iter()
            .map(|detector| (detector.id().to_string(), detector.version().to_string()))
            .collect()
    }
}

#[cfg(test)]
#[path = "detectors/tests.rs"]
mod tests;
