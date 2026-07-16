use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::detection_report::{
    DETECTION_REPORT_SCHEMA_VERSION, DetectionReportV1, DetectorCoverageV1,
    DetectorEvaluationStatusV1, DetectorProfileIdentityV1, DetectorProfileV1,
    TelemetryDiagnosticSeverityV1, TelemetryDiagnosticV1,
};
use super::model::{
    AgentBehaviorTrace, ApprovalOutcome, BEHAVIOR_FINDING_SCHEMA_VERSION, BehaviorFinding,
    ClaimedOutcomeStatus, EscalationStatus, EvidenceRef, FinalOutcomeStatus, FindingCertaintyV1,
    FindingSeverity, OperationEffect, OutcomeClaim, PolicyDecisionOutcome, RecoveryStatus,
    RetrySafety, RuleMatchCertaintyV1, StateObservation, ToolCallFact, ToolCallStatus,
    ToolRequirement,
};
use crate::model::FactQuality;

mod policy;
mod recovery;
mod support;

pub use policy::{ApprovalBypassDetector, ExcessiveToolUsageDetector, PolicyViolationDetector};
pub use recovery::RecoveryAnalyzer;

use recovery::{
    claim_has_success_evidence, claim_matches_call, equivalent_call_key, has_material_progress,
};
use support::{build_finding, error_kind, finding_for_call, signature_subject};

pub const DETERMINISTIC_DETECTOR_VERSION: &str = "6";

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
    max_call_window: usize,
}

impl Default for RepeatedToolFailureDetector {
    fn default() -> Self {
        Self {
            max_failures: 3,
            max_call_window: 8,
        }
    }
}

impl RepeatedToolFailureDetector {
    pub fn new(max_failures: usize) -> Self {
        Self {
            max_failures: max_failures.max(1),
            ..Self::default()
        }
    }

    pub fn with_call_window(max_failures: usize, max_call_window: usize) -> Self {
        Self {
            max_failures: max_failures.max(1),
            max_call_window: max_call_window.max(max_failures.max(1)),
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
        let mut active: BTreeMap<RepeatedFailureKey, Vec<usize>> = BTreeMap::new();
        let mut episodes = Vec::new();
        for (index, call) in trace.tool_calls.iter().enumerate() {
            if has_material_progress(call) || call.status == ToolCallStatus::Succeeded {
                finish_repeated_episodes(&mut active, self.max_failures, &mut episodes);
                continue;
            }
            let Some(key) = repeated_failure_key(call) else {
                continue;
            };
            if active
                .get(&key)
                .is_some_and(|indices| index.saturating_sub(indices[0]) >= self.max_call_window)
                && let Some(expired) = active.remove(&key)
                && expired.len() >= self.max_failures
            {
                episodes.push(expired);
            }
            active.entry(key).or_default().push(index);
        }
        finish_repeated_episodes(&mut active, self.max_failures, &mut episodes);

        episodes
            .into_iter()
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
                metadata.insert("episode_start_call".to_string(), json!(indices[0]));
                metadata.insert("episode_end_call".to_string(), json!(last_index));
                metadata.insert("call_window".to_string(), json!(self.max_call_window));
                if let Some(fingerprint) = &call.invocation_fingerprint {
                    metadata.insert("invocation_fingerprint".to_string(), json!(fingerprint));
                }
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RepeatedFailureKey {
    tool_name: String,
    compatible_identity: String,
    invocation_fingerprint: String,
    error_kind: String,
}

fn repeated_failure_key(call: &ToolCallFact) -> Option<RepeatedFailureKey> {
    if !call.status.is_failure()
        || !matches!(
            call.invocation_fingerprint_quality,
            FactQuality::Explicit | FactQuality::Derived
        )
    {
        return None;
    }
    let compatible_identity = if matches!(
        call.operation_source_quality,
        FactQuality::Explicit | FactQuality::Derived
    ) {
        format!("operation:{}", call.operation.as_ref()?)
    } else if matches!(
        call.tool_name_source_quality,
        FactQuality::Explicit | FactQuality::Derived
    ) {
        format!("tool:{}", call.tool_name)
    } else {
        return None;
    };
    Some(RepeatedFailureKey {
        tool_name: call.tool_name.clone(),
        compatible_identity,
        invocation_fingerprint: call.invocation_fingerprint.clone()?,
        error_kind: error_kind(call).unwrap_or_else(|| "unknown".into()),
    })
}

fn finish_repeated_episodes(
    active: &mut BTreeMap<RepeatedFailureKey, Vec<usize>>,
    minimum_failures: usize,
    episodes: &mut Vec<Vec<usize>>,
) {
    episodes.extend(
        std::mem::take(active)
            .into_values()
            .filter(|indices| indices.len() >= minimum_failures),
    );
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
        let mut segment = Vec::new();
        for call in &trace.tool_calls {
            let key = equivalent_call_key(call);
            if has_material_progress(call) || key.is_none() {
                if let Some(episode) = loop_episode(&segment, self.max_equivalent_calls) {
                    findings.push(self.finding(trace, &episode));
                }
                segment.clear();
                continue;
            }
            segment.push((key.expect("checked equivalent key"), call));
        }
        if let Some(episode) = loop_episode(&segment, self.max_equivalent_calls) {
            findings.push(self.finding(trace, &episode));
        }
        findings
    }
}

impl ToolCallLoopDetector {
    fn finding(&self, trace: &AgentBehaviorTrace, episode: &LoopEpisode<'_>) -> BehaviorFinding {
        let last = episode
            .equivalent_calls
            .last()
            .expect("loop episode has equivalent calls");
        let evidence = episode
            .all_calls
            .iter()
            .flat_map(|call| call.evidence.clone())
            .collect();
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "call_count".to_string(),
            json!(episode.equivalent_calls.len()),
        );
        metadata.insert(
            "equivalent_call_limit".to_string(),
            json!(self.max_equivalent_calls),
        );
        metadata.insert("cycle_length".to_string(), json!(episode.cycle_length));
        build_finding(
            trace,
            self,
            FindingSeverity::Medium,
            loop_recovery(trace),
            signature_subject(last),
            error_kind(last),
            evidence,
            metadata,
        )
    }
}

struct LoopEpisode<'a> {
    equivalent_calls: Vec<&'a ToolCallFact>,
    all_calls: Vec<&'a ToolCallFact>,
    cycle_length: usize,
}

fn loop_episode<'a>(
    segment: &[((String, String, String), &'a ToolCallFact)],
    minimum_equivalent_calls: usize,
) -> Option<LoopEpisode<'a>> {
    let mut positions = BTreeMap::<&(String, String, String), Vec<usize>>::new();
    for (index, (key, _)) in segment.iter().enumerate() {
        positions.entry(key).or_default().push(index);
    }
    let keys = segment.iter().map(|(key, _)| key).collect::<Vec<_>>();
    let mut candidates = Vec::new();
    for equivalent_positions in positions.values() {
        for window in equivalent_positions.windows(minimum_equivalent_calls) {
            let cycle_length = window[1].saturating_sub(window[0]);
            let reference = &keys[window[0].saturating_add(1)..window[1]];
            if window
                .windows(2)
                .all(|pair| &keys[pair[0].saturating_add(1)..pair[1]] == reference)
            {
                candidates.push((window[0], *window.last()?, cycle_length, window.to_vec()));
            }
        }
    }
    candidates.sort_by_key(|(start, end, cycle_length, _)| (*end, *start, *cycle_length));
    let (start, end, cycle_length, equivalent_positions) = candidates.into_iter().next()?;
    Some(LoopEpisode {
        equivalent_calls: equivalent_positions
            .into_iter()
            .map(|index| segment[index].1)
            .collect(),
        all_calls: segment[start..=end].iter().map(|(_, call)| *call).collect(),
        cycle_length,
    })
}

fn loop_recovery(trace: &AgentBehaviorTrace) -> RecoveryStatus {
    match trace.final_outcome.status {
        FinalOutcomeStatus::Completed | FinalOutcomeStatus::SafelyRefused => {
            RecoveryStatus::Recovered
        }
        FinalOutcomeStatus::Failed | FinalOutcomeStatus::Incomplete => RecoveryStatus::Unrecovered,
        FinalOutcomeStatus::Escalated
            if trace.final_outcome.escalation == EscalationStatus::RequiredAndPerformed =>
        {
            RecoveryStatus::Recovered
        }
        FinalOutcomeStatus::Escalated | FinalOutcomeStatus::Unknown => RecoveryStatus::Unknown,
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
    profile: DetectorProfileIdentityV1,
}

impl Default for DeterministicDetectorSet {
    fn default() -> Self {
        Self::from_profile(DetectorProfileV1::conservative())
            .expect("the built-in conservative profile is valid")
    }
}

impl DeterministicDetectorSet {
    pub fn new(detectors: Vec<Box<dyn TraceDetector>>) -> Self {
        Self {
            detectors,
            profile: DetectorProfileIdentityV1 {
                profile_id: "traceeval.custom".into(),
                profile_version: DETERMINISTIC_DETECTOR_VERSION.into(),
            },
        }
    }

    pub fn from_profile(profile: DetectorProfileV1) -> crate::Result<Self> {
        profile
            .validate()
            .map_err(|message| crate::TraceEvalError::InvalidDetectorProfile {
                profile_id: profile.identity.profile_id.clone(),
                message,
            })?;
        let mut detectors: Vec<Box<dyn TraceDetector>> = Vec::new();
        for detector_id in &profile.enabled_detectors {
            let detector: Box<dyn TraceDetector> = match detector_id.as_str() {
                "terminal_tool_failure" => Box::new(TerminalToolFailureDetector),
                "repeated_tool_failure" => Box::new(RepeatedToolFailureDetector::with_call_window(
                    profile.repeated_failure.minimum_failures,
                    profile.repeated_failure.maximum_call_window,
                )),
                "tool_call_loop" => Box::new(ToolCallLoopDetector::new(
                    profile.tool_loop.minimum_equivalent_calls,
                )),
                "uncertain_mutation_state" => Box::new(UncertainMutationStateDetector),
                "false_success_claim" => Box::new(FalseSuccessClaimDetector),
                "approval_bypass" => Box::new(ApprovalBypassDetector),
                "policy_violation" => Box::new(PolicyViolationDetector),
                "excessive_tool_usage" => {
                    let budget = profile
                        .tool_usage_budget
                        .as_ref()
                        .expect("profile validation requires a budget");
                    Box::new(ExcessiveToolUsageDetector::new(
                        budget.maximum_calls,
                        budget.maximum_total_duration_ms,
                    ))
                }
                "unresolved_escalation" => Box::new(UnresolvedEscalationDetector),
                "missing_resolution" => Box::new(MissingResolutionDetector),
                unknown => {
                    return Err(crate::TraceEvalError::InvalidDetectorProfile {
                        profile_id: profile.identity.profile_id.clone(),
                        message: format!("unknown detector {unknown:?}"),
                    });
                }
            };
            detectors.push(detector);
        }
        Ok(Self {
            detectors,
            profile: profile.identity,
        })
    }

    pub fn profile(&self) -> &DetectorProfileIdentityV1 {
        &self.profile
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

    /// Runs the conservative, coverage-aware API. Unlike `detect`, this never turns a
    /// missing required fact into an actionable finding.
    pub fn detect_report(&self, trace: &AgentBehaviorTrace) -> DetectionReportV1 {
        let mut findings = Vec::new();
        let mut telemetry_diagnostics = trace_diagnostics(trace);
        telemetry_diagnostics.extend(call_diagnostics(trace));
        let mut detector_coverage = BTreeMap::new();
        for detector in &self.detectors {
            let coverage = coverage_for_detector(trace, detector.id());
            if coverage.status == DetectorEvaluationStatusV1::Evaluated {
                findings.extend(detector.detect(trace));
            } else {
                telemetry_diagnostics.push(TelemetryDiagnosticV1 {
                    code: "detector_inconclusive_missing_facts".into(),
                    severity: TelemetryDiagnosticSeverityV1::Warning,
                    message: format!(
                        "{} was not evaluated because required telemetry is missing: {}",
                        detector.id(),
                        coverage
                            .missing_facts
                            .iter()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    trace_id: trace.trace_id.clone(),
                    detector_id: Some(detector.id().into()),
                    call_id: None,
                    evidence: trace.evidence.clone(),
                });
            }
            detector_coverage.insert(detector.id().to_string(), coverage);
        }
        telemetry_diagnostics.sort_by(|left, right| {
            (&left.code, &left.detector_id, &left.call_id).cmp(&(
                &right.code,
                &right.detector_id,
                &right.call_id,
            ))
        });
        DetectionReportV1 {
            schema_version: DETECTION_REPORT_SCHEMA_VERSION.into(),
            trace_id: trace.trace_id.clone(),
            input_schema_version: trace.input_schema_version.clone(),
            profile: self.profile.clone(),
            detector_versions: self.detector_versions(),
            input_coverage: trace.coverage.clone(),
            detector_coverage,
            findings,
            telemetry_diagnostics,
        }
    }

    pub fn detect_reports(&self, traces: &[AgentBehaviorTrace]) -> Vec<DetectionReportV1> {
        traces
            .iter()
            .map(|trace| self.detect_report(trace))
            .collect()
    }

    pub fn detector_versions(&self) -> BTreeMap<String, String> {
        self.detectors
            .iter()
            .map(|detector| (detector.id().to_string(), detector.version().to_string()))
            .collect()
    }
}

fn coverage_for_detector(trace: &AgentBehaviorTrace, detector_id: &str) -> DetectorCoverageV1 {
    let required_facts = required_facts(trace, detector_id);
    let observed_facts = required_facts
        .iter()
        .filter(|fact| fact_is_observed(trace, fact))
        .cloned()
        .collect::<BTreeSet<_>>();
    let missing_facts = required_facts
        .difference(&observed_facts)
        .cloned()
        .collect::<BTreeSet<_>>();
    let semantic_coverage = if required_facts.is_empty() {
        1.0
    } else {
        observed_facts.len() as f32 / required_facts.len() as f32
    };
    let status = match detector_id {
        "repeated_tool_failure" => {
            let failures = trace
                .tool_calls
                .iter()
                .filter(|call| call.status.is_failure())
                .collect::<Vec<_>>();
            let eligible = failures
                .iter()
                .filter(|call| repeated_failure_key(call).is_some())
                .count();
            if eligible >= 3 || (failures.len() < 3 && fact_is_observed(trace, "source_status")) {
                DetectorEvaluationStatusV1::Evaluated
            } else {
                DetectorEvaluationStatusV1::Inconclusive
            }
        }
        "tool_call_loop" => {
            let eligible = trace
                .tool_calls
                .iter()
                .filter(|call| equivalent_call_key(call).is_some())
                .count();
            if eligible >= 4 || trace.tool_calls.len() < 4 {
                DetectorEvaluationStatusV1::Evaluated
            } else {
                DetectorEvaluationStatusV1::Inconclusive
            }
        }
        _ if missing_facts.is_empty() => DetectorEvaluationStatusV1::Evaluated,
        _ => DetectorEvaluationStatusV1::Inconclusive,
    };
    DetectorCoverageV1 {
        detector_id: detector_id.into(),
        status,
        required_facts,
        observed_facts,
        missing_facts,
        semantic_coverage,
    }
}

fn required_facts(trace: &AgentBehaviorTrace, detector_id: &str) -> BTreeSet<String> {
    if trace.tool_calls.is_empty()
        && matches!(
            detector_id,
            "terminal_tool_failure"
                | "repeated_tool_failure"
                | "tool_call_loop"
                | "uncertain_mutation_state"
                | "approval_bypass"
                | "excessive_tool_usage"
        )
    {
        return BTreeSet::new();
    }
    let facts: &[&str] = match detector_id {
        "terminal_tool_failure" => &["source_status", "tool_requirement", "final_outcome"],
        "repeated_tool_failure" => &["source_status", "tool_identity", "invocation_fingerprint"],
        "tool_call_loop" => &["source_status", "tool_identity", "invocation_fingerprint"],
        "uncertain_mutation_state" => &["source_status", "operation_identity", "state_observation"],
        "false_success_claim" => &["final_outcome", "source_status"],
        "approval_bypass" | "policy_violation" => &["policy_semantics"],
        "excessive_tool_usage" => &["numeric_duration"],
        "unresolved_escalation" => &["final_outcome", "policy_semantics"],
        "missing_resolution" => &["final_outcome"],
        _ => &[],
    };
    facts.iter().map(|fact| (*fact).to_string()).collect()
}

fn fact_is_observed(trace: &AgentBehaviorTrace, fact: &str) -> bool {
    match fact {
        "source_status" => trace
            .tool_calls
            .iter()
            .all(|call| quality_is_observed(call.status_quality)),
        "tool_identity" => trace
            .tool_calls
            .iter()
            .filter(|call| call.status.is_failure())
            .all(|call| {
                (call.operation.is_some() && quality_is_observed(call.operation_source_quality))
                    || quality_is_observed(call.tool_name_source_quality)
            }),
        "operation_identity" => trace.tool_calls.iter().all(|call| {
            call.operation.is_some() && quality_is_observed(call.operation_source_quality)
        }),
        "invocation_fingerprint" => trace
            .tool_calls
            .iter()
            .filter(|call| call.status.is_failure())
            .all(|call| {
                call.invocation_fingerprint.is_some()
                    && quality_is_observed(call.invocation_fingerprint_quality)
            }),
        "numeric_duration" => trace
            .tool_calls
            .iter()
            .all(|call| call.duration_nano.is_some()),
        "tool_requirement" => trace
            .tool_calls
            .iter()
            .filter(|call| call.status.is_failure())
            .all(|call| call.requirement != ToolRequirement::Unknown),
        "state_observation" => trace.tool_calls.iter().all(|call| {
            call.effect != OperationEffect::Mutating
                || call
                    .state_change
                    .as_ref()
                    .is_some_and(|state| !matches!(state.observation, StateObservation::Unknown))
        }),
        "final_outcome" => trace.coverage.final_outcome != FactQuality::Missing,
        "policy_semantics" => !trace.policy_decisions.is_empty(),
        _ => false,
    }
}

fn quality_is_observed(quality: FactQuality) -> bool {
    matches!(
        quality,
        FactQuality::Explicit | FactQuality::Derived | FactQuality::Inferred
    )
}

fn trace_diagnostics(trace: &AgentBehaviorTrace) -> Vec<TelemetryDiagnosticV1> {
    if trace.coverage.final_outcome == FactQuality::Missing {
        return vec![TelemetryDiagnosticV1 {
            code: "missing_final_outcome".into(),
            severity: TelemetryDiagnosticSeverityV1::Warning,
            message: "No explicit terminal outcome was observed; the generic profile does not classify missing telemetry as a failure.".into(),
            trace_id: trace.trace_id.clone(),
            detector_id: None,
            call_id: None,
            evidence: trace.final_outcome.evidence.clone(),
        }];
    }
    if matches!(
        trace.final_outcome.status,
        FinalOutcomeStatus::Incomplete | FinalOutcomeStatus::Unknown
    ) {
        return vec![TelemetryDiagnosticV1 {
            code: "non_terminal_final_outcome".into(),
            severity: TelemetryDiagnosticSeverityV1::Info,
            message: "The observed outcome is non-terminal; it is diagnostic-only unless a selected protocol explicitly requires a terminal signal.".into(),
            trace_id: trace.trace_id.clone(),
            detector_id: None,
            call_id: None,
            evidence: trace.final_outcome.evidence.clone(),
        }];
    }
    Vec::new()
}

fn call_diagnostics(trace: &AgentBehaviorTrace) -> Vec<TelemetryDiagnosticV1> {
    let mut diagnostics = Vec::new();
    for call in &trace.tool_calls {
        for (fact, quality) in [
            ("source_status", call.status_quality),
            (
                "tool_identity",
                if quality_is_observed(call.operation_source_quality) {
                    call.operation_source_quality
                } else {
                    call.tool_name_source_quality
                },
            ),
            (
                "invocation_fingerprint",
                call.invocation_fingerprint_quality,
            ),
        ] {
            if quality_is_observed(quality) {
                continue;
            }
            let ambiguous = quality == FactQuality::Ambiguous;
            diagnostics.push(TelemetryDiagnosticV1 {
                code: if ambiguous {
                    format!("ambiguous_{fact}")
                } else {
                    format!("missing_{fact}")
                },
                severity: TelemetryDiagnosticSeverityV1::Warning,
                message: if ambiguous {
                    format!("tool call {} has conflicting {fact}", call.call_id)
                } else {
                    format!("tool call {} does not provide {fact}", call.call_id)
                },
                trace_id: trace.trace_id.clone(),
                detector_id: None,
                call_id: Some(call.call_id.clone()),
                evidence: call.evidence.clone(),
            });
        }
    }
    diagnostics
}

#[cfg(test)]
#[path = "detectors/tests.rs"]
mod tests;
