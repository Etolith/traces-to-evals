use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::input::{BehaviorInputCoverageV1, BehaviorInputProvenanceV1};
use crate::model::FactQuality;
use crate::{Result, TraceEvalError};

pub const AGENT_BEHAVIOR_TRACE_SCHEMA_VERSION: &str = "traceeval.agent_behavior_trace.v2";
pub const BEHAVIOR_FINDING_SCHEMA_VERSION: &str = "agent.behavior.finding.v1";
pub const EVAL_CANDIDATE_SCHEMA_VERSION: &str = "traceeval.eval_candidate.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub kind: String,
    pub identity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
}

impl EvidenceRef {
    pub fn new(kind: impl Into<String>, identity: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            identity: identity.into(),
            span_id: None,
        }
    }

    pub fn span(span_id: impl Into<String>) -> Self {
        let span_id = span_id.into();
        Self {
            kind: "span".to_string(),
            identity: format!("span:{span_id}"),
            span_id: Some(span_id),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateChangeRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    pub observation: StateObservation,
    pub artifact: EvidenceRef,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateObservation {
    VerifiedChanged,
    VerifiedUnchanged,
    Unverified,
    Ambiguous,
    Conflicting,
    #[default]
    Unknown,
}

impl StateObservation {
    pub fn is_verified(self) -> bool {
        matches!(self, Self::VerifiedChanged | Self::VerifiedUnchanged)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationEffect {
    ReadOnly,
    Mutating,
    Verifying,
    Compensating,
    Escalating,
    #[default]
    Unknown,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrySafety {
    Idempotent,
    NonIdempotent,
    #[default]
    Unknown,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRequirement {
    Required,
    Optional,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    Succeeded,
    Failed,
    TimedOut,
    Cancelled,
    Unknown,
}

impl ToolCallStatus {
    pub fn is_failure(self) -> bool {
        matches!(self, Self::Failed | Self::TimedOut | Self::Cancelled)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedToolError {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redacted_message_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallFact {
    pub call_id: String,
    pub tool_name: String,
    #[serde(default)]
    pub tool_name_source_quality: FactQuality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(default)]
    pub operation_source_quality: FactQuality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invocation_fingerprint: Option<String>,
    #[serde(default)]
    pub invocation_fingerprint_quality: FactQuality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_fingerprint: Option<String>,
    #[serde(default)]
    pub result_fingerprint_quality: FactQuality,
    #[serde(default)]
    pub effect: OperationEffect,
    #[serde(default)]
    pub retry_safety: RetrySafety,
    #[serde(default)]
    pub requirement: ToolRequirement,
    #[serde(default = "default_attempt")]
    pub attempt: u32,
    pub started_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at_unix_nano: Option<u64>,
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_nano: Option<u64>,
    pub status: ToolCallStatus,
    #[serde(default)]
    pub status_quality: FactQuality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<NormalizedToolError>,
    #[serde(default)]
    pub approval_required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_outcome: Option<ApprovalOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_change: Option<StateChangeRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
}

fn default_attempt() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalOutcome {
    Approved,
    Denied,
    Cancelled,
    NotRequested,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    User,
    Assistant,
    Tool,
    System,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentTurn {
    pub turn_id: String,
    pub role: AgentRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyDecisionOutcome {
    Allowed,
    Denied,
    Required,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub decision_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    pub outcome: PolicyDecisionOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinalOutcomeStatus {
    Completed,
    Failed,
    SafelyRefused,
    Escalated,
    Incomplete,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimedOutcomeStatus {
    Succeeded,
    Failed,
    NotCompleted,
    StateUnknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeClaim {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub status: ClaimedOutcomeStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationStatus {
    NotRequired,
    RequiredAndPerformed,
    RequiredAndMissing,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalOutcome {
    pub status: FinalOutcomeStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub claims: Vec<OutcomeClaim>,
    #[serde(default)]
    pub escalation: EscalationStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
}

impl Default for FinalOutcome {
    fn default() -> Self {
        Self {
            status: FinalOutcomeStatus::Unknown,
            claims: Vec::new(),
            escalation: EscalationStatus::Unknown,
            evidence: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentBehaviorTrace {
    pub schema_version: String,
    #[serde(default = "legacy_input_schema_version")]
    pub input_schema_version: String,
    pub trace_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_summary: Option<String>,
    #[serde(default)]
    pub turns: Vec<AgentTurn>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallFact>,
    #[serde(default)]
    pub policy_decisions: Vec<PolicyDecision>,
    pub final_outcome: FinalOutcome,
    #[serde(default)]
    pub coverage: BehaviorInputCoverageV1,
    #[serde(default)]
    pub provenance: BehaviorInputProvenanceV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_at_unix_nano: Option<u64>,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl AgentBehaviorTrace {
    pub fn new(trace_id: impl Into<String>) -> Self {
        Self {
            schema_version: AGENT_BEHAVIOR_TRACE_SCHEMA_VERSION.to_string(),
            input_schema_version: legacy_input_schema_version(),
            trace_id: trace_id.into(),
            input_summary: None,
            turns: Vec::new(),
            tool_calls: Vec::new(),
            policy_decisions: Vec::new(),
            final_outcome: FinalOutcome::default(),
            coverage: BehaviorInputCoverageV1::default(),
            provenance: BehaviorInputProvenanceV1::default(),
            observed_at_unix_nano: None,
            evidence: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }
}

fn legacy_input_schema_version() -> String {
    "traceeval.legacy_trace_projection.v1".into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "llm-judge-openai", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum FindingSeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryStatus {
    Recovered,
    Unrecovered,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehaviorFinding {
    pub schema_version: String,
    pub finding_id: String,
    pub detector_id: String,
    pub detector_version: String,
    pub trace_id: String,
    pub kind: String,
    pub severity: FindingSeverity,
    pub recovery: RecoveryStatus,
    /// Compatibility field for v1 consumers. Deterministic detectors leave this unset;
    /// probabilistic detectors must also populate `certainty.calibrated_failure_risk`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub certainty: FindingCertaintyV1,
    pub failure_signature: String,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleMatchCertaintyV1 {
    Exact,
    BoundedInference,
    #[default]
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FindingCertaintyV1 {
    pub rule_match: RuleMatchCertaintyV1,
    pub semantic_coverage: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing_facts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibrated_failure_risk: Option<f32>,
}

impl Default for FindingCertaintyV1 {
    fn default() -> Self {
        Self {
            rule_match: RuleMatchCertaintyV1::Inconclusive,
            semantic_coverage: 0.0,
            missing_facts: Vec::new(),
            calibrated_failure_risk: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalCandidateStatus {
    Candidate,
    Reviewed,
    Accepted,
    Rejected,
    Superseded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateGenerator {
    pub name: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateReviewDecision {
    Approve,
    Reject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateReview {
    pub reviewer_ref: String,
    pub reviewed_at: String,
    pub decision: CandidateReviewDecision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedactedCandidateInput {
    summary: String,
    redaction_policy_version: String,
    evidence: Vec<EvidenceRef>,
}

impl RedactedCandidateInput {
    pub fn new(
        summary: impl Into<String>,
        redaction_policy_version: impl Into<String>,
        evidence: Vec<EvidenceRef>,
    ) -> Result<Self> {
        let input = Self {
            summary: summary.into(),
            redaction_policy_version: redaction_policy_version.into(),
            evidence,
        };
        input.validate()?;
        Ok(input)
    }

    pub fn summary(&self) -> &str {
        &self.summary
    }

    pub fn redaction_policy_version(&self) -> &str {
        &self.redaction_policy_version
    }

    pub fn evidence(&self) -> &[EvidenceRef] {
        &self.evidence
    }

    pub fn validate(&self) -> Result<()> {
        if self.summary.trim().is_empty() {
            return Err(invalid_candidate_input("summary must not be empty"));
        }
        if self.redaction_policy_version.trim().is_empty() {
            return Err(invalid_candidate_input(
                "redaction_policy_version must not be empty",
            ));
        }
        if self.evidence.is_empty()
            || self.evidence.iter().any(|evidence| {
                evidence.kind.trim().is_empty() || evidence.identity.trim().is_empty()
            })
        {
            return Err(invalid_candidate_input(
                "at least one valid redaction evidence reference is required",
            ));
        }
        Ok(())
    }
}

fn invalid_candidate_input(message: impl Into<String>) -> TraceEvalError {
    TraceEvalError::InvalidCandidateInput {
        message: message.into(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalCandidate {
    pub schema_version: String,
    pub candidate_id: String,
    pub definition_hash: String,
    pub status: EvalCandidateStatus,
    pub source_trace_id: String,
    pub source_finding_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_cluster_refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_packet_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposed_input: Option<RedactedCandidateInput>,
    #[serde(default)]
    pub proposed_expected_behavior: Vec<String>,
    pub proposed_rubric: String,
    pub proposed_grader: String,
    pub generator: CandidateGenerator,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review: Option<CandidateReview>,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}
