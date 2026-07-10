use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const AGENT_BEHAVIOR_TRACE_SCHEMA_VERSION: &str = "traceeval.agent_behavior_trace.v1";
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
    pub evidence: EvidenceRef,
    pub changed: bool,
    #[serde(default)]
    pub verified: bool,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(default = "default_attempt")]
    pub attempt: u32,
    pub started_at: String,
    #[serde(default)]
    pub duration_ms: u64,
    pub status: ToolCallStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<NormalizedToolError>,
    #[serde(default)]
    pub approval_required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_outcome: Option<ApprovalOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_change: Option<StateChangeRef>,
    #[serde(default)]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotent: Option<bool>,
    #[serde(default)]
    pub mutating: bool,
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
    Succeeded,
    Failed,
    Refused,
    Escalated,
    Incomplete,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalOutcome {
    pub status: FinalOutcomeStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_summary: Option<String>,
    #[serde(default)]
    pub claimed_success: bool,
    #[serde(default)]
    pub resolution_present: bool,
    #[serde(default)]
    pub escalation_required: bool,
    #[serde(default)]
    pub escalation_performed: bool,
    #[serde(default)]
    pub failure_acknowledged: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
}

impl Default for FinalOutcome {
    fn default() -> Self {
        Self {
            status: FinalOutcomeStatus::Unknown,
            response_summary: None,
            claimed_success: false,
            resolution_present: false,
            escalation_required: false,
            escalation_performed: false,
            failure_acknowledged: false,
            evidence: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentBehaviorTrace {
    pub schema_version: String,
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
    pub evidence: Vec<EvidenceRef>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl AgentBehaviorTrace {
    pub fn new(trace_id: impl Into<String>) -> Self {
        Self {
            schema_version: AGENT_BEHAVIOR_TRACE_SCHEMA_VERSION.to_string(),
            trace_id: trace_id.into(),
            input_summary: None,
            turns: Vec::new(),
            tool_calls: Vec::new(),
            policy_decisions: Vec::new(),
            final_outcome: FinalOutcome::default(),
            evidence: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    pub failure_signature: String,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalCandidate {
    pub schema_version: String,
    pub candidate_id: String,
    pub status: EvalCandidateStatus,
    pub source_trace_id: String,
    pub source_finding_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposed_input: Option<String>,
    #[serde(default)]
    pub proposed_expected_behavior: Vec<String>,
    pub proposed_rubric: String,
    pub proposed_grader: String,
    pub generator: CandidateGenerator,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}
