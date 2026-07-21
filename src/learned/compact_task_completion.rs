use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    ContractError, EvaluationEvidenceCatalogV1, EvaluationEvidenceKindV1,
    EvaluationEvidenceLocationV1, LearnedAbstentionReasonV1, canonical_content_id,
    require_non_empty, require_sha256,
};

pub const COMPACT_TASK_COMPLETION_PROJECTION_SCHEMA_VERSION: &str =
    "traceeval.compact_task_completion_projection.v1";
pub const BINARY_TASK_COMPLETION_DECISION_SCHEMA_VERSION: &str =
    "traceeval.binary_task_completion_decision.v1";
const COMPACT_TASK_COMPLETION_PROJECTION_HASH_DOMAIN: &str =
    "traceeval.compact-task-completion-projection.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactTaskCompletionVariantV1 {
    GoalAndFinalResponse,
    MandatoryEvidence,
    MandatoryWithRecovery,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCompletionEvidenceLaneV1 {
    FinalResponse,
    Mandatory,
    FailureRecovery,
    GoalRelevant,
    StructuralMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceFactActorV1 {
    User,
    Assistant,
    Tool,
    System,
    ChildAgent,
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceFactKindV1 {
    UserRequest,
    UserClarification,
    AssistantMessage,
    ToolCall,
    ToolResult,
    Error,
    Verification,
    ArtifactMutation,
    ExternalAction,
    ChildAgentStart,
    ChildAgentResult,
    Cancellation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceFactStatusV1 {
    Unknown,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskCompletionGoalBundleV1 {
    pub primary_request: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub amendments: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub success_criteria: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requested_side_effects: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requested_verification: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_context: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub superseded_requirements: Vec<String>,
    pub token_count: u32,
}

impl TaskCompletionGoalBundleV1 {
    fn validate(&self) -> Result<(), ContractError> {
        require_non_empty(&self.primary_request, "goal primary_request", compact_error)?;
        if self.token_count == 0 {
            return Err(compact_error("goal token_count must be greater than zero"));
        }
        for value in self
            .amendments
            .iter()
            .chain(&self.success_criteria)
            .chain(&self.requested_side_effects)
            .chain(&self.requested_verification)
            .chain(&self.constraints)
            .chain(&self.agent_context)
            .chain(&self.superseded_requirements)
        {
            require_non_empty(value, "goal bundle entry", compact_error)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionTraceFactV1 {
    /// Short display alias such as `E0042`. It is local to this projection;
    /// `evidence_key` is the durable immutable-trace catalog identity.
    pub evidence_id: String,
    pub evidence_key: String,
    pub sequence: u32,
    pub actor: TraceFactActorV1,
    pub kind: TraceFactKindV1,
    pub status: TraceFactStatusV1,
    pub lane: TaskCompletionEvidenceLaneV1,
    pub mandatory: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    pub summary: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub structured_facts: BTreeMap<String, Value>,
    pub token_count: u32,
}

impl TaskCompletionTraceFactV1 {
    fn validate(&self) -> Result<(), ContractError> {
        validate_evidence_id(&self.evidence_id)?;
        require_non_empty(&self.evidence_key, "fact evidence_key", compact_error)?;
        require_non_empty(&self.summary, "fact summary", compact_error)?;
        if self.token_count == 0 {
            return Err(compact_error("fact token_count must be greater than zero"));
        }
        for (value, field) in [
            (self.span_id.as_deref(), "fact span_id"),
            (self.parent_span_id.as_deref(), "fact parent_span_id"),
            (self.tool_name.as_deref(), "fact tool_name"),
        ] {
            if let Some(value) = value {
                require_non_empty(value, field, compact_error)?;
            }
        }
        if self.mandatory
            && !matches!(
                self.lane,
                TaskCompletionEvidenceLaneV1::Mandatory
                    | TaskCompletionEvidenceLaneV1::FinalResponse
            )
        {
            return Err(compact_error(
                "mandatory facts must use the mandatory or final-response lane",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskCompletionRecoveryChainV1 {
    pub chain_id: String,
    pub evidence_ids: Vec<String>,
    pub token_count: u32,
}

impl TaskCompletionRecoveryChainV1 {
    fn validate(&self, facts: &BTreeSet<&str>) -> Result<(), ContractError> {
        require_non_empty(&self.chain_id, "recovery chain_id", compact_error)?;
        if self.evidence_ids.len() < 2 {
            return Err(compact_error(
                "recovery chains require at least two evidence facts",
            ));
        }
        if self.token_count == 0 {
            return Err(compact_error(
                "recovery chain token_count must be greater than zero",
            ));
        }
        let mut seen = BTreeSet::new();
        for evidence_id in &self.evidence_ids {
            validate_evidence_id(evidence_id)?;
            if !facts.contains(evidence_id.as_str()) {
                return Err(compact_error(format!(
                    "recovery chain references unknown fact {evidence_id}"
                )));
            }
            if !seen.insert(evidence_id) {
                return Err(compact_error(format!(
                    "recovery chain repeats fact {evidence_id}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactTaskCompletionTokenBudgetV1 {
    pub tokenizer_id: String,
    pub max_input_tokens: u32,
    pub original_tokens: u32,
    pub projected_tokens: u32,
    pub rubric_tokens: u32,
    pub goal_tokens: u32,
    pub final_response_tokens: u32,
    pub mandatory_tokens: u32,
    pub recovery_tokens: u32,
    pub goal_relevant_tokens: u32,
    pub metadata_tokens: u32,
}

impl CompactTaskCompletionTokenBudgetV1 {
    fn validate(&self, goal_tokens: u32) -> Result<(), ContractError> {
        require_non_empty(&self.tokenizer_id, "tokenizer_id", compact_error)?;
        if self.max_input_tokens == 0 {
            return Err(compact_error("max_input_tokens must be greater than zero"));
        }
        if self.projected_tokens > self.max_input_tokens {
            return Err(compact_error("projected input exceeds the token budget"));
        }
        if self.goal_tokens != goal_tokens {
            return Err(compact_error(
                "goal token accounting does not match the goal bundle",
            ));
        }
        let accounted = self
            .rubric_tokens
            .checked_add(self.goal_tokens)
            .and_then(|value| value.checked_add(self.final_response_tokens))
            .and_then(|value| value.checked_add(self.mandatory_tokens))
            .and_then(|value| value.checked_add(self.recovery_tokens))
            .and_then(|value| value.checked_add(self.goal_relevant_tokens))
            .and_then(|value| value.checked_add(self.metadata_tokens))
            .ok_or_else(|| compact_error("token accounting overflowed u32"))?;
        if accounted != self.projected_tokens {
            return Err(compact_error(
                "section token counts do not sum to projected_tokens",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactTaskCompletionProjectionStatsV1 {
    pub included_facts: u32,
    pub omitted_facts: u32,
    pub mandatory_facts: u32,
    pub mandatory_facts_omitted: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactTaskCompletionProjectionV1 {
    pub schema_version: String,
    pub projector_version: String,
    pub variant: CompactTaskCompletionVariantV1,
    pub target_key: String,
    pub target_revision: String,
    pub trace_context_binding_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_release_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_projection_release_id: Option<String>,
    pub projection_hash: String,
    pub goal: TaskCompletionGoalBundleV1,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub facts: Vec<TaskCompletionTraceFactV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recovery_chains: Vec<TaskCompletionRecoveryChainV1>,
    pub token_budget: CompactTaskCompletionTokenBudgetV1,
    pub stats: CompactTaskCompletionProjectionStatsV1,
    pub evidence_catalog: EvaluationEvidenceCatalogV1,
}

#[derive(Serialize)]
struct CompactProjectionIdentity<'a> {
    schema_version: &'a str,
    projector_version: &'a str,
    variant: CompactTaskCompletionVariantV1,
    target_key: &'a str,
    target_revision: &'a str,
    trace_context_binding_id: &'a str,
    context_release_id: Option<&'a str>,
    context_projection_release_id: Option<&'a str>,
    goal: &'a TaskCompletionGoalBundleV1,
    facts: &'a [TaskCompletionTraceFactV1],
    recovery_chains: &'a [TaskCompletionRecoveryChainV1],
    token_budget: &'a CompactTaskCompletionTokenBudgetV1,
    stats: &'a CompactTaskCompletionProjectionStatsV1,
    evidence: Vec<CompactEvidenceIdentity<'a>>,
}

#[derive(Serialize)]
struct CompactEvidenceIdentity<'a> {
    evidence_key: &'a str,
    evidence_kind: EvaluationEvidenceKindV1,
    location: &'a EvaluationEvidenceLocationV1,
    applicable_criterion_ids: &'a BTreeSet<String>,
}

impl CompactTaskCompletionProjectionV1 {
    /// Seals the projection after selection and exact tokenization. Evidence
    /// catalog identities are synchronized to the immutable projection hash.
    pub fn seal(mut self) -> Result<Self, ContractError> {
        self.projection_hash = self.compute_hash()?;
        self.evidence_catalog.target_key = self.target_key.clone();
        self.evidence_catalog.target_revision = self.target_revision.clone();
        self.evidence_catalog.projection_hash = self.projection_hash.clone();
        for record in self.evidence_catalog.entries.values_mut() {
            record.target_key = self.target_key.clone();
            record.target_revision = self.target_revision.clone();
            record.projection_hash = self.projection_hash.clone();
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != COMPACT_TASK_COMPLETION_PROJECTION_SCHEMA_VERSION {
            return Err(compact_error(
                "unsupported compact task-completion projection schema",
            ));
        }
        require_non_empty(&self.projector_version, "projector_version", compact_error)?;
        require_non_empty(&self.target_key, "target_key", compact_error)?;
        require_non_empty(&self.target_revision, "target_revision", compact_error)?;
        require_sha256(
            &self.trace_context_binding_id,
            "trace_context_binding_id",
            compact_error,
        )?;
        require_sha256(&self.projection_hash, "projection_hash", compact_error)?;
        if let Some(value) = &self.context_release_id {
            require_sha256(value, "context_release_id", compact_error)?;
        }
        if let Some(value) = &self.context_projection_release_id {
            require_sha256(value, "context_projection_release_id", compact_error)?;
        }
        self.goal.validate()?;
        self.token_budget.validate(self.goal.token_count)?;
        if self.stats.included_facts as usize != self.facts.len() {
            return Err(compact_error(
                "included_facts does not match the projected fact count",
            ));
        }
        if self.stats.mandatory_facts_omitted > self.stats.mandatory_facts {
            return Err(compact_error(
                "mandatory_facts_omitted cannot exceed mandatory_facts",
            ));
        }
        let included_mandatory = self.facts.iter().filter(|fact| fact.mandatory).count();
        if self
            .stats
            .mandatory_facts
            .saturating_sub(self.stats.mandatory_facts_omitted) as usize
            != included_mandatory
        {
            return Err(compact_error(
                "mandatory fact accounting does not match the projected facts",
            ));
        }
        if self.stats.mandatory_facts_omitted != 0
            && self.variant != CompactTaskCompletionVariantV1::GoalAndFinalResponse
        {
            return Err(compact_error(
                "only the goal-and-final-response ablation may omit mandatory facts",
            ));
        }
        let mut evidence_ids = BTreeSet::new();
        let mut evidence_keys = BTreeSet::new();
        for fact in &self.facts {
            fact.validate()?;
            if !evidence_ids.insert(fact.evidence_id.as_str()) {
                return Err(compact_error(format!(
                    "duplicate fact evidence_id {}",
                    fact.evidence_id
                )));
            }
            if !evidence_keys.insert(fact.evidence_key.as_str()) {
                return Err(compact_error(format!(
                    "duplicate fact evidence_key {}",
                    fact.evidence_key
                )));
            }
            if !self
                .evidence_catalog
                .entries
                .contains_key(&fact.evidence_key)
            {
                return Err(compact_error(format!(
                    "fact {} references unknown evidence key {}",
                    fact.evidence_id, fact.evidence_key
                )));
            }
        }
        let mut chain_ids = BTreeSet::new();
        for chain in &self.recovery_chains {
            chain.validate(&evidence_ids)?;
            if !chain_ids.insert(chain.chain_id.as_str()) {
                return Err(compact_error(format!(
                    "duplicate recovery chain {}",
                    chain.chain_id
                )));
            }
        }
        self.evidence_catalog.validate()?;
        if self.evidence_catalog.target_key != self.target_key
            || self.evidence_catalog.target_revision != self.target_revision
            || self.evidence_catalog.projection_hash != self.projection_hash
        {
            return Err(compact_error(
                "compact projection and evidence catalog identities do not match",
            ));
        }
        if self.compute_hash()? != self.projection_hash {
            return Err(compact_error(
                "projection_hash does not match compact projection content",
            ));
        }
        Ok(())
    }

    fn compute_hash(&self) -> Result<String, ContractError> {
        let evidence = self
            .evidence_catalog
            .entries
            .iter()
            .map(|(evidence_key, record)| CompactEvidenceIdentity {
                evidence_key,
                evidence_kind: record.evidence_kind,
                location: &record.location,
                applicable_criterion_ids: &record.applicable_criterion_ids,
            })
            .collect();
        canonical_content_id(
            COMPACT_TASK_COMPLETION_PROJECTION_HASH_DOMAIN,
            &CompactProjectionIdentity {
                schema_version: &self.schema_version,
                projector_version: &self.projector_version,
                variant: self.variant,
                target_key: &self.target_key,
                target_revision: &self.target_revision,
                trace_context_binding_id: &self.trace_context_binding_id,
                context_release_id: self.context_release_id.as_deref(),
                context_projection_release_id: self.context_projection_release_id.as_deref(),
                goal: &self.goal,
                facts: &self.facts,
                recovery_chains: &self.recovery_chains,
                token_budget: &self.token_budget,
                stats: &self.stats,
                evidence,
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskCompletionInferenceProvenanceV1 {
    pub runtime: String,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_version: Option<String>,
    pub tokenizer_id: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub latency_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_microusd: Option<u64>,
}

impl TaskCompletionInferenceProvenanceV1 {
    fn validate(&self) -> Result<(), ContractError> {
        require_non_empty(&self.runtime, "inference runtime", compact_error)?;
        require_non_empty(&self.model_id, "model_id", compact_error)?;
        require_non_empty(&self.tokenizer_id, "tokenizer_id", compact_error)?;
        if let Some(model_hash) = &self.model_hash {
            require_sha256(model_hash, "model_hash", compact_error)?;
        }
        if let Some(prompt_version) = &self.prompt_version {
            require_non_empty(prompt_version, "prompt_version", compact_error)?;
        }
        if self.input_tokens == 0 {
            return Err(compact_error("input_tokens must be greater than zero"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryTaskCompletionOutcomeV1 {
    Completed,
    Incomplete,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryTaskCompletionDecisionV1 {
    pub schema_version: String,
    pub evaluator_release_id: String,
    pub target_key: String,
    pub target_revision: String,
    pub trace_context_binding_id: String,
    pub projection_hash: String,
    pub outcome: BinaryTaskCompletionOutcomeV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_logit_difference: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probability_complete: Option<f64>,
    pub threshold: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abstention_reason: Option<LearnedAbstentionReasonV1>,
    pub inference: TaskCompletionInferenceProvenanceV1,
}

impl BinaryTaskCompletionDecisionV1 {
    pub fn validate_against(
        &self,
        projection: &CompactTaskCompletionProjectionV1,
    ) -> Result<(), ContractError> {
        projection.validate()?;
        if self.schema_version != BINARY_TASK_COMPLETION_DECISION_SCHEMA_VERSION {
            return Err(compact_error(
                "unsupported binary task-completion decision schema",
            ));
        }
        require_sha256(
            &self.evaluator_release_id,
            "evaluator_release_id",
            compact_error,
        )?;
        require_sha256(
            &self.trace_context_binding_id,
            "trace_context_binding_id",
            compact_error,
        )?;
        require_sha256(&self.projection_hash, "projection_hash", compact_error)?;
        if let Some(calibration_model_id) = &self.calibration_model_id {
            require_sha256(calibration_model_id, "calibration_model_id", compact_error)?;
        }
        if self.target_key != projection.target_key
            || self.target_revision != projection.target_revision
            || self.trace_context_binding_id != projection.trace_context_binding_id
            || self.projection_hash != projection.projection_hash
        {
            return Err(compact_error(
                "binary decision does not match its compact projection",
            ));
        }
        validate_probability(self.threshold, "threshold")?;
        if let Some(raw) = self.raw_logit_difference
            && !raw.is_finite()
        {
            return Err(compact_error("raw_logit_difference must be finite"));
        }
        if let Some(probability) = self.probability_complete {
            validate_probability(probability, "probability_complete")?;
        }
        match self.outcome {
            BinaryTaskCompletionOutcomeV1::Abstain => {
                if self.raw_logit_difference.is_some()
                    || self.probability_complete.is_some()
                    || self.abstention_reason.is_none()
                {
                    return Err(compact_error(
                        "abstention requires a reason and no classification score",
                    ));
                }
            }
            BinaryTaskCompletionOutcomeV1::Completed
            | BinaryTaskCompletionOutcomeV1::Incomplete => {
                if self.raw_logit_difference.is_none()
                    || self.probability_complete.is_none()
                    || self.abstention_reason.is_some()
                {
                    return Err(compact_error(
                        "decisive outcomes require logits and probability without abstention",
                    ));
                }
                let predicted_complete =
                    self.probability_complete.unwrap_or_default() >= self.threshold;
                if predicted_complete != (self.outcome == BinaryTaskCompletionOutcomeV1::Completed)
                {
                    return Err(compact_error(
                        "outcome is inconsistent with probability_complete and threshold",
                    ));
                }
            }
        }
        let known_ids = projection
            .facts
            .iter()
            .map(|fact| fact.evidence_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        for evidence_id in &self.evidence_ids {
            if !known_ids.contains(evidence_id.as_str()) {
                return Err(compact_error(format!(
                    "decision references unknown evidence id {evidence_id}"
                )));
            }
            if !seen.insert(evidence_id) {
                return Err(compact_error(format!(
                    "decision repeats evidence id {evidence_id}"
                )));
            }
        }
        if let Some(reason_code) = &self.reason_code {
            require_non_empty(reason_code, "reason_code", compact_error)?;
        }
        if let Some(explanation) = &self.explanation {
            require_non_empty(explanation, "explanation", compact_error)?;
        }
        self.inference.validate()
    }
}

fn validate_probability(value: f64, field: &str) -> Result<(), ContractError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(compact_error(format!(
            "{field} must be finite and within [0, 1]"
        )));
    }
    Ok(())
}

fn validate_evidence_id(value: &str) -> Result<(), ContractError> {
    let digits = value.strip_prefix('E').unwrap_or_default();
    if digits.len() < 4 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(compact_error(
            "evidence_id must use the projection-local E0001 form",
        ));
    }
    Ok(())
}

fn compact_error(message: impl Into<String>) -> ContractError {
    ContractError::InvalidTaskCompletion(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn projection() -> CompactTaskCompletionProjectionV1 {
        let placeholder = digest('0');
        let facts = vec![
            TaskCompletionTraceFactV1 {
                evidence_id: "E0001".into(),
                evidence_key: "request".into(),
                sequence: 1,
                actor: TraceFactActorV1::User,
                kind: TraceFactKindV1::UserRequest,
                status: TraceFactStatusV1::Succeeded,
                lane: TaskCompletionEvidenceLaneV1::Mandatory,
                mandatory: true,
                span_id: Some("root".into()),
                parent_span_id: None,
                tool_name: None,
                summary: "Fix authentication and run the tests.".into(),
                structured_facts: BTreeMap::new(),
                token_count: 10,
            },
            TaskCompletionTraceFactV1 {
                evidence_id: "E0002".into(),
                evidence_key: "test-result".into(),
                sequence: 2,
                actor: TraceFactActorV1::Tool,
                kind: TraceFactKindV1::Verification,
                status: TraceFactStatusV1::Succeeded,
                lane: TaskCompletionEvidenceLaneV1::Mandatory,
                mandatory: true,
                span_id: Some("tool-1".into()),
                parent_span_id: Some("root".into()),
                tool_name: Some("terminal".into()),
                summary: "284 tests passed, 0 failed.".into(),
                structured_facts: BTreeMap::new(),
                token_count: 9,
            },
        ];
        let entries = facts
            .iter()
            .map(|fact| {
                (
                    fact.evidence_key.clone(),
                    crate::learned::EvaluationEvidenceRecordV1 {
                        target_key: "trace-1".into(),
                        target_revision: "revision-1".into(),
                        projection_hash: placeholder.clone(),
                        evidence_kind: EvaluationEvidenceKindV1::Span,
                        location: EvaluationEvidenceLocationV1::Span {
                            span_id: fact.span_id.clone().unwrap(),
                        },
                        applicable_criterion_ids: BTreeSet::new(),
                    },
                )
            })
            .collect();
        CompactTaskCompletionProjectionV1 {
            schema_version: COMPACT_TASK_COMPLETION_PROJECTION_SCHEMA_VERSION.into(),
            projector_version: "perseval.task-completion-projector.v1".into(),
            variant: CompactTaskCompletionVariantV1::MandatoryEvidence,
            target_key: "trace-1".into(),
            target_revision: "revision-1".into(),
            trace_context_binding_id: digest('1'),
            context_release_id: Some(digest('2')),
            context_projection_release_id: Some(digest('3')),
            projection_hash: placeholder.clone(),
            goal: TaskCompletionGoalBundleV1 {
                primary_request: "Fix authentication.".into(),
                amendments: Vec::new(),
                success_criteria: vec!["Tests pass.".into()],
                requested_side_effects: Vec::new(),
                requested_verification: vec!["Run the test suite.".into()],
                constraints: Vec::new(),
                agent_context: vec!["Rust workspace.".into()],
                superseded_requirements: Vec::new(),
                token_count: 20,
            },
            facts,
            recovery_chains: Vec::new(),
            token_budget: CompactTaskCompletionTokenBudgetV1 {
                tokenizer_id: "smollm3-tokenizer".into(),
                max_input_tokens: 6_144,
                original_tokens: 26_711,
                projected_tokens: 69,
                rubric_tokens: 30,
                goal_tokens: 20,
                final_response_tokens: 0,
                mandatory_tokens: 19,
                recovery_tokens: 0,
                goal_relevant_tokens: 0,
                metadata_tokens: 0,
            },
            stats: CompactTaskCompletionProjectionStatsV1 {
                included_facts: 2,
                omitted_facts: 10,
                mandatory_facts: 2,
                mandatory_facts_omitted: 0,
            },
            evidence_catalog: EvaluationEvidenceCatalogV1 {
                target_key: "trace-1".into(),
                target_revision: "revision-1".into(),
                projection_hash: placeholder,
                entries,
            },
        }
    }

    fn inference() -> TaskCompletionInferenceProvenanceV1 {
        TaskCompletionInferenceProvenanceV1 {
            runtime: "llama.cpp-b9637".into(),
            model_id: "SmolLM3-3B-Q4_K_M".into(),
            model_hash: Some(digest('4')),
            prompt_version: Some("binary-completion-v1".into()),
            tokenizer_id: "smollm3-tokenizer".into(),
            input_tokens: 69,
            output_tokens: 1,
            latency_ms: 750,
            cost_microusd: Some(0),
        }
    }

    #[test]
    fn sealed_projection_validates_and_binds_catalog_content() {
        let sealed = projection().seal().unwrap();
        sealed.validate().unwrap();

        let mut changed = sealed.clone();
        changed
            .evidence_catalog
            .entries
            .get_mut("test-result")
            .unwrap()
            .location = EvaluationEvidenceLocationV1::Span {
            span_id: "different".into(),
        };
        assert!(changed.validate().is_err());
    }

    #[test]
    fn projection_rejects_omitted_mandatory_facts_and_bad_budget_math() {
        let mut omitted = projection();
        omitted.stats.mandatory_facts_omitted = 1;
        assert!(omitted.seal().is_err());

        let mut ablation = projection();
        ablation.variant = CompactTaskCompletionVariantV1::GoalAndFinalResponse;
        ablation.facts.remove(1);
        ablation.stats.included_facts = 1;
        ablation.stats.mandatory_facts_omitted = 1;
        ablation.token_budget.mandatory_tokens = 10;
        ablation.token_budget.projected_tokens = 60;
        ablation.seal().unwrap();

        let mut bad_budget = projection();
        bad_budget.token_budget.projected_tokens += 1;
        assert!(bad_budget.seal().is_err());
    }

    #[test]
    fn decisive_binary_result_is_threshold_consistent_and_evidence_bound() {
        let projection = projection().seal().unwrap();
        let decision = BinaryTaskCompletionDecisionV1 {
            schema_version: BINARY_TASK_COMPLETION_DECISION_SCHEMA_VERSION.into(),
            evaluator_release_id: digest('5'),
            target_key: projection.target_key.clone(),
            target_revision: projection.target_revision.clone(),
            trace_context_binding_id: projection.trace_context_binding_id.clone(),
            projection_hash: projection.projection_hash.clone(),
            outcome: BinaryTaskCompletionOutcomeV1::Completed,
            raw_logit_difference: Some(1.5),
            probability_complete: Some(0.82),
            threshold: 0.54,
            calibration_model_id: Some(digest('6')),
            evidence_ids: vec!["E0002".into()],
            reason_code: Some("verified_success".into()),
            explanation: None,
            abstention_reason: None,
            inference: inference(),
        };
        decision.validate_against(&projection).unwrap();

        let mut inconsistent = decision.clone();
        inconsistent.outcome = BinaryTaskCompletionOutcomeV1::Incomplete;
        assert!(inconsistent.validate_against(&projection).is_err());

        let mut invented = decision;
        invented.evidence_ids = vec!["E9999".into()];
        assert!(invented.validate_against(&projection).is_err());
    }

    #[test]
    fn abstention_has_no_synthetic_score() {
        let projection = projection().seal().unwrap();
        let decision = BinaryTaskCompletionDecisionV1 {
            schema_version: BINARY_TASK_COMPLETION_DECISION_SCHEMA_VERSION.into(),
            evaluator_release_id: digest('5'),
            target_key: projection.target_key.clone(),
            target_revision: projection.target_revision.clone(),
            trace_context_binding_id: projection.trace_context_binding_id.clone(),
            projection_hash: projection.projection_hash.clone(),
            outcome: BinaryTaskCompletionOutcomeV1::Abstain,
            raw_logit_difference: None,
            probability_complete: None,
            threshold: 0.5,
            calibration_model_id: None,
            evidence_ids: Vec::new(),
            reason_code: None,
            explanation: None,
            abstention_reason: Some(LearnedAbstentionReasonV1::ProviderUnavailable),
            inference: inference(),
        };
        decision.validate_against(&projection).unwrap();
    }
}
