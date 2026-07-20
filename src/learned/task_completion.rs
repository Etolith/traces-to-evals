use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    model::{SourceSpanStatus, Span, SpanKind, Trace},
    providers::chat::{ChatClient, ChatRequest, ResponseSchema},
};

use super::{
    AgentCapabilityV1, AgentContextReleaseV1, ContextFieldV1, ContextProjectionClassV1,
    ContextProjectionV1, ContractError, EvaluationCriterionV1, EvaluationEvidenceCatalogV1,
    EvaluationEvidenceCitationV1, EvaluationEvidenceKindV1, EvaluationEvidenceLocationV1,
    EvaluationEvidenceRecordV1, EvaluationImplementationV1, EvaluationTargetKind,
    EvaluatorReleaseSpecV1, LEARNED_EVALUATION_SCHEMA_VERSION, LearnedAbstentionReasonV1,
    LearnedEvaluationV1, LearnedTaskKind, LearnedVerdictV1, ProviderResponseEnvelopeV1,
    SuccessCriterionImportanceV1, TraceContextBindingResolutionV1, TraceContextBindingV1,
    canonical_content_id, require_non_empty, require_sha256,
};

pub const TASK_COMPLETION_PROJECTION_SCHEMA_VERSION: &str =
    "traceeval.task_completion_projection.v1";
pub const TASK_COMPLETION_JUDGMENT_SCHEMA_VERSION: &str = "traceeval.task_completion_judgment.v1";
pub const TASK_COMPLETION_PROJECTOR_VERSION: &str = "traceeval.task-completion-projector.v2";
const TASK_COMPLETION_PROJECTOR_VERSION_V1: &str = "traceeval.task-completion-projector.v1";
const TASK_COMPLETION_PROJECTION_HASH_DOMAIN: &str = "traceeval.task-completion-projection.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCompletionContentPolicyV1 {
    /// Project topology, statuses, identities, and hashes only. No free-form
    /// trace or context text may be sent to a model.
    StructuredOnly,
    /// The caller asserts that selected context and trace summaries were
    /// reviewed and redacted before this projector receives them.
    PreRedactedSummaries,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionDeclaredFieldV1 {
    pub field_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionCriterionSpecV1 {
    pub criterion_id: String,
    pub field_id: String,
    pub importance: SuccessCriterionImportanceV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub required_evidence_kinds: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionCapabilityV1 {
    pub capability_id: String,
    pub field_id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowed_operations: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub prohibited_operations: BTreeSet<String>,
    pub requires_approval: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionToolObservationV1 {
    pub sequence: u32,
    pub span_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    pub tool_name: String,
    pub source_status: SourceSpanStatus,
    pub error_present: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub structured_facts: BTreeMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_evidence_key: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub input_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_evidence_key: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub output_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at_unix_nano: Option<u64>,
    pub evidence_key: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionTraceObservationV1 {
    pub trace_id: String,
    pub span_count: u32,
    pub tool_span_count: u32,
    pub error_span_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_span_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_status: Option<SourceSpanStatus>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub structured_facts: BTreeMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_summary: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub input_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_summary: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub output_truncated: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionProjectionV1 {
    pub schema_version: String,
    pub projector_version: String,
    pub content_policy: TaskCompletionContentPolicyV1,
    pub max_tool_observations: u32,
    pub max_summary_bytes: u32,
    pub target_key: String,
    pub target_revision: String,
    pub trace_context_binding_id: String,
    pub context_release_id: Option<String>,
    pub context_projection_release_id: Option<String>,
    pub projection_hash: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub declared_fields: Vec<TaskCompletionDeclaredFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub criteria: Vec<TaskCompletionCriterionSpecV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<TaskCompletionCapabilityV1>,
    pub trace: TaskCompletionTraceObservationV1,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<TaskCompletionToolObservationV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing_required_context: Vec<String>,
    pub truncated: bool,
    pub evidence_catalog: EvaluationEvidenceCatalogV1,
}

#[derive(Serialize)]
struct ProjectionIdentity<'a> {
    schema_version: &'a str,
    projector_version: &'a str,
    content_policy: TaskCompletionContentPolicyV1,
    max_tool_observations: u32,
    max_summary_bytes: u32,
    target_key: &'a str,
    target_revision: &'a str,
    trace_context_binding_id: &'a str,
    context_release_id: Option<&'a str>,
    context_projection_release_id: Option<&'a str>,
    declared_fields: &'a [TaskCompletionDeclaredFieldV1],
    criteria: &'a [TaskCompletionCriterionSpecV1],
    capabilities: &'a [TaskCompletionCapabilityV1],
    trace: &'a TaskCompletionTraceObservationV1,
    tools: &'a [TaskCompletionToolObservationV1],
    missing_required_context: &'a [String],
    truncated: bool,
}

impl TaskCompletionProjectionV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != TASK_COMPLETION_PROJECTION_SCHEMA_VERSION {
            return Err(task_error("unsupported task-completion projection schema"));
        }
        if !matches!(
            self.projector_version.as_str(),
            TASK_COMPLETION_PROJECTOR_VERSION | TASK_COMPLETION_PROJECTOR_VERSION_V1
        ) {
            return Err(task_error("unsupported task-completion projector version"));
        }
        if self.max_tool_observations == 0 || self.max_summary_bytes == 0 {
            return Err(task_error("projection bounds must be greater than zero"));
        }
        require_non_empty(&self.target_key, "target_key", task_error)?;
        require_non_empty(&self.target_revision, "target_revision", task_error)?;
        require_sha256(
            &self.trace_context_binding_id,
            "trace_context_binding_id",
            task_error,
        )?;
        require_sha256(&self.projection_hash, "projection_hash", task_error)?;
        if let Some(release_id) = &self.context_release_id {
            require_sha256(release_id, "context_release_id", task_error)?;
        }
        if let Some(release_id) = &self.context_projection_release_id {
            require_sha256(release_id, "context_projection_release_id", task_error)?;
        }
        if self.trace.trace_id.trim().is_empty() {
            return Err(task_error("trace_id must not be empty"));
        }
        if self.tools.len() > u32::MAX as usize {
            return Err(task_error("tool observation count exceeds u32"));
        }
        for key in self
            .trace
            .evidence_keys
            .iter()
            .chain(self.tools.iter().flat_map(tool_evidence_keys))
        {
            if !self.evidence_catalog.entries.contains_key(key) {
                return Err(task_error(format!(
                    "projection observation references unknown evidence key {key}"
                )));
            }
        }
        let mut criterion_ids = BTreeSet::new();
        for criterion in &self.criteria {
            require_non_empty(&criterion.criterion_id, "criterion_id", task_error)?;
            if !criterion_ids.insert(criterion.criterion_id.as_str()) {
                return Err(task_error(format!(
                    "duplicate task criterion {}",
                    criterion.criterion_id
                )));
            }
        }
        self.evidence_catalog.validate()?;
        if self.evidence_catalog.target_key != self.target_key
            || self.evidence_catalog.target_revision != self.target_revision
            || self.evidence_catalog.projection_hash != self.projection_hash
        {
            return Err(task_error(
                "projection and evidence catalog identities do not match",
            ));
        }
        if self.content_policy == TaskCompletionContentPolicyV1::StructuredOnly
            && (self.trace.input_summary.is_some()
                || self.trace.output_summary.is_some()
                || self
                    .tools
                    .iter()
                    .any(|tool| tool.input_summary.is_some() || tool.output_summary.is_some()))
        {
            return Err(task_error(
                "structured-only projection cannot contain free-form summaries",
            ));
        }
        if self.projector_version == TASK_COMPLETION_PROJECTOR_VERSION_V1
            && (!self.trace.structured_facts.is_empty()
                || self.trace.input_truncated
                || self.trace.output_truncated
                || self.tools.iter().any(|tool| {
                    !tool.structured_facts.is_empty()
                        || tool.input_summary.is_some()
                        || tool.output_summary.is_some()
                        || tool.input_evidence_key.is_some()
                        || tool.output_evidence_key.is_some()
                        || tool.input_truncated
                        || tool.output_truncated
                }))
        {
            return Err(task_error(
                "v1 task-completion projections cannot contain v2 observations",
            ));
        }
        validate_structured_facts(&self.trace.structured_facts)?;
        for (summary, field) in [
            (self.trace.input_summary.as_deref(), "trace input"),
            (self.trace.output_summary.as_deref(), "trace output"),
        ] {
            if let Some(summary) = summary
                && summary.len() > self.max_summary_bytes as usize
            {
                return Err(task_error(format!(
                    "{field} exceeds the configured summary bound"
                )));
            }
        }
        for tool in &self.tools {
            validate_structured_facts(&tool.structured_facts)?;
            validate_projected_summary(
                tool.input_summary.as_deref(),
                tool.input_evidence_key.as_deref(),
                self.max_summary_bytes,
                "tool input",
            )?;
            validate_projected_summary(
                tool.output_summary.as_deref(),
                tool.output_evidence_key.as_deref(),
                self.max_summary_bytes,
                "tool output",
            )?;
        }
        let recomputed = self.compute_hash()?;
        if recomputed != self.projection_hash {
            return Err(task_error(
                "projection_hash does not match projection content",
            ));
        }
        Ok(())
    }

    fn compute_hash(&self) -> Result<String, ContractError> {
        canonical_content_id(
            TASK_COMPLETION_PROJECTION_HASH_DOMAIN,
            &ProjectionIdentity {
                schema_version: &self.schema_version,
                projector_version: &self.projector_version,
                content_policy: self.content_policy,
                max_tool_observations: self.max_tool_observations,
                max_summary_bytes: self.max_summary_bytes,
                target_key: &self.target_key,
                target_revision: &self.target_revision,
                trace_context_binding_id: &self.trace_context_binding_id,
                context_release_id: self.context_release_id.as_deref(),
                context_projection_release_id: self.context_projection_release_id.as_deref(),
                declared_fields: &self.declared_fields,
                criteria: &self.criteria,
                capabilities: &self.capabilities,
                trace: &self.trace,
                tools: &self.tools,
                missing_required_context: &self.missing_required_context,
                truncated: self.truncated,
            },
        )
    }

    pub fn projector_release_id(&self) -> Result<String, ContractError> {
        TaskCompletionProjectorV1 {
            content_policy: self.content_policy,
            max_tool_observations: self.max_tool_observations,
            max_summary_bytes: self.max_summary_bytes,
        }
        .release_id_for_version(&self.projector_version)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCompletionOutcomeV1 {
    Completed,
    Partial,
    Failed,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCompletionCriterionOutcomeV1 {
    Satisfied,
    PartiallySatisfied,
    Unsatisfied,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionCriterionJudgmentV1 {
    pub criterion_id: String,
    pub outcome: TaskCompletionCriterionOutcomeV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionJudgmentV1 {
    pub schema_version: String,
    pub outcome: TaskCompletionOutcomeV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_reported_confidence: Option<f64>,
    pub explanation: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_keys: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub criteria: Vec<TaskCompletionCriterionJudgmentV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abstention_reason: Option<LearnedAbstentionReasonV1>,
}

impl TaskCompletionJudgmentV1 {
    pub fn validate_against(
        &self,
        projection: &TaskCompletionProjectionV1,
    ) -> Result<(), ContractError> {
        projection.validate()?;
        if self.schema_version != TASK_COMPLETION_JUDGMENT_SCHEMA_VERSION {
            return Err(task_error("unsupported task-completion judgment schema"));
        }
        require_non_empty(&self.explanation, "explanation", task_error)?;
        validate_probability(self.completion_score, "completion_score")?;
        validate_probability(self.model_reported_confidence, "model_reported_confidence")?;
        let declared = projection
            .criteria
            .iter()
            .map(|criterion| (criterion.criterion_id.as_str(), criterion.importance))
            .collect::<BTreeMap<_, _>>();
        let mut seen = BTreeSet::new();
        for criterion in &self.criteria {
            if !declared.contains_key(criterion.criterion_id.as_str()) {
                return Err(task_error(format!(
                    "judgment references unknown criterion {}",
                    criterion.criterion_id
                )));
            }
            if !seen.insert(criterion.criterion_id.as_str()) {
                return Err(task_error(format!(
                    "duplicate criterion judgment {}",
                    criterion.criterion_id
                )));
            }
            validate_probability(criterion.score, "criterion score")?;
            validate_evidence_keys(&criterion.evidence_keys, projection)?;
            let decisive = !matches!(
                criterion.outcome,
                TaskCompletionCriterionOutcomeV1::Unresolved
            );
            if decisive && criterion.evidence_keys.is_empty() {
                return Err(task_error(
                    "decisive criterion judgment requires observed evidence",
                ));
            }
        }
        validate_evidence_keys(&self.evidence_keys, projection)?;

        match self.outcome {
            TaskCompletionOutcomeV1::Abstain => {
                if self.abstention_reason.is_none() || self.completion_score.is_some() {
                    return Err(task_error(
                        "abstention requires a reason and cannot include a completion score",
                    ));
                }
                if self.criteria.iter().any(|criterion| {
                    !matches!(
                        criterion.outcome,
                        TaskCompletionCriterionOutcomeV1::Unresolved
                    )
                }) {
                    return Err(task_error(
                        "abstention cannot make decisive criterion claims",
                    ));
                }
            }
            TaskCompletionOutcomeV1::Completed
            | TaskCompletionOutcomeV1::Partial
            | TaskCompletionOutcomeV1::Failed => {
                if !projection.missing_required_context.is_empty() || projection.truncated {
                    return Err(task_error(
                        "decisive task judgment is forbidden for incomplete projections",
                    ));
                }
                if self.abstention_reason.is_some()
                    || self.completion_score.is_none()
                    || self.evidence_keys.is_empty()
                {
                    return Err(task_error(
                        "decisive task judgment requires score and observed evidence",
                    ));
                }
                for (criterion_id, importance) in &declared {
                    if *importance == SuccessCriterionImportanceV1::Must
                        && !seen.contains(criterion_id)
                    {
                        return Err(task_error(format!(
                            "missing must-criterion judgment {criterion_id}"
                        )));
                    }
                }
            }
        }

        let must_outcomes = self.criteria.iter().filter(|criterion| {
            declared.get(criterion.criterion_id.as_str())
                == Some(&SuccessCriterionImportanceV1::Must)
        });
        match self.outcome {
            TaskCompletionOutcomeV1::Completed => {
                if must_outcomes.into_iter().any(|criterion| {
                    criterion.outcome != TaskCompletionCriterionOutcomeV1::Satisfied
                }) {
                    return Err(task_error(
                        "completed requires every must criterion to be satisfied",
                    ));
                }
            }
            TaskCompletionOutcomeV1::Partial => {
                let any_progress = self.criteria.iter().any(|criterion| {
                    matches!(
                        criterion.outcome,
                        TaskCompletionCriterionOutcomeV1::Satisfied
                            | TaskCompletionCriterionOutcomeV1::PartiallySatisfied
                    )
                });
                let any_gap = self.criteria.iter().any(|criterion| {
                    criterion.outcome != TaskCompletionCriterionOutcomeV1::Satisfied
                });
                if !any_progress || !any_gap {
                    return Err(task_error(
                        "partial requires both observed progress and an unresolved gap",
                    ));
                }
            }
            TaskCompletionOutcomeV1::Failed => {
                if !must_outcomes.into_iter().any(|criterion| {
                    criterion.outcome == TaskCompletionCriterionOutcomeV1::Unsatisfied
                }) {
                    return Err(task_error("failed requires an unsatisfied must criterion"));
                }
            }
            TaskCompletionOutcomeV1::Abstain => {}
        }
        Ok(())
    }

    pub fn into_learned_evaluation(
        mut self,
        projection: &TaskCompletionProjectionV1,
        binding: &TraceContextBindingV1,
        evaluator_release: &EvaluatorReleaseSpecV1,
    ) -> Result<LearnedEvaluationV1, ContractError> {
        deduplicate_evidence_keys(&mut self.evidence_keys);
        for criterion in &mut self.criteria {
            deduplicate_evidence_keys(&mut criterion.evidence_keys);
        }
        self.validate_against(projection)?;
        validate_release_and_binding(projection, binding, evaluator_release)?;
        let verdict = match self.outcome {
            TaskCompletionOutcomeV1::Completed => LearnedVerdictV1::Pass,
            TaskCompletionOutcomeV1::Partial | TaskCompletionOutcomeV1::Failed => {
                LearnedVerdictV1::Fail
            }
            TaskCompletionOutcomeV1::Abstain => LearnedVerdictV1::Abstain,
        };
        let label = match self.outcome {
            TaskCompletionOutcomeV1::Completed => Some("completed".into()),
            TaskCompletionOutcomeV1::Partial => Some("partial".into()),
            TaskCompletionOutcomeV1::Failed => Some("failed".into()),
            TaskCompletionOutcomeV1::Abstain => None,
        };
        let criteria = self
            .criteria
            .iter()
            .map(|criterion| EvaluationCriterionV1 {
                criterion_id: criterion.criterion_id.clone(),
                label: criterion_label(criterion.outcome.clone()).into(),
                score: criterion.score,
                passed: criterion.outcome == TaskCompletionCriterionOutcomeV1::Satisfied,
                evidence_keys: criterion.evidence_keys.clone(),
            })
            .collect::<Vec<_>>();
        let mut evidence = Vec::new();
        for key in &self.evidence_keys {
            evidence.push(citation_for(key, None, projection)?);
        }
        for criterion in &self.criteria {
            for key in &criterion.evidence_keys {
                evidence.push(citation_for(
                    key,
                    Some(criterion.criterion_id.clone()),
                    projection,
                )?);
            }
        }
        evidence.sort_by(|left, right| {
            (&left.evidence_key, &left.criterion_id)
                .cmp(&(&right.evidence_key, &right.criterion_id))
        });
        evidence.dedup_by(|left, right| {
            left.evidence_key == right.evidence_key && left.criterion_id == right.criterion_id
        });

        let evaluation = LearnedEvaluationV1 {
            schema_version: LEARNED_EVALUATION_SCHEMA_VERSION.into(),
            evaluator_release_id: evaluator_release.release_id()?,
            target_key: projection.target_key.clone(),
            target_revision: projection.target_revision.clone(),
            trace_context_binding_id: binding.binding_id()?,
            projection_hash: projection.projection_hash.clone(),
            verdict,
            label,
            score: self.completion_score,
            model_reported_confidence: self.model_reported_confidence,
            explanation: self.explanation,
            evidence,
            criteria,
            abstention_reason: self.abstention_reason,
        };
        evaluation.validate_against(&projection.evidence_catalog)?;
        Ok(evaluation)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionExecutionV1 {
    pub evaluation: LearnedEvaluationV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderResponseEnvelopeV1>,
}

fn tool_evidence_keys(tool: &TaskCompletionToolObservationV1) -> impl Iterator<Item = &String> {
    std::iter::once(&tool.evidence_key)
        .chain(tool.input_evidence_key.iter())
        .chain(tool.output_evidence_key.iter())
}

fn validate_projected_summary(
    summary: Option<&str>,
    evidence_key: Option<&str>,
    maximum_bytes: u32,
    field: &str,
) -> Result<(), ContractError> {
    match (summary, evidence_key) {
        (Some(summary), Some(evidence_key)) => {
            require_non_empty(summary, field, task_error)?;
            require_non_empty(evidence_key, &format!("{field} evidence key"), task_error)?;
            if summary.len() > maximum_bytes as usize {
                return Err(task_error(format!(
                    "{field} exceeds the configured summary bound"
                )));
            }
        }
        (None, None) => {}
        _ => {
            return Err(task_error(format!(
                "{field} and its evidence key must be present together"
            )));
        }
    }
    Ok(())
}

fn validate_structured_facts(facts: &BTreeMap<String, Value>) -> Result<(), ContractError> {
    for (key, value) in facts {
        if !is_task_completion_fact_key(key) || !is_bounded_fact_value(value) {
            return Err(task_error(format!(
                "unsupported or unbounded task-completion fact {key}"
            )));
        }
    }
    Ok(())
}

fn task_completion_structured_facts(
    attributes: &BTreeMap<String, Value>,
) -> BTreeMap<String, Value> {
    attributes
        .iter()
        .filter_map(|(key, value)| {
            let normalized = key.to_ascii_lowercase();
            (is_task_completion_fact_key(&normalized) && is_bounded_fact_value(value))
                .then(|| (normalized, value.clone()))
        })
        .collect()
}

fn is_task_completion_fact_key(key: &str) -> bool {
    matches!(
        key,
        "gen_ai.operation.name"
            | "gen_ai.tool.name"
            | "gen_ai.tool.status"
            | "agent.operation"
            | "agent.operation.effect"
            | "agent.operation.retry_safety"
            | "agent.tool.requirement"
            | "agent.tool.status"
            | "agent.approval.required"
            | "agent.approval.outcome"
            | "agent.state.observation"
            | "agent.final.status"
            | "final.status"
            | "agent.escalation.status"
            | "final.escalation.status"
            | "agent.outcome.claim.status"
            | "final.outcome.claim.status"
            | "tool.status"
            | "tool.result.success"
            | "tool.timeout"
            | "tool.cancelled"
            | "tool.operation"
            | "tool.effect"
            | "tool.retry_safety"
            | "tool.requirement"
            | "tool.approval.required"
            | "tool.approval.outcome"
            | "tool.state.observation"
            | "operation"
            | "operation.name"
            | "operation.effect"
            | "operation.retry_safety"
            | "operation.requirement"
            | "execution.status"
            | "execution.timeout"
            | "error.type"
            | "error.code"
            | "error.retryable"
            | "tool.error.kind"
            | "tool.error.code"
            | "tool.error.retryable"
            | "exception.type"
            | "exception.escaped"
            | "http.status_code"
            | "rpc.status_code"
            | "result.success"
            | "result.ok"
            | "policy.outcome"
            | "guardrail.outcome"
    )
}

fn is_bounded_fact_value(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => true,
        Value::String(value) => value.chars().count() <= 256,
        Value::Array(_) | Value::Object(_) => false,
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn project_span_summary(
    content_policy: TaskCompletionContentPolicyV1,
    value: Option<&str>,
    evidence_prefix: &str,
    evidence_kind: EvaluationEvidenceKindV1,
    span_id: &str,
    maximum_bytes: u32,
    pending_records: &mut BTreeMap<
        String,
        (EvaluationEvidenceKindV1, EvaluationEvidenceLocationV1),
    >,
) -> (Option<String>, Option<String>, bool) {
    if content_policy != TaskCompletionContentPolicyV1::PreRedactedSummaries {
        return (None, None, false);
    }
    let Some(value) = value else {
        return (None, None, false);
    };
    let truncated = value.len() > maximum_bytes as usize;
    let summary = bound_utf8(value, maximum_bytes);
    if summary.is_empty() {
        return (None, None, truncated);
    }
    let key = evidence_key(evidence_prefix, span_id);
    pending_records.insert(
        key.clone(),
        (
            evidence_kind,
            EvaluationEvidenceLocationV1::Segment {
                span_id: span_id.into(),
                start_byte: 0,
                end_byte: u32::try_from(summary.len()).unwrap_or(u32::MAX),
            },
        ),
    );
    (Some(summary), Some(key), truncated)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskCompletionProjectorV1 {
    pub content_policy: TaskCompletionContentPolicyV1,
    pub max_tool_observations: u32,
    pub max_summary_bytes: u32,
}

impl Default for TaskCompletionProjectorV1 {
    fn default() -> Self {
        Self {
            content_policy: TaskCompletionContentPolicyV1::StructuredOnly,
            max_tool_observations: 256,
            max_summary_bytes: 4_096,
        }
    }
}

impl TaskCompletionProjectorV1 {
    pub fn release_id(&self) -> Result<String, ContractError> {
        self.release_id_for_version(TASK_COMPLETION_PROJECTOR_VERSION)
    }

    fn release_id_for_version(&self, projector_version: &str) -> Result<String, ContractError> {
        if self.max_tool_observations == 0 || self.max_summary_bytes == 0 {
            return Err(task_error("projector bounds must be greater than zero"));
        }
        let identity = match projector_version {
            TASK_COMPLETION_PROJECTOR_VERSION_V1 => serde_json::json!({
                "projector_version": TASK_COMPLETION_PROJECTOR_VERSION_V1,
                "content_policy": self.content_policy,
                "max_tool_observations": self.max_tool_observations,
                "max_summary_bytes": self.max_summary_bytes,
                "truncation_policy": "abstain_on_material_truncation",
                "ordering": "start_time_unix_nano_then_span_id",
            }),
            TASK_COMPLETION_PROJECTOR_VERSION => serde_json::json!({
                "projector_version": TASK_COMPLETION_PROJECTOR_VERSION,
                "content_policy": self.content_policy,
                "max_tool_observations": self.max_tool_observations,
                "max_summary_bytes": self.max_summary_bytes,
                "truncation_policy": "abstain_on_material_truncation",
                "ordering": "start_time_unix_nano_then_span_id",
                "structured_fact_policy": "task_completion_allowlist_v1",
                "authorized_tool_content": "bounded_input_output_segments_v1",
                "content_truncation_policy": "field_level_flags_and_material_abstention_v1",
            }),
            _ => return Err(task_error("unsupported task-completion projector version")),
        };
        canonical_content_id("traceeval.task-completion-projector-release.v1", &identity)
    }

    pub fn project(
        &self,
        target_key: &str,
        target_revision: &str,
        binding: &TraceContextBindingV1,
        context: Option<&AgentContextReleaseV1>,
        context_projection: Option<&ContextProjectionV1>,
        trace: &Trace,
    ) -> Result<TaskCompletionProjectionV1, ContractError> {
        require_non_empty(target_key, "target_key", task_error)?;
        require_non_empty(target_revision, "target_revision", task_error)?;
        require_non_empty(&trace.id, "trace_id", task_error)?;
        binding.validate()?;
        if binding.target_key != target_key || binding.target_revision != target_revision {
            return Err(task_error(
                "binding target does not match requested target revision",
            ));
        }
        if self.max_tool_observations == 0 || self.max_summary_bytes == 0 {
            return Err(task_error("projector bounds must be greater than zero"));
        }

        let binding_id = binding.binding_id()?;
        let mut missing = Vec::new();
        let mut context_release_id = None;
        let mut context_projection_release_id = None;
        let mut declared_fields = Vec::new();
        let mut criteria = Vec::new();
        let mut capabilities = Vec::new();
        match binding.resolution {
            TraceContextBindingResolutionV1::Resolved => {
                let Some(context) = context else {
                    missing.push("resolved_context_release_unavailable".into());
                    return self.finish_projection(
                        target_key,
                        target_revision,
                        binding_id,
                        None,
                        None,
                        declared_fields,
                        criteria,
                        capabilities,
                        missing,
                        trace,
                    );
                };
                context.validate()?;
                let release_id = context.release_id()?;
                if binding.agent_context_release_id.as_deref() != Some(release_id.as_str()) {
                    return Err(task_error(
                        "resolved binding points to a different context release",
                    ));
                }
                context_release_id = Some(release_id);
                let Some(context_projection) = context_projection else {
                    missing.push("context_projection_unavailable".into());
                    return self.finish_projection(
                        target_key,
                        target_revision,
                        binding_id,
                        context_release_id,
                        None,
                        declared_fields,
                        criteria,
                        capabilities,
                        missing,
                        trace,
                    );
                };
                context_projection.validate_against(context)?;
                let projection_id = context_projection.release_id()?;
                context_projection_release_id = Some(projection_id);
                let include_content = self.content_policy
                    == TaskCompletionContentPolicyV1::PreRedactedSummaries
                    && context_projection.projection_class
                        != ContextProjectionClassV1::StructuralOnly;

                project_field(
                    &mut declared_fields,
                    &context.intent.purpose,
                    context_projection,
                    include_content,
                );
                for field in context
                    .intent
                    .supported_tasks
                    .iter()
                    .chain(context.intent.explicit_non_goals.iter())
                    .chain(context.intent.refusal_requirements.iter())
                    .chain(context.intent.escalation_requirements.iter())
                    .chain(context.intent.acceptable_partial_completion.iter())
                    .chain(context.evaluation_context.required_evidence_types.iter())
                {
                    project_field(
                        &mut declared_fields,
                        field,
                        context_projection,
                        include_content,
                    );
                }
                for criterion in &context.intent.success_criteria {
                    if context_projection
                        .included_field_ids
                        .contains(&criterion.metadata.field_id)
                    {
                        criteria.push(TaskCompletionCriterionSpecV1 {
                            criterion_id: criterion.criterion_id.clone(),
                            field_id: criterion.metadata.field_id.clone(),
                            importance: criterion.importance,
                            description: include_content.then(|| criterion.description.clone()),
                            required_evidence_kinds: criterion.required_evidence_kinds.clone(),
                        });
                    }
                }
                for capability in &context.capabilities {
                    if context_projection
                        .included_field_ids
                        .contains(&capability.metadata.field_id)
                    {
                        capabilities.push(project_capability(capability));
                    }
                }
                if !include_content {
                    missing.push("task_intent_content_not_authorized".into());
                }
                if criteria.is_empty() {
                    missing.push("success_criteria_unavailable".into());
                } else if criteria
                    .iter()
                    .any(|criterion| criterion.description.is_none())
                {
                    missing.push("success_criteria_content_unavailable".into());
                }
            }
            TraceContextBindingResolutionV1::Unresolved => {
                missing.push("context_binding_unresolved".into());
            }
            TraceContextBindingResolutionV1::Ambiguous => {
                missing.push("context_binding_ambiguous".into());
            }
        }
        declared_fields.sort_by(|left, right| left.field_id.cmp(&right.field_id));
        criteria.sort_by(|left, right| left.criterion_id.cmp(&right.criterion_id));
        capabilities.sort_by(|left, right| left.capability_id.cmp(&right.capability_id));
        missing.sort();
        missing.dedup();
        self.finish_projection(
            target_key,
            target_revision,
            binding_id,
            context_release_id,
            context_projection_release_id,
            declared_fields,
            criteria,
            capabilities,
            missing,
            trace,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_projection(
        &self,
        target_key: &str,
        target_revision: &str,
        binding_id: String,
        context_release_id: Option<String>,
        context_projection_release_id: Option<String>,
        declared_fields: Vec<TaskCompletionDeclaredFieldV1>,
        criteria: Vec<TaskCompletionCriterionSpecV1>,
        capabilities: Vec<TaskCompletionCapabilityV1>,
        missing_required_context: Vec<String>,
        trace: &Trace,
    ) -> Result<TaskCompletionProjectionV1, ContractError> {
        let mut spans = trace.spans.iter().collect::<Vec<_>>();
        spans.sort_by(|left, right| {
            (left.start_time_unix_nano, &left.id).cmp(&(right.start_time_unix_nano, &right.id))
        });
        let terminal = spans
            .iter()
            .copied()
            .filter(|span| span.kind == SpanKind::Agent || span.parent_id.is_none())
            .max_by(|left, right| {
                (left.end_time_unix_nano, &left.id).cmp(&(right.end_time_unix_nano, &right.id))
            });
        let mut trace_observation = TaskCompletionTraceObservationV1 {
            trace_id: trace.id.clone(),
            span_count: u32::try_from(spans.len()).unwrap_or(u32::MAX),
            tool_span_count: 0,
            error_span_count: u32::try_from(
                spans
                    .iter()
                    .filter(|span| span.source_status == SourceSpanStatus::Error)
                    .count(),
            )
            .unwrap_or(u32::MAX),
            terminal_span_id: terminal.map(|span| span.id.clone()),
            terminal_status: terminal.map(|span| span.source_status),
            structured_facts: terminal.map_or_else(BTreeMap::new, |span| {
                task_completion_structured_facts(&span.attributes)
            }),
            input_summary: None,
            input_truncated: false,
            output_summary: None,
            output_truncated: false,
            evidence_keys: Vec::new(),
        };
        let mut pending_records = BTreeMap::new();
        let mut truncated = false;
        if let Some(terminal) = terminal {
            let key = evidence_key("terminal-span", &terminal.id);
            trace_observation.evidence_keys.push(key.clone());
            pending_records.insert(
                key,
                (
                    EvaluationEvidenceKindV1::Span,
                    EvaluationEvidenceLocationV1::Span {
                        span_id: terminal.id.clone(),
                    },
                ),
            );
            if self.content_policy == TaskCompletionContentPolicyV1::PreRedactedSummaries {
                if let Some(input) = terminal.input.as_deref() {
                    trace_observation.input_truncated =
                        input.len() > self.max_summary_bytes as usize;
                    trace_observation.input_summary =
                        Some(bound_utf8(input, self.max_summary_bytes));
                }
                if let Some(output) = terminal.output.as_deref() {
                    trace_observation.output_truncated =
                        output.len() > self.max_summary_bytes as usize;
                    let summary = bound_utf8(output, self.max_summary_bytes);
                    if !summary.is_empty() {
                        let key = evidence_key("terminal-output", &terminal.id);
                        pending_records.insert(
                            key.clone(),
                            (
                                EvaluationEvidenceKindV1::OutputSegment,
                                EvaluationEvidenceLocationV1::Segment {
                                    span_id: terminal.id.clone(),
                                    start_byte: 0,
                                    end_byte: u32::try_from(summary.len()).unwrap_or(u32::MAX),
                                },
                            ),
                        );
                        trace_observation.evidence_keys.push(key);
                        trace_observation.output_summary = Some(summary);
                    }
                }
            }
        }
        let all_tool_spans = spans
            .iter()
            .copied()
            .filter(|span| is_tool_span(span))
            .collect::<Vec<_>>();
        trace_observation.tool_span_count = u32::try_from(all_tool_spans.len()).unwrap_or(u32::MAX);
        truncated |= all_tool_spans.len() > self.max_tool_observations as usize;
        let mut tools = Vec::new();
        for (index, span) in all_tool_spans
            .into_iter()
            .take(self.max_tool_observations as usize)
            .enumerate()
        {
            let key = evidence_key("tool-span", &span.id);
            pending_records.insert(
                key.clone(),
                (
                    EvaluationEvidenceKindV1::Span,
                    EvaluationEvidenceLocationV1::Span {
                        span_id: span.id.clone(),
                    },
                ),
            );
            let (input_summary, input_evidence_key, input_truncated) = project_span_summary(
                self.content_policy,
                span.input.as_deref(),
                "tool-input",
                EvaluationEvidenceKindV1::InputSegment,
                &span.id,
                self.max_summary_bytes,
                &mut pending_records,
            );
            let (output_summary, output_evidence_key, output_truncated) = project_span_summary(
                self.content_policy,
                span.output.as_deref(),
                "tool-output",
                EvaluationEvidenceKindV1::OutputSegment,
                &span.id,
                self.max_summary_bytes,
                &mut pending_records,
            );
            tools.push(TaskCompletionToolObservationV1 {
                sequence: u32::try_from(index).unwrap_or(u32::MAX),
                span_id: span.id.clone(),
                parent_span_id: span.parent_id.clone(),
                tool_name: span.name.clone(),
                source_status: span.source_status,
                error_present: span.error.is_some(),
                structured_facts: task_completion_structured_facts(&span.attributes),
                input_summary,
                input_evidence_key,
                input_truncated,
                output_summary,
                output_evidence_key,
                output_truncated,
                started_at_unix_nano: span.start_time_unix_nano,
                evidence_key: key,
            });
        }

        let placeholder = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
        let mut projection = TaskCompletionProjectionV1 {
            schema_version: TASK_COMPLETION_PROJECTION_SCHEMA_VERSION.into(),
            projector_version: TASK_COMPLETION_PROJECTOR_VERSION.into(),
            content_policy: self.content_policy,
            max_tool_observations: self.max_tool_observations,
            max_summary_bytes: self.max_summary_bytes,
            target_key: target_key.into(),
            target_revision: target_revision.into(),
            trace_context_binding_id: binding_id,
            context_release_id,
            context_projection_release_id,
            projection_hash: placeholder.into(),
            declared_fields,
            criteria,
            capabilities,
            trace: trace_observation,
            tools,
            missing_required_context,
            truncated,
            evidence_catalog: EvaluationEvidenceCatalogV1 {
                target_key: target_key.into(),
                target_revision: target_revision.into(),
                projection_hash: placeholder.into(),
                entries: BTreeMap::new(),
            },
        };
        let projection_hash = projection.compute_hash()?;
        projection.projection_hash = projection_hash.clone();
        projection.evidence_catalog.projection_hash = projection_hash.clone();
        projection.evidence_catalog.entries = pending_records
            .into_iter()
            .map(|(key, (evidence_kind, location))| {
                (
                    key,
                    EvaluationEvidenceRecordV1 {
                        target_key: target_key.into(),
                        target_revision: target_revision.into(),
                        projection_hash: projection_hash.clone(),
                        evidence_kind,
                        location,
                        applicable_criterion_ids: BTreeSet::new(),
                    },
                )
            })
            .collect();
        projection.validate()?;
        Ok(projection)
    }
}

pub struct TaskCompletionEvaluator<C> {
    client: C,
    model: String,
    system_prompt: String,
    response_schema: ResponseSchema,
    evaluator_release: EvaluatorReleaseSpecV1,
}

#[cfg(feature = "llm-judge-openai")]
pub type OpenAiTaskCompletionEvaluator =
    TaskCompletionEvaluator<crate::providers::openai_dive::chat::OpenAiChatClient>;

#[cfg(feature = "llm-judge-openai")]
impl OpenAiTaskCompletionEvaluator {
    /// Creates the production OpenAI-backed evaluator. The API key is read by
    /// the provider client from `OPENAI_API_KEY`; it is never serialized into
    /// the evaluator release or provider response envelope.
    pub fn from_env(
        model: impl Into<String>,
        evaluator_release: EvaluatorReleaseSpecV1,
    ) -> Result<Self, ContractError> {
        let model = model.into();
        Self::new(
            crate::providers::openai_dive::chat::OpenAiChatClient::from_env(),
            model,
            evaluator_release,
        )
    }
}

impl<C> TaskCompletionEvaluator<C>
where
    C: ChatClient,
{
    pub fn new(
        client: C,
        model: impl Into<String>,
        evaluator_release: EvaluatorReleaseSpecV1,
    ) -> Result<Self, ContractError> {
        evaluator_release.validate()?;
        if evaluator_release.task_kind != LearnedTaskKind::TaskCompletion
            || evaluator_release.target_kind != EvaluationTargetKind::TraceRevision
        {
            return Err(task_error(
                "task-completion evaluator requires task_completion over trace_revision",
            ));
        }
        let model = model.into();
        require_non_empty(&model, "model", task_error)?;
        let (system_prompt, rubric, release_response_schema, decoding_parameters) =
            match &evaluator_release.implementation {
                EvaluationImplementationV1::PromptJudge {
                    requested_model,
                    system_prompt,
                    rubric,
                    response_schema,
                    decoding_parameters,
                    ..
                } => {
                    if requested_model != &model {
                        return Err(task_error(
                            "runtime model must match the immutable evaluator release",
                        ));
                    }
                    (system_prompt, rubric, response_schema, decoding_parameters)
                }
                _ => {
                    return Err(task_error(
                        "task-completion provider execution requires a prompt judge",
                    ));
                }
            };
        if !decoding_parameters.is_empty() {
            return Err(task_error(
                "task-completion decoding parameters are unsupported by this runtime",
            ));
        }
        let response_schema = task_completion_response_schema();
        if release_response_schema != &response_schema.schema {
            return Err(task_error(
                "evaluator release response schema does not match the task-completion judgment schema",
            ));
        }
        Ok(Self {
            client,
            model,
            system_prompt: format!(
                "{}\n\nTask-completion rubric:\n{}",
                system_prompt.trim(),
                rubric.trim()
            ),
            response_schema,
            evaluator_release,
        })
    }

    pub async fn evaluate(
        &self,
        projection: &TaskCompletionProjectionV1,
        binding: &TraceContextBindingV1,
    ) -> anyhow::Result<TaskCompletionExecutionV1> {
        projection.validate()?;
        validate_release_and_binding(projection, binding, &self.evaluator_release)?;
        if let Some(reason) = local_abstention_reason(projection, binding) {
            return Ok(TaskCompletionExecutionV1 {
                evaluation: local_abstention(projection, binding, &self.evaluator_release, reason)?,
                provider: None,
            });
        }

        let request = ChatRequest {
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            user_prompt: serde_json::to_string(projection)?,
            response_schema: self.response_schema.clone(),
            context_id: Some(projection.projection_hash.clone()),
        };
        let envelope = self
            .client
            .complete_json_enveloped::<TaskCompletionJudgmentV1>(request)
            .await?;
        let provider = envelope.provider_response.clone();
        let evaluation = match envelope.output.into_learned_evaluation(
            projection,
            binding,
            &self.evaluator_release,
        ) {
            Ok(evaluation) => evaluation,
            Err(error) => {
                let mut evaluation = local_abstention(
                    projection,
                    binding,
                    &self.evaluator_release,
                    LearnedAbstentionReasonV1::InvalidProviderOutput,
                )?;
                evaluation.explanation = format!(
                    "Task completion abstained because the provider output failed contract validation: {error}"
                );
                evaluation
            }
        };
        Ok(TaskCompletionExecutionV1 {
            evaluation,
            provider: Some(provider),
        })
    }
}

fn validate_release_and_binding(
    projection: &TaskCompletionProjectionV1,
    binding: &TraceContextBindingV1,
    evaluator_release: &EvaluatorReleaseSpecV1,
) -> Result<(), ContractError> {
    evaluator_release.validate()?;
    binding.validate()?;
    if evaluator_release.task_kind != LearnedTaskKind::TaskCompletion
        || evaluator_release.target_kind != EvaluationTargetKind::TraceRevision
    {
        return Err(task_error(
            "evaluator release has the wrong task or target kind",
        ));
    }
    if binding.target_key != projection.target_key
        || binding.target_revision != projection.target_revision
        || binding.binding_id()? != projection.trace_context_binding_id
    {
        return Err(task_error("binding does not match task projection"));
    }
    if evaluator_release.projection_release_id != projection.projector_release_id()? {
        return Err(task_error(
            "evaluator projection release does not match projector identity",
        ));
    }
    if projection
        .context_projection_release_id
        .as_ref()
        .is_some_and(|release_id| release_id != &evaluator_release.context_projection_release_id)
    {
        return Err(task_error(
            "evaluator context projection release does not match projection",
        ));
    }
    Ok(())
}

fn local_abstention_reason(
    projection: &TaskCompletionProjectionV1,
    binding: &TraceContextBindingV1,
) -> Option<LearnedAbstentionReasonV1> {
    match binding.resolution {
        TraceContextBindingResolutionV1::Unresolved
        | TraceContextBindingResolutionV1::Ambiguous => {
            return Some(LearnedAbstentionReasonV1::ContextUnresolved);
        }
        TraceContextBindingResolutionV1::Resolved => {}
    }
    if !projection.missing_required_context.is_empty() {
        return Some(LearnedAbstentionReasonV1::ContextInsufficient);
    }
    if projection.truncated {
        return Some(LearnedAbstentionReasonV1::ContentTruncated);
    }
    if projection.evidence_catalog.entries.is_empty() {
        return Some(LearnedAbstentionReasonV1::EvidenceInsufficient);
    }
    None
}

fn local_abstention(
    projection: &TaskCompletionProjectionV1,
    binding: &TraceContextBindingV1,
    evaluator_release: &EvaluatorReleaseSpecV1,
    reason: LearnedAbstentionReasonV1,
) -> Result<LearnedEvaluationV1, ContractError> {
    TaskCompletionJudgmentV1 {
        schema_version: TASK_COMPLETION_JUDGMENT_SCHEMA_VERSION.into(),
        outcome: TaskCompletionOutcomeV1::Abstain,
        completion_score: None,
        model_reported_confidence: None,
        explanation: format!("Task completion abstained: {reason:?}"),
        evidence_keys: Vec::new(),
        criteria: projection
            .criteria
            .iter()
            .map(|criterion| TaskCompletionCriterionJudgmentV1 {
                criterion_id: criterion.criterion_id.clone(),
                outcome: TaskCompletionCriterionOutcomeV1::Unresolved,
                score: None,
                evidence_keys: Vec::new(),
            })
            .collect(),
        abstention_reason: Some(reason),
    }
    .into_learned_evaluation(projection, binding, evaluator_release)
}

fn task_completion_response_schema() -> ResponseSchema {
    ResponseSchema {
        name: "task_completion_judgment_v1".into(),
        description: Some("Evidence-grounded task-completion judgment".into()),
        strict: true,
        schema: json!({
            "type": "object",
            "additionalProperties": false,
            "required": [
                "schema_version",
                "outcome",
                "completion_score",
                "model_reported_confidence",
                "explanation",
                "evidence_keys",
                "criteria",
                "abstention_reason"
            ],
            "properties": {
                "schema_version": {"type": "string", "const": TASK_COMPLETION_JUDGMENT_SCHEMA_VERSION},
                "outcome": {"type": "string", "enum": ["completed", "partial", "failed", "abstain"]},
                "completion_score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "model_reported_confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "explanation": {"type": "string"},
                "evidence_keys": {"type": "array", "items": {"type": "string"}},
                "criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["criterion_id", "outcome", "score", "evidence_keys"],
                        "properties": {
                            "criterion_id": {"type": "string"},
                            "outcome": {"type": "string", "enum": ["satisfied", "partially_satisfied", "unsatisfied", "unresolved"]},
                            "score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                            "evidence_keys": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "abstention_reason": {
                    "type": ["string", "null"],
                    "enum": ["context_unresolved", "context_insufficient", "content_unavailable", "content_truncated", "privacy_blocked", "evidence_insufficient", "out_of_distribution", "provider_unavailable", "invalid_provider_output", "not_applicable", null]
                }
            }
        }),
    }
}

pub fn task_completion_judgment_response_schema() -> Value {
    task_completion_response_schema().schema
}

fn project_field(
    output: &mut Vec<TaskCompletionDeclaredFieldV1>,
    field: &ContextFieldV1,
    projection: &ContextProjectionV1,
    include_content: bool,
) {
    if projection
        .included_field_ids
        .contains(&field.metadata.field_id)
    {
        output.push(TaskCompletionDeclaredFieldV1 {
            field_id: field.metadata.field_id.clone(),
            value: include_content.then(|| field.value.clone()),
        });
    }
}

fn project_capability(capability: &AgentCapabilityV1) -> TaskCompletionCapabilityV1 {
    TaskCompletionCapabilityV1 {
        capability_id: capability.capability_id.clone(),
        field_id: capability.metadata.field_id.clone(),
        name: capability.name.clone(),
        allowed_operations: capability.allowed_operations.clone(),
        prohibited_operations: capability.prohibited_operations.clone(),
        requires_approval: capability.requires_approval,
    }
}

fn is_tool_span(span: &Span) -> bool {
    span.kind == SpanKind::Tool
        || span
            .attributes
            .get("openinference.span.kind")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind.eq_ignore_ascii_case("tool"))
}

fn bound_utf8(value: &str, max_bytes: u32) -> String {
    if value.len() <= max_bytes as usize {
        return value.to_owned();
    }
    let mut end = max_bytes as usize;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

fn evidence_key(prefix: &str, identity: &str) -> String {
    format!("{prefix}:{identity}")
}

fn validate_evidence_keys(
    keys: &[String],
    projection: &TaskCompletionProjectionV1,
) -> Result<(), ContractError> {
    let mut seen = BTreeSet::new();
    for key in keys {
        if !seen.insert(key.as_str()) {
            return Err(task_error(format!("duplicate evidence key {key}")));
        }
        if !projection.evidence_catalog.entries.contains_key(key) {
            return Err(task_error(format!("unknown evidence key {key}")));
        }
    }
    Ok(())
}

fn deduplicate_evidence_keys(keys: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    keys.retain(|key| seen.insert(key.clone()));
}

fn citation_for(
    key: &str,
    criterion_id: Option<String>,
    projection: &TaskCompletionProjectionV1,
) -> Result<EvaluationEvidenceCitationV1, ContractError> {
    let record = projection
        .evidence_catalog
        .entries
        .get(key)
        .ok_or_else(|| task_error(format!("unknown evidence key {key}")))?;
    Ok(EvaluationEvidenceCitationV1 {
        evidence_key: key.into(),
        evidence_kind: record.evidence_kind,
        location: record.location.clone(),
        criterion_id,
    })
}

fn criterion_label(outcome: TaskCompletionCriterionOutcomeV1) -> &'static str {
    match outcome {
        TaskCompletionCriterionOutcomeV1::Satisfied => "satisfied",
        TaskCompletionCriterionOutcomeV1::PartiallySatisfied => "partially_satisfied",
        TaskCompletionCriterionOutcomeV1::Unsatisfied => "unsatisfied",
        TaskCompletionCriterionOutcomeV1::Unresolved => "unresolved",
    }
}

fn validate_probability(value: Option<f64>, field: &str) -> Result<(), ContractError> {
    if let Some(value) = value
        && (!value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(task_error(format!(
            "{field} must be finite and between 0 and 1"
        )));
    }
    Ok(())
}

fn task_error(message: impl Into<String>) -> ContractError {
    ContractError::InvalidTaskCompletion(message.into())
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use serde::de::DeserializeOwned;

    use super::*;

    fn assert_strict_object_requirements(schema: &Value) {
        if schema.get("type").and_then(Value::as_str) == Some("object") {
            let properties = schema
                .get("properties")
                .and_then(Value::as_object)
                .expect("object schemas must declare properties");
            let property_names = properties.keys().cloned().collect::<BTreeSet<_>>();
            let required_names = schema
                .get("required")
                .and_then(Value::as_array)
                .expect("strict object schemas must declare required")
                .iter()
                .map(|name| {
                    name.as_str()
                        .expect("required names must be strings")
                        .to_string()
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(
                required_names, property_names,
                "strict provider schemas must require every property; nullable fields remain nullable"
            );
        }
        match schema {
            Value::Object(object) => {
                for value in object.values() {
                    assert_strict_object_requirements(value);
                }
            }
            Value::Array(values) => {
                for value in values {
                    assert_strict_object_requirements(value);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn task_completion_response_schema_meets_strict_provider_requirements() {
        assert_strict_object_requirements(&task_completion_judgment_response_schema());
    }
    use crate::learned::{
        AGENT_CONTEXT_RELEASE_SCHEMA_VERSION, AgentArchitectureContextV1, AgentEvaluationContextV1,
        AgentIdentityContextV1, AgentIntentContextV1, AgentPolicyContextV1, ContextFieldMetadataV1,
        ContextFieldProvenanceV1, ContextReviewStateV1, ContextSensitivityV1,
        EVALUATOR_RELEASE_SCHEMA_VERSION, EvaluationInputBoundsV1,
        TRACE_CONTEXT_BINDING_SCHEMA_VERSION, TraceContextBindingProvenanceV1,
    };

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn field(id: &str, value: Value) -> ContextFieldV1 {
        ContextFieldV1 {
            metadata: ContextFieldMetadataV1 {
                field_id: id.into(),
                provenance: ContextFieldProvenanceV1::UserDeclared,
                source_snapshot_id: digest('a'),
                source_locator: None,
                captured_at: "2026-01-01T00:00:00Z".into(),
                fresh_until: None,
                review_state: ContextReviewStateV1::Approved,
                sensitivity: ContextSensitivityV1::HostedPreRedacted,
                inference_confidence: None,
            },
            value,
        }
    }

    fn context() -> AgentContextReleaseV1 {
        let criterion_field = field("criterion-field", json!("reply successfully"));
        AgentContextReleaseV1 {
            schema_version: AGENT_CONTEXT_RELEASE_SCHEMA_VERSION.into(),
            agent_id: "agent".into(),
            identity: AgentIdentityContextV1 {
                application_name: field("app", json!("agent")),
                owner: field("owner", json!("team")),
                environment: field("env", json!("test")),
                build_version_selectors: Vec::new(),
                entry_points: Vec::new(),
                user_personas: Vec::new(),
                supported_domains: Vec::new(),
                languages: Vec::new(),
                risk_tier: field("risk", json!("low")),
            },
            intent: AgentIntentContextV1 {
                purpose: field("purpose", json!("complete the requested task")),
                supported_tasks: Vec::new(),
                explicit_non_goals: Vec::new(),
                success_criteria: vec![super::super::SuccessCriterionV1 {
                    metadata: criterion_field.metadata,
                    criterion_id: "criterion-1".into(),
                    description: "reply successfully".into(),
                    importance: SuccessCriterionImportanceV1::Must,
                    required_evidence_kinds: BTreeSet::new(),
                    business_impact_weight: None,
                }],
                acceptable_partial_completion: None,
                refusal_requirements: Vec::new(),
                escalation_requirements: Vec::new(),
            },
            capabilities: Vec::new(),
            architecture: AgentArchitectureContextV1::default(),
            policy: AgentPolicyContextV1::default(),
            evaluation_context: AgentEvaluationContextV1::default(),
        }
    }

    fn binding(context: &AgentContextReleaseV1) -> TraceContextBindingV1 {
        TraceContextBindingV1 {
            schema_version: TRACE_CONTEXT_BINDING_SCHEMA_VERSION.into(),
            target_key: "trace-1".into(),
            target_revision: "rev-1".into(),
            resolution: TraceContextBindingResolutionV1::Resolved,
            agent_context_release_id: Some(context.release_id().unwrap()),
            binding_rule_release_id: digest('b'),
            binding_provenance: TraceContextBindingProvenanceV1::ExplicitInstrumentation,
            candidate_context_release_ids: BTreeSet::new(),
        }
    }

    fn context_projection(context: &AgentContextReleaseV1) -> ContextProjectionV1 {
        ContextProjectionV1 {
            context_release_id: context.release_id().unwrap(),
            projection_class: ContextProjectionClassV1::HostedPreRedacted,
            projector_version: "context-projector-v1".into(),
            redaction_version: "redaction-v1".into(),
            included_field_ids: ["purpose".into(), "criterion-field".into()]
                .into_iter()
                .collect(),
        }
    }

    fn trace() -> Trace {
        let mut root = Span::new("root", "agent").with_kind(SpanKind::Agent);
        root.input = Some("update the requested file and verify the result".into());
        root.output = Some("done".into());
        root.source_status = SourceSpanStatus::Ok;
        root.end_time_unix_nano = Some(3);
        root.attributes
            .insert("agent.final.status".into(), json!("completed"));
        let mut tool = Span::new("tool-1", "write_file").with_kind(SpanKind::Tool);
        tool.parent_id = Some("root".into());
        tool.source_status = SourceSpanStatus::Ok;
        tool.start_time_unix_nano = Some(2);
        tool.input = Some("src/web.js".into());
        tool.output = Some("updated 3 lines".into());
        tool.attributes
            .insert("agent.state.observation".into(), json!("verified_changed"));
        Trace::new("trace-1").with_span(root).with_span(tool)
    }

    fn release(projection: &TaskCompletionProjectionV1) -> EvaluatorReleaseSpecV1 {
        EvaluatorReleaseSpecV1 {
            schema_version: EVALUATOR_RELEASE_SCHEMA_VERSION.into(),
            name: "task completion".into(),
            task_kind: LearnedTaskKind::TaskCompletion,
            target_kind: EvaluationTargetKind::TraceRevision,
            implementation: EvaluationImplementationV1::PromptJudge {
                provider: "test".into(),
                requested_model: "test-model".into(),
                system_prompt: "judge".into(),
                rubric: "task completion".into(),
                response_schema: task_completion_judgment_response_schema(),
                decoding_parameters: BTreeMap::new(),
                parser_version: "v1".into(),
                normalizer_version: "v1".into(),
            },
            projection_release_id: projection.projector_release_id().unwrap(),
            context_projection_release_id: projection
                .context_projection_release_id
                .clone()
                .unwrap(),
            applicable_taxonomy_release_id: None,
            applicable_taxonomy_node_ids: BTreeSet::new(),
            input_bounds: EvaluationInputBoundsV1 {
                max_subjects: 1,
                max_evidence_items: 100,
                max_input_bytes: 100_000,
                max_output_bytes: 10_000,
            },
            evidence_schema_version: "v1".into(),
            abstention_policy: json!({}),
            code_artifact_hash: digest('c'),
        }
    }

    #[derive(Clone)]
    struct FakeChatClient {
        calls: Arc<AtomicUsize>,
        output: Value,
    }

    #[async_trait::async_trait]
    impl ChatClient for FakeChatClient {
        async fn complete_json<T>(&self, _request: ChatRequest) -> anyhow::Result<T>
        where
            T: DeserializeOwned + Send,
        {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(serde_json::from_value(self.output.clone())?)
        }
    }

    #[derive(Clone)]
    struct RecordingChatClient {
        request: Arc<Mutex<Option<ChatRequest>>>,
        output: Value,
    }

    #[async_trait::async_trait]
    impl ChatClient for RecordingChatClient {
        async fn complete_json<T>(&self, request: ChatRequest) -> anyhow::Result<T>
        where
            T: DeserializeOwned + Send,
        {
            *self.request.lock().unwrap() = Some(request);
            Ok(serde_json::from_value(self.output.clone())?)
        }
    }

    #[test]
    fn native_child_tool_spans_are_preserved_and_hash_is_stable() {
        let context = context();
        let binding = binding(&context);
        let context_projection = context_projection(&context);
        let projector = TaskCompletionProjectorV1 {
            content_policy: TaskCompletionContentPolicyV1::PreRedactedSummaries,
            ..Default::default()
        };
        let first = projector
            .project(
                "trace-1",
                "rev-1",
                &binding,
                Some(&context),
                Some(&context_projection),
                &trace(),
            )
            .unwrap();
        let mut reordered = trace();
        reordered.spans.reverse();
        let second = projector
            .project(
                "trace-1",
                "rev-1",
                &binding,
                Some(&context),
                Some(&context_projection),
                &reordered,
            )
            .unwrap();

        assert_eq!(first.projection_hash, second.projection_hash);
        assert_eq!(first.tools.len(), 1);
        assert_eq!(first.tools[0].parent_span_id.as_deref(), Some("root"));
        assert_eq!(
            first.trace.structured_facts.get("agent.final.status"),
            Some(&json!("completed"))
        );
        assert_eq!(first.tools[0].input_summary.as_deref(), Some("src/web.js"));
        assert_eq!(
            first.tools[0].output_summary.as_deref(),
            Some("updated 3 lines")
        );
        assert_eq!(
            first.tools[0]
                .structured_facts
                .get("agent.state.observation"),
            Some(&json!("verified_changed"))
        );
        assert!(
            first
                .evidence_catalog
                .entries
                .contains_key("tool-span:tool-1")
        );
        assert!(
            first
                .evidence_catalog
                .entries
                .contains_key("tool-input:tool-1")
        );
        assert!(
            first
                .evidence_catalog
                .entries
                .contains_key("tool-output:tool-1")
        );
    }

    #[test]
    fn bounded_content_truncation_is_visible_without_discarding_structural_evidence() {
        let context = context();
        let binding = binding(&context);
        let context_projection = context_projection(&context);
        let projector = TaskCompletionProjectorV1 {
            content_policy: TaskCompletionContentPolicyV1::PreRedactedSummaries,
            max_summary_bytes: 32,
            ..Default::default()
        };
        let mut long_trace = trace();
        long_trace
            .spans
            .iter_mut()
            .find(|span| span.id == "root")
            .unwrap()
            .output = Some("x".repeat(128));
        let projection = projector
            .project(
                "trace-1",
                "rev-1",
                &binding,
                Some(&context),
                Some(&context_projection),
                &long_trace,
            )
            .unwrap();

        assert!(projection.trace.output_truncated);
        assert_eq!(
            projection.trace.output_summary.as_deref(),
            Some("x".repeat(32).as_str())
        );
        assert!(!projection.truncated);
        assert_eq!(local_abstention_reason(&projection, &binding), None);
    }

    #[test]
    fn structured_only_context_abstains_without_guessing() {
        let context = context();
        let binding = binding(&context);
        let context_projection = ContextProjectionV1 {
            projection_class: ContextProjectionClassV1::StructuralOnly,
            ..context_projection(&context)
        };
        let projection = TaskCompletionProjectorV1::default()
            .project(
                "trace-1",
                "rev-1",
                &binding,
                Some(&context),
                Some(&context_projection),
                &trace(),
            )
            .unwrap();
        let evaluation = local_abstention(
            &projection,
            &binding,
            &release(&projection),
            local_abstention_reason(&projection, &binding).unwrap(),
        )
        .unwrap();

        assert_eq!(evaluation.verdict, LearnedVerdictV1::Abstain);
        assert_eq!(
            evaluation.abstention_reason,
            Some(LearnedAbstentionReasonV1::ContextInsufficient)
        );
    }

    #[test]
    fn insufficient_context_never_calls_the_provider() {
        let context = context();
        let binding = binding(&context);
        let context_projection = ContextProjectionV1 {
            projection_class: ContextProjectionClassV1::StructuralOnly,
            ..context_projection(&context)
        };
        let projection = TaskCompletionProjectorV1::default()
            .project(
                "trace-1",
                "rev-1",
                &binding,
                Some(&context),
                Some(&context_projection),
                &trace(),
            )
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let evaluator = TaskCompletionEvaluator::new(
            FakeChatClient {
                calls: calls.clone(),
                output: json!({}),
            },
            "test-model",
            release(&projection),
        )
        .unwrap();
        let execution =
            futures::executor::block_on(evaluator.evaluate(&projection, &binding)).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(execution.provider.is_none());
        assert_eq!(execution.evaluation.verdict, LearnedVerdictV1::Abstain);
    }

    #[test]
    fn fabricated_provider_evidence_becomes_typed_abstention() {
        let context = context();
        let binding = binding(&context);
        let context_projection = context_projection(&context);
        let projector = TaskCompletionProjectorV1 {
            content_policy: TaskCompletionContentPolicyV1::PreRedactedSummaries,
            ..Default::default()
        };
        let projection = projector
            .project(
                "trace-1",
                "rev-1",
                &binding,
                Some(&context),
                Some(&context_projection),
                &trace(),
            )
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let evaluator = TaskCompletionEvaluator::new(
            FakeChatClient {
                calls: calls.clone(),
                output: serde_json::to_value(TaskCompletionJudgmentV1 {
                    schema_version: TASK_COMPLETION_JUDGMENT_SCHEMA_VERSION.into(),
                    outcome: TaskCompletionOutcomeV1::Completed,
                    completion_score: Some(1.0),
                    model_reported_confidence: Some(0.9),
                    explanation: "completed".into(),
                    evidence_keys: vec!["fabricated".into()],
                    criteria: vec![TaskCompletionCriterionJudgmentV1 {
                        criterion_id: "criterion-1".into(),
                        outcome: TaskCompletionCriterionOutcomeV1::Satisfied,
                        score: Some(1.0),
                        evidence_keys: vec!["fabricated".into()],
                    }],
                    abstention_reason: None,
                })
                .unwrap(),
            },
            "test-model",
            release(&projection),
        )
        .unwrap();
        let execution =
            futures::executor::block_on(evaluator.evaluate(&projection, &binding)).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(execution.provider.is_some());
        assert_eq!(
            execution.evaluation.abstention_reason,
            Some(LearnedAbstentionReasonV1::InvalidProviderOutput)
        );
        assert!(
            execution
                .evaluation
                .explanation
                .contains("unknown evidence key")
        );
    }

    #[test]
    fn repeated_valid_provider_citations_are_normalized() {
        let context = context();
        let binding = binding(&context);
        let context_projection = context_projection(&context);
        let projection = TaskCompletionProjectorV1 {
            content_policy: TaskCompletionContentPolicyV1::PreRedactedSummaries,
            ..Default::default()
        }
        .project(
            "trace-1",
            "rev-1",
            &binding,
            Some(&context),
            Some(&context_projection),
            &trace(),
        )
        .unwrap();
        let evidence_key = "tool-span:tool-1".to_string();
        let calls = Arc::new(AtomicUsize::new(0));
        let evaluator = TaskCompletionEvaluator::new(
            FakeChatClient {
                calls: calls.clone(),
                output: serde_json::to_value(TaskCompletionJudgmentV1 {
                    schema_version: TASK_COMPLETION_JUDGMENT_SCHEMA_VERSION.into(),
                    outcome: TaskCompletionOutcomeV1::Completed,
                    completion_score: Some(1.0),
                    model_reported_confidence: Some(0.9),
                    explanation: "completed".into(),
                    evidence_keys: vec![evidence_key.clone(), evidence_key.clone()],
                    criteria: vec![TaskCompletionCriterionJudgmentV1 {
                        criterion_id: "criterion-1".into(),
                        outcome: TaskCompletionCriterionOutcomeV1::Satisfied,
                        score: Some(1.0),
                        evidence_keys: vec![evidence_key.clone(), evidence_key],
                    }],
                    abstention_reason: None,
                })
                .unwrap(),
            },
            "test-model",
            release(&projection),
        )
        .unwrap();
        let execution =
            futures::executor::block_on(evaluator.evaluate(&projection, &binding)).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(execution.evaluation.verdict, LearnedVerdictV1::Pass);
        assert_eq!(execution.evaluation.evidence.len(), 2);
    }

    #[test]
    fn provider_request_uses_the_exact_release_prompt_rubric_and_schema() {
        let context = context();
        let binding = binding(&context);
        let context_projection = context_projection(&context);
        let projection = TaskCompletionProjectorV1 {
            content_policy: TaskCompletionContentPolicyV1::PreRedactedSummaries,
            ..Default::default()
        }
        .project(
            "trace-1",
            "rev-1",
            &binding,
            Some(&context),
            Some(&context_projection),
            &trace(),
        )
        .unwrap();
        let mut evaluator_release = release(&projection);
        if let EvaluationImplementationV1::PromptJudge {
            system_prompt,
            rubric,
            ..
        } = &mut evaluator_release.implementation
        {
            *system_prompt = "SYSTEM-PROMPT-IDENTITY".into();
            *rubric = "RUBRIC-IDENTITY".into();
        }
        let request = Arc::new(Mutex::new(None));
        let evaluator = TaskCompletionEvaluator::new(
            RecordingChatClient {
                request: request.clone(),
                output: serde_json::to_value(TaskCompletionJudgmentV1 {
                    schema_version: TASK_COMPLETION_JUDGMENT_SCHEMA_VERSION.into(),
                    outcome: TaskCompletionOutcomeV1::Completed,
                    completion_score: Some(1.0),
                    model_reported_confidence: Some(0.9),
                    explanation: "completed from observed terminal evidence".into(),
                    evidence_keys: vec!["terminal-span:root".into()],
                    criteria: vec![TaskCompletionCriterionJudgmentV1 {
                        criterion_id: "criterion-1".into(),
                        outcome: TaskCompletionCriterionOutcomeV1::Satisfied,
                        score: Some(1.0),
                        evidence_keys: vec!["terminal-span:root".into()],
                    }],
                    abstention_reason: None,
                })
                .unwrap(),
            },
            "test-model",
            evaluator_release,
        )
        .unwrap();
        let execution =
            futures::executor::block_on(evaluator.evaluate(&projection, &binding)).unwrap();
        assert_eq!(execution.evaluation.verdict, LearnedVerdictV1::Pass);
        let request = request.lock().unwrap().clone().unwrap();
        assert_eq!(
            request.system_prompt,
            "SYSTEM-PROMPT-IDENTITY\n\nTask-completion rubric:\nRUBRIC-IDENTITY"
        );
        assert_eq!(
            request.response_schema.schema,
            task_completion_judgment_response_schema()
        );
    }

    #[test]
    fn unsupported_release_decoding_policy_fails_before_provider_execution() {
        let context = context();
        let binding = binding(&context);
        let context_projection = context_projection(&context);
        let projection = TaskCompletionProjectorV1 {
            content_policy: TaskCompletionContentPolicyV1::PreRedactedSummaries,
            ..Default::default()
        }
        .project(
            "trace-1",
            "rev-1",
            &binding,
            Some(&context),
            Some(&context_projection),
            &trace(),
        )
        .unwrap();
        let mut evaluator_release = release(&projection);
        if let EvaluationImplementationV1::PromptJudge {
            decoding_parameters,
            ..
        } = &mut evaluator_release.implementation
        {
            decoding_parameters.insert("temperature".into(), json!(0));
        }
        let calls = Arc::new(AtomicUsize::new(0));
        assert!(
            TaskCompletionEvaluator::new(
                FakeChatClient {
                    calls: calls.clone(),
                    output: json!({}),
                },
                "test-model",
                evaluator_release,
            )
            .is_err()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn declared_context_is_not_achievement_evidence() {
        let context = context();
        let binding = binding(&context);
        let context_projection = context_projection(&context);
        let projector = TaskCompletionProjectorV1 {
            content_policy: TaskCompletionContentPolicyV1::PreRedactedSummaries,
            ..Default::default()
        };
        let projection = projector
            .project(
                "trace-1",
                "rev-1",
                &binding,
                Some(&context),
                Some(&context_projection),
                &Trace::new("trace-1"),
            )
            .unwrap();
        let judgment = TaskCompletionJudgmentV1 {
            schema_version: TASK_COMPLETION_JUDGMENT_SCHEMA_VERSION.into(),
            outcome: TaskCompletionOutcomeV1::Completed,
            completion_score: Some(1.0),
            model_reported_confidence: Some(1.0),
            explanation: "declared goal says success".into(),
            evidence_keys: Vec::new(),
            criteria: vec![TaskCompletionCriterionJudgmentV1 {
                criterion_id: "criterion-1".into(),
                outcome: TaskCompletionCriterionOutcomeV1::Satisfied,
                score: Some(1.0),
                evidence_keys: Vec::new(),
            }],
            abstention_reason: None,
        };
        assert!(judgment.validate_against(&projection).is_err());
    }
}
