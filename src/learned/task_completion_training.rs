use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    CompactTaskCompletionProjectionV1, ContractError, TaskCompletionGoalBundleV1,
    TaskCompletionRecoveryChainV1, TaskCompletionTraceFactV1, TraceFactKindV1, TraceFactStatusV1,
    canonical_content_id, require_non_empty, require_sha256,
};

pub const TASK_COMPLETION_EVIDENCE_FEATURE_RECORD_SCHEMA_VERSION: &str =
    "traceeval.task_completion_evidence_feature_record.v1";
pub const TASK_COMPLETION_STRUCTURED_FEATURE_SET_VERSION: &str =
    "traceeval.task_completion_structured_evidence.v1";
pub const TASK_COMPLETION_TRAINING_RECORD_SCHEMA_VERSION: &str =
    "traceeval.task_completion_training_record.v1";

pub const FEATURE_NAMES: [&str; 39] = [
    "included_fact_count_log1p",
    "omitted_fact_count_log1p",
    "evidence_coverage",
    "projected_token_ratio",
    "compression_ratio",
    "final_response_present",
    "recovery_chain_count_log1p",
    "recovered_failure_fraction",
    "failed_fact_fraction",
    "succeeded_fact_fraction",
    "unfinished_fact_fraction",
    "verification_succeeded_count_log1p",
    "verification_failed_count_log1p",
    "verification_missing",
    "mutation_succeeded_count_log1p",
    "mutation_failed_count_log1p",
    "external_succeeded_count_log1p",
    "external_failed_count_log1p",
    "tool_succeeded_count_log1p",
    "tool_failed_count_log1p",
    "child_succeeded_count_log1p",
    "child_failed_count_log1p",
    "unfinished_fact_count_log1p",
    "failed_fact_count_log1p",
    "succeeded_fact_count_log1p",
    "last_fact_failed",
    "last_fact_succeeded",
    "failure_recency",
    "successes_after_last_failure_log1p",
    "failures_after_last_success_log1p",
    "distinct_tool_count_log1p",
    "mandatory_fact_fraction",
    "goal_relevant_fact_fraction",
    "final_response_token_ratio",
    "recovery_token_ratio",
    "user_amendment_count_log1p",
    "requested_verification_count_log1p",
    "requested_side_effect_count_log1p",
    "constraint_count_log1p",
];

/// Label-free, revision-bound measurements of projected trace evidence.
///
/// Source, model, benchmark reward, environment success, and gold-label fields
/// are intentionally absent. These values describe evidence; the calibrated
/// model remains responsible for the completion decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionEvidenceFeatureRecordV1 {
    pub schema_version: String,
    pub feature_set_version: String,
    pub feature_record_id: String,
    pub target_key: String,
    pub target_revision: String,
    pub trace_context_binding_id: String,
    pub projection_hash: String,
    pub projector_version: String,
    pub feature_names: Vec<String>,
    pub feature_values: Vec<f64>,
}

impl TaskCompletionEvidenceFeatureRecordV1 {
    pub fn from_projection(
        projection: &CompactTaskCompletionProjectionV1,
    ) -> Result<Self, ContractError> {
        projection.validate()?;
        if projection.stats.mandatory_facts_omitted != 0 {
            return Err(training_error(format!(
                "mandatory evidence was omitted for {}",
                projection.target_key
            )));
        }

        let values = extract_values(projection);
        if values.len() != FEATURE_NAMES.len() || !values.iter().all(|value| value.is_finite()) {
            return Err(training_error(format!(
                "invalid structured feature vector for {}",
                projection.target_key
            )));
        }
        let names = FEATURE_NAMES.map(String::from).to_vec();
        let feature_record_id = canonical_content_id(
            TASK_COMPLETION_EVIDENCE_FEATURE_RECORD_SCHEMA_VERSION,
            &json!({
                "feature_set_version": TASK_COMPLETION_STRUCTURED_FEATURE_SET_VERSION,
                "target_key": projection.target_key,
                "target_revision": projection.target_revision,
                "trace_context_binding_id": projection.trace_context_binding_id,
                "projection_hash": projection.projection_hash,
                "projector_version": projection.projector_version,
                "feature_names": names,
                "feature_values": values,
            }),
        )?;
        let record = Self {
            schema_version: TASK_COMPLETION_EVIDENCE_FEATURE_RECORD_SCHEMA_VERSION.into(),
            feature_set_version: TASK_COMPLETION_STRUCTURED_FEATURE_SET_VERSION.into(),
            feature_record_id,
            target_key: projection.target_key.clone(),
            target_revision: projection.target_revision.clone(),
            trace_context_binding_id: projection.trace_context_binding_id.clone(),
            projection_hash: projection.projection_hash.clone(),
            projector_version: projection.projector_version.clone(),
            feature_names: names,
            feature_values: values,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        validate_feature_record(self)
    }
}

/// Label-free, privacy-bounded training input derived from a sealed projection.
///
/// Source identity, benchmark rewards, human labels, provider judgments, and
/// split assignments are intentionally absent. Private training code joins
/// labels only after validating this record and its immutable projection hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCompletionTrainingRecordV1 {
    pub schema_version: String,
    pub training_record_id: String,
    pub target_key: String,
    pub target_revision: String,
    pub trace_context_binding_id: String,
    pub projection_hash: String,
    pub projector_version: String,
    pub goal: TaskCompletionGoalBundleV1,
    pub facts: Vec<TaskCompletionTraceFactV1>,
    pub recovery_chains: Vec<TaskCompletionRecoveryChainV1>,
    pub structured_features: TaskCompletionEvidenceFeatureRecordV1,
}

impl TaskCompletionTrainingRecordV1 {
    pub fn from_projection(
        projection: &CompactTaskCompletionProjectionV1,
    ) -> Result<Self, ContractError> {
        let structured_features =
            TaskCompletionEvidenceFeatureRecordV1::from_projection(projection)?;
        let training_record_id = training_record_id(projection, &structured_features)?;
        let record = Self {
            schema_version: TASK_COMPLETION_TRAINING_RECORD_SCHEMA_VERSION.into(),
            training_record_id,
            target_key: projection.target_key.clone(),
            target_revision: projection.target_revision.clone(),
            trace_context_binding_id: projection.trace_context_binding_id.clone(),
            projection_hash: projection.projection_hash.clone(),
            projector_version: projection.projector_version.clone(),
            goal: projection.goal.clone(),
            facts: projection.facts.clone(),
            recovery_chains: projection.recovery_chains.clone(),
            structured_features,
        };
        record.validate_against(projection)?;
        Ok(record)
    }

    pub fn validate_against(
        &self,
        projection: &CompactTaskCompletionProjectionV1,
    ) -> Result<(), ContractError> {
        projection.validate()?;
        let expected_structured_features =
            TaskCompletionEvidenceFeatureRecordV1::from_projection(projection)?;
        self.structured_features.validate()?;
        if self.schema_version != TASK_COMPLETION_TRAINING_RECORD_SCHEMA_VERSION {
            return Err(training_error(
                "unsupported task-completion training record schema",
            ));
        }
        if self.target_key != projection.target_key
            || self.target_revision != projection.target_revision
            || self.trace_context_binding_id != projection.trace_context_binding_id
            || self.projection_hash != projection.projection_hash
            || self.projector_version != projection.projector_version
            || self.goal != projection.goal
            || self.facts != projection.facts
            || self.recovery_chains != projection.recovery_chains
        {
            return Err(training_error(
                "training record does not match its sealed projection",
            ));
        }
        if self.structured_features.target_key != self.target_key
            || self.structured_features.target_revision != self.target_revision
            || self.structured_features.trace_context_binding_id != self.trace_context_binding_id
            || self.structured_features.projection_hash != self.projection_hash
            || self.structured_features.projector_version != self.projector_version
        {
            return Err(training_error(
                "structured feature record does not match its training record",
            ));
        }
        if self.structured_features != expected_structured_features {
            return Err(training_error(
                "structured feature record does not match its sealed projection",
            ));
        }
        if self.training_record_id != training_record_id(projection, &self.structured_features)? {
            return Err(training_error(
                "training_record_id does not match training record content",
            ));
        }
        Ok(())
    }
}

fn training_record_id(
    projection: &CompactTaskCompletionProjectionV1,
    structured_features: &TaskCompletionEvidenceFeatureRecordV1,
) -> Result<String, ContractError> {
    canonical_content_id(
        TASK_COMPLETION_TRAINING_RECORD_SCHEMA_VERSION,
        &json!({
            "target_key": projection.target_key,
            "target_revision": projection.target_revision,
            "trace_context_binding_id": projection.trace_context_binding_id,
            "projection_hash": projection.projection_hash,
            "projector_version": projection.projector_version,
            "goal": projection.goal,
            "facts": projection.facts,
            "recovery_chains": projection.recovery_chains,
            "structured_feature_record_id": structured_features.feature_record_id,
        }),
    )
}

fn extract_values(projection: &CompactTaskCompletionProjectionV1) -> Vec<f64> {
    let facts = &projection.facts;
    let fact_count = facts.len() as f64;
    let failed = status_count(facts, TraceFactStatusV1::Failed);
    let succeeded = status_count(facts, TraceFactStatusV1::Succeeded);
    let unfinished = facts
        .iter()
        .filter(|fact| {
            matches!(
                fact.status,
                TraceFactStatusV1::Unknown
                    | TraceFactStatusV1::Running
                    | TraceFactStatusV1::Cancelled
            )
        })
        .count() as f64;
    let last = facts.iter().max_by_key(|fact| fact.sequence);
    let last_failure_sequence = facts
        .iter()
        .filter(|fact| fact.status == TraceFactStatusV1::Failed)
        .map(|fact| fact.sequence)
        .max();
    let last_success_sequence = facts
        .iter()
        .filter(|fact| fact.status == TraceFactStatusV1::Succeeded)
        .map(|fact| fact.sequence)
        .max();
    let max_sequence = facts.iter().map(|fact| fact.sequence).max().unwrap_or(0);
    let successes_after_last_failure = last_failure_sequence.map_or(0, |sequence| {
        facts
            .iter()
            .filter(|fact| fact.sequence > sequence && fact.status == TraceFactStatusV1::Succeeded)
            .count()
    });
    let failures_after_last_success = last_success_sequence.map_or(0, |sequence| {
        facts
            .iter()
            .filter(|fact| fact.sequence > sequence && fact.status == TraceFactStatusV1::Failed)
            .count()
    });
    let distinct_tools = facts
        .iter()
        .filter_map(|fact| fact.tool_name.as_deref())
        .collect::<BTreeSet<_>>()
        .len();
    let mandatory = facts.iter().filter(|fact| fact.mandatory).count() as f64;
    let goal_relevant = facts.iter().filter(|fact| !fact.mandatory).count() as f64;
    let included = projection.stats.included_facts as f64;
    let omitted = projection.stats.omitted_facts as f64;
    let projected_tokens = projection.token_budget.projected_tokens as f64;

    vec![
        log_count(included as usize),
        log_count(omitted as usize),
        ratio(included, included + omitted),
        ratio(
            projected_tokens,
            projection.token_budget.max_input_tokens as f64,
        ),
        ratio(
            projected_tokens,
            projection.token_budget.original_tokens as f64,
        ),
        binary(facts.iter().any(|fact| {
            fact.kind == TraceFactKindV1::AssistantMessage
                && matches!(
                    fact.lane,
                    super::TaskCompletionEvidenceLaneV1::FinalResponse
                )
        })),
        log_count(projection.recovery_chains.len()),
        ratio(projection.recovery_chains.len() as f64, failed),
        ratio(failed, fact_count),
        ratio(succeeded, fact_count),
        ratio(unfinished, fact_count),
        log_kind_status(
            facts,
            TraceFactKindV1::Verification,
            TraceFactStatusV1::Succeeded,
        ),
        log_kind_status(
            facts,
            TraceFactKindV1::Verification,
            TraceFactStatusV1::Failed,
        ),
        binary(
            !facts
                .iter()
                .any(|fact| fact.kind == TraceFactKindV1::Verification),
        ),
        log_kind_status(
            facts,
            TraceFactKindV1::ArtifactMutation,
            TraceFactStatusV1::Succeeded,
        ),
        log_kind_status(
            facts,
            TraceFactKindV1::ArtifactMutation,
            TraceFactStatusV1::Failed,
        ),
        log_kind_status(
            facts,
            TraceFactKindV1::ExternalAction,
            TraceFactStatusV1::Succeeded,
        ),
        log_kind_status(
            facts,
            TraceFactKindV1::ExternalAction,
            TraceFactStatusV1::Failed,
        ),
        log_kind_status(
            facts,
            TraceFactKindV1::ToolResult,
            TraceFactStatusV1::Succeeded,
        ),
        log_kind_status(
            facts,
            TraceFactKindV1::ToolResult,
            TraceFactStatusV1::Failed,
        ),
        log_kind_status(
            facts,
            TraceFactKindV1::ChildAgentResult,
            TraceFactStatusV1::Succeeded,
        ),
        log_kind_status(
            facts,
            TraceFactKindV1::ChildAgentResult,
            TraceFactStatusV1::Failed,
        ),
        log_count(unfinished as usize),
        log_count(failed as usize),
        log_count(succeeded as usize),
        binary(last.is_some_and(|fact| fact.status == TraceFactStatusV1::Failed)),
        binary(last.is_some_and(|fact| fact.status == TraceFactStatusV1::Succeeded)),
        ratio(
            last_failure_sequence.unwrap_or(0) as f64,
            max_sequence as f64,
        ),
        log_count(successes_after_last_failure),
        log_count(failures_after_last_success),
        log_count(distinct_tools),
        ratio(mandatory, fact_count),
        ratio(goal_relevant, fact_count),
        ratio(
            projection.token_budget.final_response_tokens as f64,
            projected_tokens,
        ),
        ratio(
            projection.token_budget.recovery_tokens as f64,
            projected_tokens,
        ),
        log_count(projection.goal.amendments.len()),
        log_count(projection.goal.requested_verification.len()),
        log_count(projection.goal.requested_side_effects.len()),
        log_count(projection.goal.constraints.len()),
    ]
}

fn status_count(facts: &[TaskCompletionTraceFactV1], status: TraceFactStatusV1) -> f64 {
    facts.iter().filter(|fact| fact.status == status).count() as f64
}

fn log_kind_status(
    facts: &[TaskCompletionTraceFactV1],
    kind: TraceFactKindV1,
    status: TraceFactStatusV1,
) -> f64 {
    log_count(
        facts
            .iter()
            .filter(|fact| fact.kind == kind && fact.status == status)
            .count(),
    )
}

fn log_count(value: usize) -> f64 {
    (value as f64).ln_1p()
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn binary(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}

fn validate_feature_record(
    record: &TaskCompletionEvidenceFeatureRecordV1,
) -> Result<(), ContractError> {
    if record.schema_version != TASK_COMPLETION_EVIDENCE_FEATURE_RECORD_SCHEMA_VERSION {
        return Err(training_error("unsupported feature record schema"));
    }
    if record.feature_set_version != TASK_COMPLETION_STRUCTURED_FEATURE_SET_VERSION {
        return Err(training_error("unsupported feature set version"));
    }
    require_sha256(
        &record.feature_record_id,
        "feature_record_id",
        training_error,
    )?;
    require_non_empty(&record.target_key, "target_key", training_error)?;
    require_non_empty(&record.target_revision, "target_revision", training_error)?;
    require_sha256(
        &record.trace_context_binding_id,
        "trace_context_binding_id",
        training_error,
    )?;
    require_sha256(&record.projection_hash, "projection_hash", training_error)?;
    require_non_empty(
        &record.projector_version,
        "projector_version",
        training_error,
    )?;
    if record.feature_names != FEATURE_NAMES.map(String::from) {
        return Err(training_error(format!(
            "feature names or ordering differ from {TASK_COMPLETION_STRUCTURED_FEATURE_SET_VERSION}"
        )));
    }
    if record.feature_values.len() != FEATURE_NAMES.len()
        || !record.feature_values.iter().all(|value| value.is_finite())
    {
        return Err(training_error(format!(
            "invalid feature vector for {}",
            record.target_key
        )));
    }
    let expected_id = canonical_content_id(
        TASK_COMPLETION_EVIDENCE_FEATURE_RECORD_SCHEMA_VERSION,
        &json!({
            "feature_set_version": record.feature_set_version,
            "target_key": record.target_key,
            "target_revision": record.target_revision,
            "trace_context_binding_id": record.trace_context_binding_id,
            "projection_hash": record.projection_hash,
            "projector_version": record.projector_version,
            "feature_names": record.feature_names,
            "feature_values": record.feature_values,
        }),
    )?;
    if record.feature_record_id != expected_id {
        return Err(training_error(format!(
            "feature record identity does not match its content for {}",
            record.target_key
        )));
    }
    Ok(())
}

fn training_error(message: impl Into<String>) -> ContractError {
    ContractError::InvalidTaskCompletion(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    use crate::learned::{
        COMPACT_TASK_COMPLETION_PROJECTION_SCHEMA_VERSION, CompactTaskCompletionProjectionStatsV1,
        CompactTaskCompletionTokenBudgetV1, CompactTaskCompletionVariantV1,
        EvaluationEvidenceCatalogV1, EvaluationEvidenceKindV1, EvaluationEvidenceLocationV1,
        EvaluationEvidenceRecordV1, TaskCompletionEvidenceLaneV1, TraceFactActorV1,
    };

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn projection() -> CompactTaskCompletionProjectionV1 {
        let placeholder = digest('0');
        let fact = TaskCompletionTraceFactV1 {
            evidence_id: "E0001".into(),
            evidence_key: "verification".into(),
            sequence: 1,
            actor: TraceFactActorV1::Tool,
            kind: TraceFactKindV1::Verification,
            status: TraceFactStatusV1::Succeeded,
            lane: TaskCompletionEvidenceLaneV1::Mandatory,
            mandatory: true,
            span_id: Some("span-1".into()),
            parent_span_id: None,
            tool_name: Some("test".into()),
            summary: "All tests passed.".into(),
            structured_facts: BTreeMap::new(),
            token_count: 5,
        };
        CompactTaskCompletionProjectionV1 {
            schema_version: COMPACT_TASK_COMPLETION_PROJECTION_SCHEMA_VERSION.into(),
            projector_version: "traceeval.compact-projector.test-v1".into(),
            variant: CompactTaskCompletionVariantV1::MandatoryEvidence,
            target_key: "trace-1".into(),
            target_revision: "revision-1".into(),
            trace_context_binding_id: digest('1'),
            context_release_id: None,
            context_projection_release_id: None,
            projection_hash: placeholder.clone(),
            goal: TaskCompletionGoalBundleV1 {
                primary_request: "Fix the bug and run tests.".into(),
                amendments: Vec::new(),
                success_criteria: vec!["The tests pass.".into()],
                requested_side_effects: Vec::new(),
                requested_verification: vec!["Run the tests.".into()],
                constraints: Vec::new(),
                agent_context: vec!["Rust project.".into()],
                superseded_requirements: Vec::new(),
                token_count: 10,
            },
            facts: vec![fact],
            recovery_chains: Vec::new(),
            token_budget: CompactTaskCompletionTokenBudgetV1 {
                tokenizer_id: "test-tokenizer".into(),
                max_input_tokens: 6_144,
                original_tokens: 100,
                projected_tokens: 20,
                rubric_tokens: 5,
                goal_tokens: 10,
                final_response_tokens: 0,
                mandatory_tokens: 5,
                recovery_tokens: 0,
                goal_relevant_tokens: 0,
                metadata_tokens: 0,
            },
            stats: CompactTaskCompletionProjectionStatsV1 {
                included_facts: 1,
                omitted_facts: 0,
                mandatory_facts: 1,
                mandatory_facts_omitted: 0,
            },
            evidence_catalog: EvaluationEvidenceCatalogV1 {
                target_key: "trace-1".into(),
                target_revision: "revision-1".into(),
                projection_hash: placeholder.clone(),
                entries: BTreeMap::from([(
                    "verification".into(),
                    EvaluationEvidenceRecordV1 {
                        target_key: "trace-1".into(),
                        target_revision: "revision-1".into(),
                        projection_hash: placeholder,
                        evidence_kind: EvaluationEvidenceKindV1::Span,
                        location: EvaluationEvidenceLocationV1::Span {
                            span_id: "span-1".into(),
                        },
                        applicable_criterion_ids: BTreeSet::new(),
                    },
                )]),
            },
        }
        .seal()
        .unwrap()
    }

    #[test]
    fn ratio_handles_absent_evidence() {
        assert_eq!(ratio(0.0, 0.0), 0.0);
        assert_eq!(ratio(1.0, 4.0), 0.25);
    }

    #[test]
    fn feature_schema_has_unique_names() {
        assert_eq!(
            FEATURE_NAMES.iter().copied().collect::<BTreeSet<_>>().len(),
            FEATURE_NAMES.len()
        );
    }

    #[test]
    fn feature_record_validation_rejects_tampering() {
        let names = FEATURE_NAMES.map(String::from).to_vec();
        let values = vec![0.0; FEATURE_NAMES.len()];
        let mut record = TaskCompletionEvidenceFeatureRecordV1 {
            schema_version: TASK_COMPLETION_EVIDENCE_FEATURE_RECORD_SCHEMA_VERSION.into(),
            feature_set_version: TASK_COMPLETION_STRUCTURED_FEATURE_SET_VERSION.into(),
            feature_record_id: String::new(),
            target_key: "trace-1".into(),
            target_revision: "revision-1".into(),
            trace_context_binding_id: digest('1'),
            projection_hash: digest('2'),
            projector_version: "projector-v1".into(),
            feature_names: names,
            feature_values: values,
        };
        record.feature_record_id = canonical_content_id(
            TASK_COMPLETION_EVIDENCE_FEATURE_RECORD_SCHEMA_VERSION,
            &json!({
                "feature_set_version": record.feature_set_version,
                "target_key": record.target_key,
                "target_revision": record.target_revision,
                "trace_context_binding_id": record.trace_context_binding_id,
                "projection_hash": record.projection_hash,
                "projector_version": record.projector_version,
                "feature_names": record.feature_names,
                "feature_values": record.feature_values,
            }),
        )
        .unwrap();
        assert!(validate_feature_record(&record).is_ok());
        record.feature_values[0] = 1.0;
        assert!(validate_feature_record(&record).is_err());
    }

    #[test]
    fn training_record_is_projection_bound_and_label_free() {
        let projection = projection();
        let mut record = TaskCompletionTrainingRecordV1::from_projection(&projection).unwrap();
        record.validate_against(&projection).unwrap();

        let serialized = serde_json::to_value(&record).unwrap();
        assert!(serialized.get("source").is_none());
        assert!(serialized.get("label").is_none());
        assert!(serialized.get("split").is_none());
        assert!(serialized.get("reward").is_none());

        record.facts[0].summary = "tampered".into();
        assert!(record.validate_against(&projection).is_err());
    }

    #[test]
    fn training_record_validation_rejects_tampered_structured_features() {
        let projection = projection();
        let mut record = TaskCompletionTrainingRecordV1::from_projection(&projection).unwrap();
        record.structured_features.feature_values[0] += 1.0;
        record.structured_features.feature_record_id = canonical_content_id(
            TASK_COMPLETION_EVIDENCE_FEATURE_RECORD_SCHEMA_VERSION,
            &json!({
                "feature_set_version": record.structured_features.feature_set_version,
                "target_key": record.structured_features.target_key,
                "target_revision": record.structured_features.target_revision,
                "trace_context_binding_id": record.structured_features.trace_context_binding_id,
                "projection_hash": record.structured_features.projection_hash,
                "projector_version": record.structured_features.projector_version,
                "feature_names": record.structured_features.feature_names,
                "feature_values": record.structured_features.feature_values,
            }),
        )
        .unwrap();
        record.training_record_id = training_record_id(&projection, &record.structured_features).unwrap();

        assert!(record.validate_against(&projection).is_err());
    }
}
