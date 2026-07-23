#![forbid(unsafe_code)]

//! Stable, provider-independent contracts shared by trace projectors,
//! applications, training pipelines, and local model runtimes.

mod canonical;
mod compact_task_completion;
mod context;
mod evaluation;
mod evaluator;
mod provider;
mod task_completion_training;
mod taxonomy;

pub use canonical::{
    AGENT_CONTEXT_RELEASE_HASH_DOMAIN, AGENT_TAXONOMY_RELEASE_HASH_DOMAIN,
    CONTEXT_PROJECTION_HASH_DOMAIN, EVALUATOR_RELEASE_HASH_DOMAIN, TAXONOMY_ASSIGNMENT_HASH_DOMAIN,
    TRACE_CONTEXT_BINDING_HASH_DOMAIN, canonical_content_id, canonical_json_bytes,
};
pub use compact_task_completion::{
    BINARY_TASK_COMPLETION_DECISION_SCHEMA_VERSION, BinaryTaskCompletionDecisionV1,
    BinaryTaskCompletionOutcomeV1, COMPACT_TASK_COMPLETION_PROJECTION_SCHEMA_VERSION,
    CompactTaskCompletionProjectionStatsV1, CompactTaskCompletionProjectionV1,
    CompactTaskCompletionTokenBudgetV1, CompactTaskCompletionVariantV1,
    TaskCompletionEvidenceLaneV1, TaskCompletionGoalBundleV1, TaskCompletionInferenceProvenanceV1,
    TaskCompletionRecoveryChainV1, TaskCompletionTraceFactV1, TraceFactActorV1, TraceFactKindV1,
    TraceFactStatusV1,
};
pub use context::{
    AGENT_CONTEXT_RELEASE_SCHEMA_VERSION, AgentArchitectureContextV1, AgentCapabilityV1,
    AgentContextReleaseV1, AgentEvaluationContextV1, AgentIdentityContextV1, AgentIntentContextV1,
    AgentPolicyContextV1, CapabilityEffectV1, CapabilityKindV1, ContextFieldMetadataV1,
    ContextFieldProvenanceV1, ContextFieldV1, ContextProjectionClassV1, ContextProjectionV1,
    ContextReviewStateV1, ContextSensitivityV1, IdempotencyClassV1, SuccessCriterionImportanceV1,
    SuccessCriterionV1, TRACE_CONTEXT_BINDING_SCHEMA_VERSION, TraceContextBindingProvenanceV1,
    TraceContextBindingResolutionV1, TraceContextBindingV1,
};
pub use evaluation::{
    EvaluationCriterionV1, EvaluationEvidenceCatalogV1, EvaluationEvidenceCitationV1,
    EvaluationEvidenceKindV1, EvaluationEvidenceLocationV1, EvaluationEvidenceRecordV1,
    LEARNED_EVALUATION_SCHEMA_VERSION, LearnedAbstentionReasonV1, LearnedEvaluationV1,
    LearnedVerdictV1,
};
pub use evaluator::{
    EVALUATOR_RELEASE_SCHEMA_VERSION, EvaluationImplementationV1, EvaluationInputBoundsV1,
    EvaluationTargetKind, EvaluatorReleaseSpecV1, LearnedTaskKind,
};
pub use provider::{
    CHAT_COMPLETION_ENVELOPE_SCHEMA_VERSION, ChatCompletionEnvelopeV1, ProviderExecutionFailureV1,
    ProviderExecutionStageV1, ProviderResponseEnvelopeV1, ProviderTokenUsageV1,
};
pub use task_completion_training::{
    FEATURE_NAMES as TASK_COMPLETION_STRUCTURED_FEATURE_NAMES,
    TASK_COMPLETION_EVIDENCE_FEATURE_RECORD_SCHEMA_VERSION,
    TASK_COMPLETION_STRUCTURED_FEATURE_SET_VERSION, TASK_COMPLETION_TRAINING_RECORD_SCHEMA_VERSION,
    TaskCompletionEvidenceFeatureRecordV1, TaskCompletionStructuredEvidenceFeaturesV2,
    TaskCompletionTrainingRecordV1,
};
pub use taxonomy::{
    AGENT_TAXONOMY_RELEASE_SCHEMA_VERSION, AgentTaxonomyReleaseV1, TaxonomyAssignmentSourceV1,
    TaxonomyAssignmentV1, TaxonomyDimensionV1, TaxonomyLineageOperationV1, TaxonomyNodeStateV1,
    TaxonomyNodeV1, TaxonomyOpenSetStateV1, TaxonomyRelationKindV1, TaxonomyRelationV1,
};

#[derive(Debug, thiserror::Error)]
pub enum ContractError {
    #[error("failed to canonicalize contract JSON: {0}")]
    CanonicalJson(#[from] serde_json::Error),
    #[error("invalid evaluator release: {0}")]
    InvalidEvaluator(String),
    #[error("invalid agent context: {0}")]
    InvalidContext(String),
    #[error("invalid taxonomy: {0}")]
    InvalidTaxonomy(String),
    #[error("invalid learned evaluation: {0}")]
    InvalidEvaluation(String),
    #[error("invalid learned calibration: {0}")]
    InvalidCalibration(String),
    #[error("invalid human agreement report: {0}")]
    InvalidAgreement(String),
    #[error("invalid task-completion contract: {0}")]
    InvalidTaskCompletion(String),
    #[error("invalid provider envelope: {0}")]
    InvalidProvider(String),
}

fn require_non_empty(
    value: &str,
    field: &str,
    error: fn(String) -> ContractError,
) -> Result<(), ContractError> {
    if value.trim().is_empty() {
        return Err(error(format!("{field} must not be empty")));
    }
    Ok(())
}

fn require_sha256(
    value: &str,
    field: &str,
    error: fn(String) -> ContractError,
) -> Result<(), ContractError> {
    let digest = value.strip_prefix("sha256:").unwrap_or_default();
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(error(format!("{field} must be a sha256: content identity")));
    }
    Ok(())
}
