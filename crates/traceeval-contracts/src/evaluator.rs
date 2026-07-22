use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    ContractError, EVALUATOR_RELEASE_HASH_DOMAIN, canonical_content_id, require_non_empty,
    require_sha256,
};

pub const EVALUATOR_RELEASE_SCHEMA_VERSION: &str = "traceeval.evaluator_release.v1";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnedTaskKind {
    TaskCompletion,
    Hallucination,
    SafetyPolicyAdherence,
    ToolUseCorrectness,
    Usefulness,
    UserFrustration,
    StepEfficiency,
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationTargetKind {
    Span,
    TraceRevision,
    SessionSnapshot,
    Claim,
    ToolCall,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationInputBoundsV1 {
    pub max_subjects: u32,
    pub max_evidence_items: u32,
    pub max_input_bytes: u64,
    pub max_output_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvaluationImplementationV1 {
    PromptJudge {
        provider: String,
        requested_model: String,
        system_prompt: String,
        rubric: String,
        response_schema: Value,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        decoding_parameters: BTreeMap<String, Value>,
        parser_version: String,
        normalizer_version: String,
    },
    LocalClassifier {
        model_artifact_id: String,
        tokenizer_artifact_id: String,
        feature_schema_id: String,
        runtime_version: String,
    },
    EmbeddingLinear {
        embedding_release_id: String,
        weights_artifact_id: String,
        feature_schema_id: String,
    },
    Hybrid {
        #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
        component_release_ids: BTreeSet<String>,
        aggregation_spec: Value,
    },
    Ensemble {
        #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
        member_release_ids: BTreeSet<String>,
        aggregation_spec: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluatorReleaseSpecV1 {
    pub schema_version: String,
    pub name: String,
    pub task_kind: LearnedTaskKind,
    pub target_kind: EvaluationTargetKind,
    pub implementation: EvaluationImplementationV1,
    pub projection_release_id: String,
    pub context_projection_release_id: String,
    /// Exact immutable taxonomy release used to interpret applicability IDs.
    /// Stable node IDs alone are insufficient because their active state,
    /// definition, and lineage can change between taxonomy releases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applicable_taxonomy_release_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub applicable_taxonomy_node_ids: BTreeSet<String>,
    pub input_bounds: EvaluationInputBoundsV1,
    pub evidence_schema_version: String,
    pub abstention_policy: Value,
    pub code_artifact_hash: String,
}

impl EvaluatorReleaseSpecV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != EVALUATOR_RELEASE_SCHEMA_VERSION {
            return Err(evaluator_error(
                "unsupported evaluator release schema version",
            ));
        }
        require_non_empty(&self.name, "name", evaluator_error)?;
        if let LearnedTaskKind::Custom(name) = &self.task_kind {
            require_non_empty(name, "custom task name", evaluator_error)?;
        }
        require_sha256(
            &self.projection_release_id,
            "projection_release_id",
            evaluator_error,
        )?;
        require_sha256(
            &self.context_projection_release_id,
            "context_projection_release_id",
            evaluator_error,
        )?;
        match (
            self.applicable_taxonomy_release_id.as_deref(),
            self.applicable_taxonomy_node_ids.is_empty(),
        ) {
            (Some(release_id), false) => require_sha256(
                release_id,
                "applicable_taxonomy_release_id",
                evaluator_error,
            )?,
            (None, false) => {
                return Err(evaluator_error(
                    "taxonomy applicability node IDs require an exact taxonomy release ID",
                ));
            }
            (Some(_), true) => {
                return Err(evaluator_error(
                    "taxonomy release applicability requires at least one taxonomy node ID",
                ));
            }
            (None, true) => {}
        }
        require_non_empty(
            &self.evidence_schema_version,
            "evidence_schema_version",
            evaluator_error,
        )?;
        require_sha256(
            &self.code_artifact_hash,
            "code_artifact_hash",
            evaluator_error,
        )?;
        if self.input_bounds.max_subjects == 0
            || self.input_bounds.max_evidence_items == 0
            || self.input_bounds.max_input_bytes == 0
            || self.input_bounds.max_output_bytes == 0
        {
            return Err(evaluator_error(
                "all input bounds must be greater than zero",
            ));
        }

        match &self.implementation {
            EvaluationImplementationV1::PromptJudge {
                provider,
                requested_model,
                system_prompt,
                rubric,
                response_schema,
                parser_version,
                normalizer_version,
                ..
            } => {
                require_non_empty(provider, "provider", evaluator_error)?;
                require_non_empty(requested_model, "requested_model", evaluator_error)?;
                require_non_empty(system_prompt, "system_prompt", evaluator_error)?;
                require_non_empty(rubric, "rubric", evaluator_error)?;
                require_non_empty(parser_version, "parser_version", evaluator_error)?;
                require_non_empty(normalizer_version, "normalizer_version", evaluator_error)?;
                if !response_schema.is_object() {
                    return Err(evaluator_error("response_schema must be a JSON object"));
                }
            }
            EvaluationImplementationV1::LocalClassifier {
                model_artifact_id,
                tokenizer_artifact_id,
                feature_schema_id,
                runtime_version,
            } => {
                require_sha256(model_artifact_id, "model_artifact_id", evaluator_error)?;
                require_sha256(
                    tokenizer_artifact_id,
                    "tokenizer_artifact_id",
                    evaluator_error,
                )?;
                require_non_empty(feature_schema_id, "feature_schema_id", evaluator_error)?;
                require_non_empty(runtime_version, "runtime_version", evaluator_error)?;
            }
            EvaluationImplementationV1::EmbeddingLinear {
                embedding_release_id,
                weights_artifact_id,
                feature_schema_id,
            } => {
                require_sha256(
                    embedding_release_id,
                    "embedding_release_id",
                    evaluator_error,
                )?;
                require_sha256(weights_artifact_id, "weights_artifact_id", evaluator_error)?;
                require_non_empty(feature_schema_id, "feature_schema_id", evaluator_error)?;
            }
            EvaluationImplementationV1::Hybrid {
                component_release_ids,
                aggregation_spec,
            } => validate_composition(component_release_ids, aggregation_spec, "hybrid")?,
            EvaluationImplementationV1::Ensemble {
                member_release_ids,
                aggregation_spec,
            } => validate_composition(member_release_ids, aggregation_spec, "ensemble")?,
        }
        Ok(())
    }

    pub fn release_id(&self) -> Result<String, ContractError> {
        self.validate()?;
        canonical_content_id(EVALUATOR_RELEASE_HASH_DOMAIN, self)
    }
}

fn validate_composition(
    release_ids: &BTreeSet<String>,
    aggregation_spec: &Value,
    kind: &str,
) -> Result<(), ContractError> {
    if release_ids.len() < 2 {
        return Err(evaluator_error(format!(
            "{kind} requires at least two component releases"
        )));
    }
    if !aggregation_spec.is_object() {
        return Err(evaluator_error(format!(
            "{kind} aggregation_spec must be a JSON object"
        )));
    }
    for release_id in release_ids {
        require_sha256(release_id, "component release id", evaluator_error)?;
    }
    Ok(())
}

fn evaluator_error(message: impl Into<String>) -> ContractError {
    ContractError::InvalidEvaluator(message.into())
}
