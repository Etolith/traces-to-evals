use std::collections::{BTreeMap, BTreeSet};

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    AGENT_CONTEXT_RELEASE_HASH_DOMAIN, CONTEXT_PROJECTION_HASH_DOMAIN, ContractError,
    TRACE_CONTEXT_BINDING_HASH_DOMAIN, canonical_content_id, require_non_empty, require_sha256,
};

pub const AGENT_CONTEXT_RELEASE_SCHEMA_VERSION: &str = "traceeval.agent_context_release.v1";
pub const TRACE_CONTEXT_BINDING_SCHEMA_VERSION: &str = "traceeval.trace_context_binding.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextFieldProvenanceV1 {
    UserDeclared,
    ConfigImport,
    ToolSchema,
    TelemetryInferred,
    SystemInferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextReviewStateV1 {
    Unreviewed,
    Approved,
    Rejected,
    Conflicting,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextSensitivityV1 {
    Public,
    Internal,
    HostedPreRedacted,
    SensitiveLocalOnly,
    Secret,
    Credential,
    HiddenLabel,
    ExpectedAnswer,
    Unclassified,
}

impl ContextSensitivityV1 {
    fn permits_hosted(self) -> bool {
        matches!(self, Self::Public | Self::HostedPreRedacted)
    }

    fn forbidden(self) -> bool {
        matches!(
            self,
            Self::Secret
                | Self::Credential
                | Self::HiddenLabel
                | Self::ExpectedAnswer
                | Self::Unclassified
        )
    }
}

/// Provenance shared by every value that can enter an activated context release.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextFieldMetadataV1 {
    pub field_id: String,
    pub provenance: ContextFieldProvenanceV1,
    pub source_snapshot_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_locator: Option<String>,
    pub captured_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fresh_until: Option<String>,
    pub review_state: ContextReviewStateV1,
    pub sensitivity: ContextSensitivityV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_confidence: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextFieldV1 {
    #[serde(flatten)]
    pub metadata: ContextFieldMetadataV1,
    pub value: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKindV1 {
    Tool,
    SubAgent,
    HumanHandoff,
    ExternalService,
    InternalOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEffectV1 {
    ReadOnly,
    Mutating,
    Verifying,
    Escalating,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdempotencyClassV1 {
    Idempotent,
    IdempotentWithKey,
    NonIdempotent,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentCapabilityV1 {
    #[serde(flatten)]
    pub metadata: ContextFieldMetadataV1,
    pub capability_id: String,
    pub name: String,
    pub kind: CapabilityKindV1,
    pub effect: CapabilityEffectV1,
    pub idempotency: IdempotencyClassV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argument_schema_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_schema_digest: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub permissions: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowed_operations: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub prohibited_operations: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub required_preconditions: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub budgets: BTreeMap<String, u64>,
    #[serde(default)]
    pub requires_approval: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuccessCriterionImportanceV1 {
    Must,
    Should,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuccessCriterionV1 {
    #[serde(flatten)]
    pub metadata: ContextFieldMetadataV1,
    pub criterion_id: String,
    pub description: String,
    pub importance: SuccessCriterionImportanceV1,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub required_evidence_kinds: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub business_impact_weight: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentIdentityContextV1 {
    pub application_name: ContextFieldV1,
    pub owner: ContextFieldV1,
    pub environment: ContextFieldV1,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub build_version_selectors: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entry_points: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub user_personas: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_domains: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub languages: Vec<ContextFieldV1>,
    pub risk_tier: ContextFieldV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentIntentContextV1 {
    pub purpose: ContextFieldV1,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_tasks: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub explicit_non_goals: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub success_criteria: Vec<SuccessCriterionV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptable_partial_completion: Option<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refusal_requirements: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub escalation_requirements: Vec<ContextFieldV1>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentArchitectureContextV1 {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub routers: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_agents: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retrieval_data_sources: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub human_handoffs: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub external_services: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_causal_topology: Vec<ContextFieldV1>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentPolicyContextV1 {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data_classifications: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retention_rules: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redaction_rules: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub external_provider_permissions: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compliance_constraints: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub learned_feature_content: Vec<ContextFieldV1>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentEvaluationContextV1 {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reusable_rubrics: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub safe_positive_examples: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub safe_negative_examples: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub known_limitations: Vec<ContextFieldV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_evidence_types: Vec<ContextFieldV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentContextReleaseV1 {
    pub schema_version: String,
    pub agent_id: String,
    pub identity: AgentIdentityContextV1,
    pub intent: AgentIntentContextV1,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<AgentCapabilityV1>,
    #[serde(default)]
    pub architecture: AgentArchitectureContextV1,
    #[serde(default)]
    pub policy: AgentPolicyContextV1,
    #[serde(default)]
    pub evaluation_context: AgentEvaluationContextV1,
}

impl AgentContextReleaseV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != AGENT_CONTEXT_RELEASE_SCHEMA_VERSION {
            return Err(context_error("unsupported agent context schema version"));
        }
        require_non_empty(&self.agent_id, "agent_id", context_error)?;

        let mut field_ids = BTreeSet::new();
        for field in self.fields() {
            validate_field_metadata(&field.metadata)?;
            validate_context_value(&field.metadata.field_id, &field.value)?;
            insert_field_id(&mut field_ids, &field.metadata.field_id)?;
        }

        let mut criterion_ids = BTreeSet::new();
        for criterion in &self.intent.success_criteria {
            validate_field_metadata(&criterion.metadata)?;
            insert_field_id(&mut field_ids, &criterion.metadata.field_id)?;
            require_non_empty(&criterion.criterion_id, "criterion_id", context_error)?;
            require_non_empty(
                &criterion.description,
                "criterion description",
                context_error,
            )?;
            validate_probability(criterion.business_impact_weight, "business_impact_weight")?;
            validate_context_value(
                &criterion.metadata.field_id,
                &serde_json::to_value(criterion).map_err(context_serialization_error)?,
            )?;
            if !criterion_ids.insert(criterion.criterion_id.as_str()) {
                return Err(context_error(format!(
                    "duplicate success criterion {}",
                    criterion.criterion_id
                )));
            }
        }

        let mut capability_ids = BTreeSet::new();
        for capability in &self.capabilities {
            validate_field_metadata(&capability.metadata)?;
            insert_field_id(&mut field_ids, &capability.metadata.field_id)?;
            require_non_empty(&capability.capability_id, "capability_id", context_error)?;
            require_non_empty(&capability.name, "capability name", context_error)?;
            for digest in [
                capability.argument_schema_digest.as_deref(),
                capability.result_schema_digest.as_deref(),
            ]
            .into_iter()
            .flatten()
            {
                require_sha256(digest, "capability schema digest", context_error)?;
            }
            validate_context_value(
                &capability.metadata.field_id,
                &serde_json::to_value(capability).map_err(context_serialization_error)?,
            )?;
            if !capability_ids.insert(capability.capability_id.as_str()) {
                return Err(context_error(format!(
                    "ambiguous duplicate capability identifier {}",
                    capability.capability_id
                )));
            }
        }
        Ok(())
    }

    pub fn release_id(&self) -> Result<String, ContractError> {
        self.validate()?;
        canonical_content_id(AGENT_CONTEXT_RELEASE_HASH_DOMAIN, self)
    }

    fn field_metadata(&self, field_id: &str) -> Option<&ContextFieldMetadataV1> {
        self.fields()
            .into_iter()
            .map(|field| &field.metadata)
            .chain(
                self.intent
                    .success_criteria
                    .iter()
                    .map(|criterion| &criterion.metadata),
            )
            .chain(
                self.capabilities
                    .iter()
                    .map(|capability| &capability.metadata),
            )
            .find(|metadata| metadata.field_id == field_id)
    }

    fn fields(&self) -> Vec<&ContextFieldV1> {
        let mut fields = vec![
            &self.identity.application_name,
            &self.identity.owner,
            &self.identity.environment,
            &self.identity.risk_tier,
            &self.intent.purpose,
        ];
        fields.extend(self.identity.build_version_selectors.iter());
        fields.extend(self.identity.entry_points.iter());
        fields.extend(self.identity.user_personas.iter());
        fields.extend(self.identity.supported_domains.iter());
        fields.extend(self.identity.languages.iter());
        fields.extend(self.intent.supported_tasks.iter());
        fields.extend(self.intent.explicit_non_goals.iter());
        fields.extend(self.intent.acceptable_partial_completion.iter());
        fields.extend(self.intent.refusal_requirements.iter());
        fields.extend(self.intent.escalation_requirements.iter());
        fields.extend(self.architecture.routers.iter());
        fields.extend(self.architecture.sub_agents.iter());
        fields.extend(self.architecture.memory.iter());
        fields.extend(self.architecture.retrieval_data_sources.iter());
        fields.extend(self.architecture.human_handoffs.iter());
        fields.extend(self.architecture.external_services.iter());
        fields.extend(self.architecture.expected_causal_topology.iter());
        fields.extend(self.policy.data_classifications.iter());
        fields.extend(self.policy.retention_rules.iter());
        fields.extend(self.policy.redaction_rules.iter());
        fields.extend(self.policy.external_provider_permissions.iter());
        fields.extend(self.policy.compliance_constraints.iter());
        fields.extend(self.policy.learned_feature_content.iter());
        fields.extend(self.evaluation_context.reusable_rubrics.iter());
        fields.extend(self.evaluation_context.safe_positive_examples.iter());
        fields.extend(self.evaluation_context.safe_negative_examples.iter());
        fields.extend(self.evaluation_context.known_limitations.iter());
        fields.extend(self.evaluation_context.required_evidence_types.iter());
        fields
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextProjectionClassV1 {
    StructuralOnly,
    LocalContent,
    HostedPreRedacted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextProjectionV1 {
    pub context_release_id: String,
    pub projection_class: ContextProjectionClassV1,
    pub projector_version: String,
    pub redaction_version: String,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub included_field_ids: BTreeSet<String>,
}

impl ContextProjectionV1 {
    pub fn validate_against(&self, context: &AgentContextReleaseV1) -> Result<(), ContractError> {
        context.validate()?;
        if self.context_release_id != context.release_id()? {
            return Err(context_error(
                "projection context_release_id does not match context",
            ));
        }
        require_non_empty(&self.projector_version, "projector_version", context_error)?;
        require_non_empty(&self.redaction_version, "redaction_version", context_error)?;
        for field_id in &self.included_field_ids {
            let metadata = context.field_metadata(field_id).ok_or_else(|| {
                context_error(format!("projection references unknown field {field_id}"))
            })?;
            if self.projection_class == ContextProjectionClassV1::HostedPreRedacted
                && !metadata.sensitivity.permits_hosted()
            {
                return Err(context_error(format!(
                    "field {field_id} is not classified for hosted projection"
                )));
            }
        }
        Ok(())
    }

    /// Returns the immutable identity of the selected fields, projection
    /// class, projector, and redaction policy.
    pub fn release_id(&self) -> Result<String, ContractError> {
        require_sha256(
            &self.context_release_id,
            "context_release_id",
            context_error,
        )?;
        require_non_empty(&self.projector_version, "projector_version", context_error)?;
        require_non_empty(&self.redaction_version, "redaction_version", context_error)?;
        canonical_content_id(CONTEXT_PROJECTION_HASH_DOMAIN, self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceContextBindingResolutionV1 {
    Resolved,
    Unresolved,
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceContextBindingProvenanceV1 {
    ExplicitInstrumentation,
    SelectorRule,
    ReviewedProjectDefault,
    Backfill,
    NoSelectorMatch,
    MultipleSelectorMatches,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceContextBindingV1 {
    pub schema_version: String,
    pub target_key: String,
    pub target_revision: String,
    pub resolution: TraceContextBindingResolutionV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_context_release_id: Option<String>,
    pub binding_rule_release_id: String,
    pub binding_provenance: TraceContextBindingProvenanceV1,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub candidate_context_release_ids: BTreeSet<String>,
}

#[derive(Serialize)]
struct TraceContextBindingIdentity<'a> {
    target_key: &'a str,
    target_revision: &'a str,
    resolution: TraceContextBindingResolutionV1,
    agent_context_release_id: Option<&'a str>,
    binding_rule_release_id: &'a str,
    binding_provenance: TraceContextBindingProvenanceV1,
    candidate_context_release_ids: &'a BTreeSet<String>,
}

impl TraceContextBindingV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != TRACE_CONTEXT_BINDING_SCHEMA_VERSION {
            return Err(context_error(
                "unsupported trace context binding schema version",
            ));
        }
        require_non_empty(&self.target_key, "target_key", context_error)?;
        require_non_empty(&self.target_revision, "target_revision", context_error)?;
        require_sha256(
            &self.binding_rule_release_id,
            "binding_rule_release_id",
            context_error,
        )?;
        for release_id in self
            .agent_context_release_id
            .iter()
            .chain(self.candidate_context_release_ids.iter())
        {
            require_sha256(release_id, "agent_context_release_id", context_error)?;
        }
        match self.resolution {
            TraceContextBindingResolutionV1::Resolved => {
                if self.agent_context_release_id.is_none()
                    || !self.candidate_context_release_ids.is_empty()
                {
                    return Err(context_error(
                        "resolved binding requires one context release and no candidates",
                    ));
                }
            }
            TraceContextBindingResolutionV1::Unresolved => {
                if self.agent_context_release_id.is_some()
                    || !self.candidate_context_release_ids.is_empty()
                {
                    return Err(context_error(
                        "unresolved binding cannot select or propose context releases",
                    ));
                }
            }
            TraceContextBindingResolutionV1::Ambiguous => {
                if self.agent_context_release_id.is_some()
                    || self.candidate_context_release_ids.len() < 2
                {
                    return Err(context_error(
                        "ambiguous binding requires at least two candidate releases",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn binding_id(&self) -> Result<String, ContractError> {
        self.validate()?;
        canonical_content_id(
            TRACE_CONTEXT_BINDING_HASH_DOMAIN,
            &TraceContextBindingIdentity {
                target_key: &self.target_key,
                target_revision: &self.target_revision,
                resolution: self.resolution,
                agent_context_release_id: self.agent_context_release_id.as_deref(),
                binding_rule_release_id: &self.binding_rule_release_id,
                binding_provenance: self.binding_provenance,
                candidate_context_release_ids: &self.candidate_context_release_ids,
            },
        )
    }
}

fn validate_field_metadata(metadata: &ContextFieldMetadataV1) -> Result<(), ContractError> {
    require_non_empty(&metadata.field_id, "field_id", context_error)?;
    require_sha256(
        &metadata.source_snapshot_id,
        "source_snapshot_id",
        context_error,
    )?;
    require_non_empty(&metadata.captured_at, "captured_at", context_error)?;
    if let Some(source_locator) = &metadata.source_locator
        && contains_forbidden_scalar(source_locator)
    {
        return Err(context_error(format!(
            "field {} source locator contains forbidden sensitive material",
            metadata.field_id
        )));
    }
    if metadata.review_state != ContextReviewStateV1::Approved {
        return Err(context_error(format!(
            "field {} must be approved before release activation",
            metadata.field_id
        )));
    }
    if metadata.sensitivity.forbidden() {
        return Err(context_error(format!(
            "field {} has forbidden sensitivity {:?}",
            metadata.field_id, metadata.sensitivity
        )));
    }
    validate_probability(metadata.inference_confidence, "inference_confidence")?;
    if matches!(
        metadata.provenance,
        ContextFieldProvenanceV1::TelemetryInferred | ContextFieldProvenanceV1::SystemInferred
    ) && metadata.inference_confidence.is_none()
    {
        return Err(context_error(format!(
            "inferred field {} requires inference_confidence",
            metadata.field_id
        )));
    }
    Ok(())
}

fn validate_context_value(field_id: &str, value: &Value) -> Result<(), ContractError> {
    if contains_forbidden_key(field_id) || value_contains_forbidden_text(value) {
        return Err(context_error(format!(
            "field {field_id} contains a forbidden label, expected answer, or credential"
        )));
    }
    Ok(())
}

fn insert_field_id<'a>(
    field_ids: &mut BTreeSet<&'a str>,
    field_id: &'a str,
) -> Result<(), ContractError> {
    if !field_ids.insert(field_id) {
        return Err(context_error(format!(
            "duplicate context field_id {field_id}"
        )));
    }
    Ok(())
}

fn value_contains_forbidden_text(value: &Value) -> bool {
    match value {
        Value::String(value) => contains_forbidden_scalar(value),
        Value::Object(object) => object.iter().any(|(key, value)| {
            contains_forbidden_key(key) || value_contains_forbidden_text(value)
        }),
        Value::Array(values) => values.iter().any(value_contains_forbidden_text),
        _ => false,
    }
}

fn normalize_for_leakage_scan(value: &str) -> (String, String) {
    let normalized: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let compact: String = normalized
        .chars()
        .filter(|character| *character != '_')
        .collect();
    (normalized, compact)
}

fn contains_forbidden_key(value: &str) -> bool {
    let (normalized, compact) = normalize_for_leakage_scan(value);
    let forbidden = [
        "benchmark_gold",
        "gold_label",
        "hidden_label",
        "expected_answer",
        "expected_output",
        "reference_answer",
        "main_task_success",
        "side_task_success",
        "api_key",
        "access_token",
        "client_secret",
        "password",
        "credential",
        "secret_key",
    ];
    forbidden.iter().any(|term| {
        normalized.contains(term)
            || compact.contains(
                &term
                    .chars()
                    .filter(|character| *character != '_')
                    .collect::<String>(),
            )
    })
}

fn contains_forbidden_scalar(value: &str) -> bool {
    let (normalized, compact) = normalize_for_leakage_scan(value);
    let forbidden_claims = [
        "benchmark_gold",
        "gold_label",
        "hidden_label",
        "expected_answer",
        "expected_output",
        "reference_answer",
        "main_task_success",
        "side_task_success",
    ];
    forbidden_claims.iter().any(|term| {
        normalized.contains(term)
            || compact.contains(
                &term
                    .chars()
                    .filter(|character| *character != '_')
                    .collect::<String>(),
            )
    }) || contains_credential_assignment(value)
        || value.contains("sk-")
        || value.contains("ghp_")
        || value.contains("github_pat_")
        || value.contains("xoxb-")
        || value.contains("xoxp-")
        || value.to_ascii_lowercase().contains("bearer ")
        || value.contains("BEGIN PRIVATE KEY")
        || value.contains("BEGIN OPENSSH PRIVATE KEY")
        || decoded_value_is_forbidden(value)
}

fn contains_credential_assignment(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    [
        "api_key",
        "api key",
        "access_token",
        "access token",
        "client_secret",
        "client secret",
        "password",
        "secret_key",
        "secret key",
    ]
    .iter()
    .any(|key| {
        lower.match_indices(key).any(|(index, _)| {
            let tail = lower[index + key.len()..].trim_start();
            tail.starts_with('=')
                || tail.starts_with(':')
                || tail.starts_with("is ")
                || tail.starts_with("is:")
        })
    })
}

fn decoded_value_is_forbidden(value: &str) -> bool {
    let candidate = value.trim();
    if candidate.len() < 12 || candidate.len() > 16_384 {
        return false;
    }
    [
        &base64::engine::general_purpose::STANDARD,
        &base64::engine::general_purpose::URL_SAFE,
        &base64::engine::general_purpose::STANDARD_NO_PAD,
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
    ]
    .into_iter()
    .filter_map(|engine| engine.decode(candidate).ok())
    .filter_map(|decoded| String::from_utf8(decoded).ok())
    .any(|decoded| {
        let (normalized, compact) = normalize_for_leakage_scan(&decoded);
        [
            "benchmark_gold",
            "gold_label",
            "hidden_label",
            "expected_answer",
            "expected_output",
            "reference_answer",
        ]
        .iter()
        .any(|term| {
            normalized.contains(term)
                || compact.contains(
                    &term
                        .chars()
                        .filter(|character| *character != '_')
                        .collect::<String>(),
                )
        }) || contains_credential_assignment(&decoded)
            || decoded.contains("sk-")
            || decoded.contains("ghp_")
            || decoded.contains("github_pat_")
            || decoded.to_ascii_lowercase().contains("bearer ")
            || decoded.contains("PRIVATE KEY")
    })
}

fn validate_probability(value: Option<f64>, field: &str) -> Result<(), ContractError> {
    if let Some(value) = value
        && (!value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(context_error(format!(
            "{field} must be finite and between 0 and 1"
        )));
    }
    Ok(())
}

fn context_serialization_error(error: serde_json::Error) -> ContractError {
    context_error(format!("failed to validate context value: {error}"))
}

fn context_error(message: impl Into<String>) -> ContractError {
    ContractError::InvalidContext(message.into())
}
