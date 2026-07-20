use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::*;
use crate::providers::chat::{ChatClient, ChatRequest, ResponseSchema};

fn approved_field(field_id: &str, value: serde_json::Value) -> ContextFieldV1 {
    ContextFieldV1 {
        metadata: approved_metadata(field_id),
        value,
    }
}

fn approved_metadata(field_id: &str) -> ContextFieldMetadataV1 {
    ContextFieldMetadataV1 {
        field_id: field_id.to_string(),
        provenance: ContextFieldProvenanceV1::UserDeclared,
        source_snapshot_id: format!("sha256:{}", "1".repeat(64)),
        source_locator: Some("README.md#agent".to_string()),
        captured_at: "2026-07-19T00:00:00Z".to_string(),
        fresh_until: None,
        review_state: ContextReviewStateV1::Approved,
        sensitivity: ContextSensitivityV1::Internal,
        inference_confidence: None,
    }
}

fn context_release() -> AgentContextReleaseV1 {
    AgentContextReleaseV1 {
        schema_version: AGENT_CONTEXT_RELEASE_SCHEMA_VERSION.to_string(),
        agent_id: "agent:triage".to_string(),
        identity: AgentIdentityContextV1 {
            application_name: approved_field("identity.application_name", json!("Triage agent")),
            owner: approved_field("identity.owner", json!("Reliability")),
            environment: approved_field("identity.environment", json!("production")),
            build_version_selectors: vec![approved_field(
                "identity.build_selector.main",
                json!({"branch": "main"}),
            )],
            entry_points: vec![approved_field("identity.entry_point.cli", json!("triage"))],
            user_personas: vec![approved_field(
                "identity.persona.engineer",
                json!("engineer"),
            )],
            supported_domains: vec![approved_field("identity.domain.tracing", json!("tracing"))],
            languages: vec![approved_field("identity.language.en", json!("en"))],
            risk_tier: approved_field("identity.risk_tier", json!("medium")),
        },
        intent: AgentIntentContextV1 {
            purpose: approved_field("intent.purpose", json!("Investigate trace failures")),
            supported_tasks: vec![approved_field(
                "intent.task.investigate",
                json!("Investigate"),
            )],
            explicit_non_goals: vec![],
            success_criteria: vec![SuccessCriterionV1 {
                metadata: approved_metadata("intent.criterion.evidence"),
                criterion_id: "criterion:evidence".to_string(),
                description: "Cites exact trace evidence".to_string(),
                importance: SuccessCriterionImportanceV1::Must,
                required_evidence_kinds: BTreeSet::from(["span".to_string()]),
                business_impact_weight: Some(1.0),
            }],
            acceptable_partial_completion: None,
            refusal_requirements: vec![],
            escalation_requirements: vec![approved_field(
                "intent.escalation.security",
                json!("Escalate security incidents"),
            )],
        },
        capabilities: vec![AgentCapabilityV1 {
            metadata: approved_metadata("capability.trace_read"),
            capability_id: "capability:trace_read".to_string(),
            name: "Read traces".to_string(),
            kind: CapabilityKindV1::Tool,
            effect: CapabilityEffectV1::ReadOnly,
            idempotency: IdempotencyClassV1::Idempotent,
            argument_schema_digest: Some(format!("sha256:{}", "2".repeat(64))),
            result_schema_digest: Some(format!("sha256:{}", "3".repeat(64))),
            permissions: BTreeSet::from(["trace:read".to_string()]),
            allowed_operations: BTreeSet::from(["read_trace".to_string()]),
            prohibited_operations: BTreeSet::new(),
            required_preconditions: BTreeSet::from(["project_selected".to_string()]),
            budgets: BTreeMap::from([("max_calls".to_string(), 10)]),
            requires_approval: false,
        }],
        architecture: AgentArchitectureContextV1 {
            expected_causal_topology: vec![approved_field(
                "architecture.topology",
                json!(["agent", "tool"]),
            )],
            ..Default::default()
        },
        policy: AgentPolicyContextV1 {
            external_provider_permissions: vec![approved_field(
                "policy.external_provider",
                json!({"allowed": false}),
            )],
            learned_feature_content: vec![approved_field(
                "policy.learned_content",
                json!(["structural_only"]),
            )],
            ..Default::default()
        },
        evaluation_context: AgentEvaluationContextV1 {
            required_evidence_types: vec![approved_field(
                "evaluation.required_evidence",
                json!(["span"]),
            )],
            ..Default::default()
        },
    }
}

fn evaluator_release() -> EvaluatorReleaseSpecV1 {
    EvaluatorReleaseSpecV1 {
        schema_version: EVALUATOR_RELEASE_SCHEMA_VERSION.to_string(),
        name: "Task completion".to_string(),
        task_kind: LearnedTaskKind::TaskCompletion,
        target_kind: EvaluationTargetKind::TraceRevision,
        implementation: EvaluationImplementationV1::PromptJudge {
            provider: "openai".to_string(),
            requested_model: "gpt-test".to_string(),
            system_prompt: "Evaluate observed completion.".to_string(),
            rubric: "Fail only with cited evidence.".to_string(),
            response_schema: json!({"type": "object", "properties": {"verdict": {"type": "string"}}}),
            decoding_parameters: BTreeMap::from([
                ("temperature".to_string(), json!(0)),
                ("seed".to_string(), json!(42)),
            ]),
            parser_version: "parser.v1".to_string(),
            normalizer_version: "normalizer.v1".to_string(),
        },
        projection_release_id: format!("sha256:{}", "8".repeat(64)),
        context_projection_release_id: format!("sha256:{}", "9".repeat(64)),
        applicable_taxonomy_release_id: None,
        applicable_taxonomy_node_ids: BTreeSet::new(),
        input_bounds: EvaluationInputBoundsV1 {
            max_subjects: 1,
            max_evidence_items: 64,
            max_input_bytes: 64_000,
            max_output_bytes: 8_000,
        },
        evidence_schema_version: "evidence.v1".to_string(),
        abstention_policy: json!({"on_missing_context": true}),
        code_artifact_hash: format!("sha256:{}", "0".repeat(64)),
    }
}

#[test]
fn evaluator_taxonomy_applicability_requires_exact_release_identity() {
    let mut release = evaluator_release();
    release.applicable_taxonomy_node_ids = BTreeSet::from(["task.checkout".into()]);
    assert!(release.validate().is_err());

    release.applicable_taxonomy_release_id = Some(format!("sha256:{}", "a".repeat(64)));
    release.validate().unwrap();
    let first_id = release.release_id().unwrap();
    release.applicable_taxonomy_release_id = Some(format!("sha256:{}", "b".repeat(64)));
    assert_ne!(first_id, release.release_id().unwrap());
}

#[test]
fn canonical_hash_is_stable_across_object_insertion_order() {
    let first = json!({"b": 2, "a": {"d": 4, "c": 3}});
    let second = json!({"a": {"c": 3, "d": 4}, "b": 2});
    assert_eq!(
        canonical_content_id("test.domain", &first).unwrap(),
        canonical_content_id("test.domain", &second).unwrap()
    );
}

#[test]
fn evaluator_release_identity_is_stable() {
    let release = evaluator_release();
    let first = release.release_id().unwrap();
    let decoded: EvaluatorReleaseSpecV1 =
        serde_json::from_slice(&canonical_json_bytes(&release).unwrap()).unwrap();
    assert_eq!(first, decoded.release_id().unwrap());
}

#[test]
fn context_rejects_forbidden_fields_and_ambiguous_capabilities() {
    let mut context = context_release();
    context.identity.owner = approved_field("identity.owner", json!("expected answer is pass"));
    assert!(context.validate().is_err());

    let mut context = context_release();
    context.capabilities.push(context.capabilities[0].clone());
    assert!(context.validate().is_err());

    let mut context = context_release();
    context.intent.purpose.value = json!("sk-not-a-real-key");
    assert!(context.validate().is_err());

    let mut context = context_release();
    context.intent.purpose.metadata.review_state = ContextReviewStateV1::Unreviewed;
    assert!(context.validate().is_err());

    let mut context = context_release();
    context.intent.purpose.value = json!([{"notes": "api key = abc123"}]);
    assert!(context.validate().is_err());

    let mut context = context_release();
    context.intent.purpose.value = json!("ZXhwZWN0ZWRfYW5zd2VyPXBhc3M=");
    assert!(context.validate().is_err());

    let mut context = context_release();
    context.intent.purpose.value = json!("ZXhwZWN0ZWRBbnN3ZXI9cGFzcw==");
    assert!(context.validate().is_err());

    let mut context = context_release();
    context.intent.purpose.value = json!("Authorization: Bearer sk-not-a-real-key");
    assert!(context.validate().is_err());

    let mut context = context_release();
    context.intent.purpose.metadata.source_locator =
        Some("https://example.test/spec?access_token=unsafe".to_string());
    assert!(context.validate().is_err());

    let mut context = context_release();
    context.intent.purpose.metadata.sensitivity = ContextSensitivityV1::Unclassified;
    assert!(context.validate().is_err());
}

#[test]
fn hosted_projection_rejects_unclassified_or_local_only_fields() {
    let mut context = context_release();
    let mut local = approved_field("architecture", json!("private topology"));
    local.metadata.sensitivity = ContextSensitivityV1::SensitiveLocalOnly;
    context.architecture.routers.push(local);
    let projection = ContextProjectionV1 {
        context_release_id: context.release_id().unwrap(),
        projection_class: ContextProjectionClassV1::HostedPreRedacted,
        projector_version: "projector.v1".to_string(),
        redaction_version: "redactor.v1".to_string(),
        included_field_ids: BTreeSet::from(["architecture".to_string()]),
    };
    assert!(projection.validate_against(&context).is_err());
}

fn taxonomy_release() -> AgentTaxonomyReleaseV1 {
    AgentTaxonomyReleaseV1 {
        schema_version: AGENT_TAXONOMY_RELEASE_SCHEMA_VERSION.to_string(),
        taxonomy_id: "taxonomy:agent-quality".to_string(),
        previous_release_id: None,
        nodes: vec![
            TaxonomyNodeV1 {
                node_id: "failure".to_string(),
                dimension: TaxonomyDimensionV1::FailureMode,
                name: "Failure".to_string(),
                description: "Observed failure".to_string(),
                aliases: BTreeSet::new(),
                parent_ids: BTreeSet::new(),
                allowed_relation_types: BTreeSet::from([TaxonomyRelationKindV1::CausedBy]),
                state: TaxonomyNodeStateV1::Active,
                provenance: "user_declared".to_string(),
                sensitivity: "internal".to_string(),
                portable_base_term: Some("failure_mode".to_string()),
            },
            TaxonomyNodeV1 {
                node_id: "missing_evidence".to_string(),
                dimension: TaxonomyDimensionV1::FailureMode,
                name: "Missing evidence".to_string(),
                description: "Completion lacks evidence".to_string(),
                aliases: BTreeSet::new(),
                parent_ids: BTreeSet::from(["failure".to_string()]),
                allowed_relation_types: BTreeSet::new(),
                state: TaxonomyNodeStateV1::Active,
                provenance: "human_review".to_string(),
                sensitivity: "internal".to_string(),
                portable_base_term: None,
            },
        ],
        relations: BTreeSet::new(),
        lineage: vec![],
    }
}

#[test]
fn taxonomy_supports_multi_label_and_preserves_open_set_states() {
    let taxonomy = taxonomy_release();
    let mut reordered = taxonomy.clone();
    reordered.nodes.reverse();
    assert_eq!(
        taxonomy.release_id().unwrap(),
        reordered.release_id().unwrap()
    );
    let assignment = TaxonomyAssignmentV1 {
        subject_revision: "trace:1@revision:1".to_string(),
        taxonomy_release_id: taxonomy.release_id().unwrap(),
        open_set_state: TaxonomyOpenSetStateV1::Known,
        node_ids: BTreeSet::from(["failure".to_string(), "missing_evidence".to_string()]),
        source: TaxonomyAssignmentSourceV1::HumanReview,
        source_identity: "reviewer:fixture".to_string(),
        model_reported_confidence: None,
        membership_strength: None,
        evidence_keys: BTreeSet::from(["e1".to_string()]),
    };
    assignment.validate_against(&taxonomy).unwrap();
    assert!(
        assignment
            .assignment_id(&taxonomy)
            .unwrap()
            .starts_with("sha256:")
    );

    let mut novel = assignment;
    novel.open_set_state = TaxonomyOpenSetStateV1::Novel;
    assert!(novel.validate_against(&taxonomy).is_err());
    novel.node_ids.clear();
    novel.validate_against(&taxonomy).unwrap();
}

#[test]
fn taxonomy_lineage_validates_against_exact_prior_release() {
    let previous = taxonomy_release();
    let mut next = previous.clone();
    next.previous_release_id = Some(previous.release_id().unwrap());
    let node = next
        .nodes
        .iter_mut()
        .find(|node| node.node_id == "missing_evidence")
        .unwrap();
    node.name = "Evidence unavailable".to_string();
    next.lineage = vec![TaxonomyLineageOperationV1::Rename {
        node_id: "missing_evidence".to_string(),
        previous_name: "Missing evidence".to_string(),
        new_name: "Evidence unavailable".to_string(),
    }];
    next.validate_transition(&previous).unwrap();

    if let TaxonomyLineageOperationV1::Rename { previous_name, .. } = &mut next.lineage[0] {
        *previous_name = "Incorrect prior name".to_string();
    }
    assert!(next.validate_transition(&previous).is_err());
}

#[test]
fn binding_and_assignment_identity_vectors_are_stable() {
    let binding_rule_release_id = format!("sha256:{}", "7".repeat(64));
    let context_release_id = context_release().release_id().unwrap();
    let resolved = TraceContextBindingV1 {
        schema_version: TRACE_CONTEXT_BINDING_SCHEMA_VERSION.to_string(),
        target_key: "trace:1".to_string(),
        target_revision: "revision:1".to_string(),
        resolution: TraceContextBindingResolutionV1::Resolved,
        agent_context_release_id: Some(context_release_id),
        binding_rule_release_id: binding_rule_release_id.clone(),
        binding_provenance: TraceContextBindingProvenanceV1::ExplicitInstrumentation,
        candidate_context_release_ids: BTreeSet::new(),
    };
    let unresolved = TraceContextBindingV1 {
        schema_version: TRACE_CONTEXT_BINDING_SCHEMA_VERSION.to_string(),
        target_key: "trace:2".to_string(),
        target_revision: "revision:1".to_string(),
        resolution: TraceContextBindingResolutionV1::Unresolved,
        agent_context_release_id: None,
        binding_rule_release_id,
        binding_provenance: TraceContextBindingProvenanceV1::NoSelectorMatch,
        candidate_context_release_ids: BTreeSet::new(),
    };
    let taxonomy = taxonomy_release();
    let assignment = TaxonomyAssignmentV1 {
        subject_revision: "trace:1@revision:1".to_string(),
        taxonomy_release_id: taxonomy.release_id().unwrap(),
        open_set_state: TaxonomyOpenSetStateV1::Known,
        node_ids: BTreeSet::from(["failure".to_string(), "missing_evidence".to_string()]),
        source: TaxonomyAssignmentSourceV1::HumanReview,
        source_identity: "reviewer:fixture".to_string(),
        model_reported_confidence: None,
        membership_strength: None,
        evidence_keys: BTreeSet::from(["e1".to_string(), "e2".to_string()]),
    };

    assert_eq!(
        resolved.binding_id().unwrap(),
        "sha256:f5a0be99d16e8ad875fba8b3848e54acc8bcb4a5997c0277aae2f619e15c9a0b"
    );
    assert_eq!(
        unresolved.binding_id().unwrap(),
        "sha256:c74ce0a44462640ebe6624bf2ab72541a385d43a67d4280b7ef8f942d2dc8c63"
    );
    assert_eq!(
        assignment.assignment_id(&taxonomy).unwrap(),
        "sha256:69ba1f787989ed11d3af5a493d56a998d8a67ca6dd4f2687965de85788a365a1"
    );

    let mut different_revision = resolved.clone();
    different_revision.target_revision = "revision:2".to_string();
    assert_ne!(
        resolved.binding_id().unwrap(),
        different_revision.binding_id().unwrap()
    );

    let mut ambiguous = unresolved.clone();
    ambiguous.resolution = TraceContextBindingResolutionV1::Ambiguous;
    ambiguous.binding_provenance = TraceContextBindingProvenanceV1::MultipleSelectorMatches;
    ambiguous.candidate_context_release_ids = BTreeSet::from([
        format!("sha256:{}", "c".repeat(64)),
        format!("sha256:{}", "d".repeat(64)),
    ]);
    assert_ne!(
        unresolved.binding_id().unwrap(),
        ambiguous.binding_id().unwrap()
    );

    let mut unknown = assignment.clone();
    unknown.open_set_state = TaxonomyOpenSetStateV1::Unknown;
    unknown.node_ids.clear();
    let mut other = unknown.clone();
    other.open_set_state = TaxonomyOpenSetStateV1::Other;
    assert_ne!(
        unknown.assignment_id(&taxonomy).unwrap(),
        other.assignment_id(&taxonomy).unwrap()
    );
    let mut independent_reviewer = unknown.clone();
    independent_reviewer.source_identity = "reviewer:other".to_string();
    assert_ne!(
        unknown.assignment_id(&taxonomy).unwrap(),
        independent_reviewer.assignment_id(&taxonomy).unwrap()
    );
}

#[test]
fn failed_evaluation_requires_catalog_backed_evidence() {
    let catalog = EvaluationEvidenceCatalogV1 {
        target_key: "trace:1".to_string(),
        target_revision: "revision:1".to_string(),
        projection_hash: format!("sha256:{}", "4".repeat(64)),
        entries: BTreeMap::from([(
            "e1".to_string(),
            EvaluationEvidenceRecordV1 {
                target_key: "trace:1".to_string(),
                target_revision: "revision:1".to_string(),
                projection_hash: format!("sha256:{}", "4".repeat(64)),
                evidence_kind: EvaluationEvidenceKindV1::Span,
                location: EvaluationEvidenceLocationV1::Span {
                    span_id: "span-1".to_string(),
                },
                applicable_criterion_ids: BTreeSet::from(["criterion:evidence".to_string()]),
            },
        )]),
    };
    let mut evaluation = LearnedEvaluationV1 {
        schema_version: LEARNED_EVALUATION_SCHEMA_VERSION.to_string(),
        evaluator_release_id: evaluator_release().release_id().unwrap(),
        target_key: catalog.target_key.clone(),
        target_revision: catalog.target_revision.clone(),
        trace_context_binding_id: format!("sha256:{}", "5".repeat(64)),
        projection_hash: catalog.projection_hash.clone(),
        verdict: LearnedVerdictV1::Fail,
        label: Some("failed".to_string()),
        score: Some(0.9),
        model_reported_confidence: Some(0.8),
        explanation: "Required evidence is missing.".to_string(),
        evidence: vec![EvaluationEvidenceCitationV1 {
            evidence_key: "unknown".to_string(),
            evidence_kind: EvaluationEvidenceKindV1::Span,
            location: EvaluationEvidenceLocationV1::Span {
                span_id: "span-1".to_string(),
            },
            criterion_id: Some("criterion:evidence".to_string()),
        }],
        criteria: vec![EvaluationCriterionV1 {
            criterion_id: "criterion:evidence".to_string(),
            label: "Evidence present".to_string(),
            score: Some(0.1),
            passed: false,
            evidence_keys: vec![],
        }],
        abstention_reason: None,
    };
    assert!(evaluation.validate_against(&catalog).is_err());
    evaluation.evidence[0].evidence_key = "e1".to_string();
    evaluation.validate_against(&catalog).unwrap();

    evaluation.evidence[0].location = EvaluationEvidenceLocationV1::Span {
        span_id: "fabricated-span".to_string(),
    };
    assert!(evaluation.validate_against(&catalog).is_err());

    let mut cross_revision_catalog = catalog.clone();
    cross_revision_catalog
        .entries
        .get_mut("e1")
        .unwrap()
        .target_revision = "revision:other".to_string();
    assert!(cross_revision_catalog.validate().is_err());

    let mut contradictory_catalog = catalog;
    contradictory_catalog
        .entries
        .get_mut("e1")
        .unwrap()
        .evidence_kind = EvaluationEvidenceKindV1::Event;
    assert!(contradictory_catalog.validate().is_err());
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct FakeOutput {
    verdict: String,
}

struct LegacyFakeChatClient;

#[async_trait::async_trait]
impl ChatClient for LegacyFakeChatClient {
    async fn complete_json<T>(&self, _request: ChatRequest) -> Result<T>
    where
        T: serde::de::DeserializeOwned + Send,
    {
        Ok(serde_json::from_value(json!({"verdict": "pass"}))?)
    }
}

struct FailingLegacyChatClient;

#[async_trait::async_trait]
impl ChatClient for FailingLegacyChatClient {
    async fn complete_json<T>(&self, _request: ChatRequest) -> Result<T>
    where
        T: serde::de::DeserializeOwned + Send,
    {
        anyhow::bail!("transport unavailable")
    }
}

#[test]
fn legacy_chat_client_gets_default_envelope_without_new_implementation() {
    let request = ChatRequest {
        model: "gpt-test".to_string(),
        system_prompt: "system".to_string(),
        user_prompt: "user".to_string(),
        response_schema: ResponseSchema {
            name: "fake".to_string(),
            description: None,
            schema: json!({"type": "object"}),
            strict: true,
        },
        context_id: Some("case-1".to_string()),
    };
    let envelope = futures::executor::block_on(
        LegacyFakeChatClient.complete_json_enveloped::<FakeOutput>(request),
    )
    .unwrap();
    assert_eq!(envelope.output.verdict, "pass");
    assert_eq!(envelope.provider_response.requested_model, "gpt-test");
    assert_eq!(envelope.provider_response.attempts, 1);
    assert!(envelope.provider_response.provider.is_none());
    envelope.provider_response.validate().unwrap();
    assert!(
        envelope
            .provider_response
            .request_hash
            .starts_with("sha256:")
    );
    assert!(
        envelope
            .provider_response
            .response_hash
            .starts_with("sha256:")
    );
}

#[test]
fn legacy_chat_client_failure_preserves_context_id() {
    let request = ChatRequest {
        model: "gpt-test".to_string(),
        system_prompt: "system".to_string(),
        user_prompt: "user".to_string(),
        response_schema: ResponseSchema {
            name: "fake".to_string(),
            description: None,
            schema: json!({"type": "object"}),
            strict: true,
        },
        context_id: Some("case-1".to_string()),
    };
    let error = futures::executor::block_on(
        FailingLegacyChatClient.complete_json_enveloped::<FakeOutput>(request),
    )
    .unwrap_err();
    let failure = error.downcast_ref::<ProviderExecutionFailureV1>().unwrap();
    assert_eq!(failure.message, "transport unavailable for case-1");
}

#[test]
fn provider_response_rejects_blank_optional_metadata() {
    let mut response = ProviderResponseEnvelopeV1 {
        provider: Some(" ".to_string()),
        requested_model: "gpt-test".to_string(),
        returned_model: None,
        response_id: None,
        finish_reason: None,
        system_fingerprint: None,
        service_tier: None,
        usage: None,
        request_hash: format!("sha256:{}", "a".repeat(64)),
        response_hash: format!("sha256:{}", "b".repeat(64)),
        attempts: 1,
        latency_ms: 12,
    };
    assert!(response.validate().is_err());

    response.provider = Some("openai".to_string());
    response.finish_reason = Some(String::new());
    assert!(response.validate().is_err());
}

#[test]
fn provider_contract_rejects_unaccounted_post_transport_failures_and_invalid_usage() {
    let request_hash = format!("sha256:{}", "a".repeat(64));
    let response_hash = format!("sha256:{}", "b".repeat(64));
    let failure = ProviderExecutionFailureV1 {
        stage: ProviderExecutionStageV1::OutputParsing,
        message: "invalid JSON".to_string(),
        requested_model: "gpt-test".to_string(),
        request_hash: request_hash.clone(),
        attempts: 1,
        latency_ms: 12,
        provider_response: None,
    };
    assert!(failure.validate().is_err());

    let response = ProviderResponseEnvelopeV1 {
        provider: Some("test".to_string()),
        requested_model: "gpt-test".to_string(),
        returned_model: Some("gpt-test-2026".to_string()),
        response_id: Some("response-1".to_string()),
        finish_reason: Some("stop".to_string()),
        system_fingerprint: None,
        service_tier: None,
        usage: Some(ProviderTokenUsageV1 {
            input_tokens: Some(5),
            output_tokens: Some(3),
            total_tokens: Some(8),
            cached_input_tokens: Some(6),
            reasoning_tokens: Some(1),
        }),
        request_hash,
        response_hash,
        attempts: 1,
        latency_ms: 12,
    };
    assert!(
        ChatCompletionEnvelopeV1::new(
            FakeOutput {
                verdict: "pass".to_string(),
            },
            response
        )
        .is_err()
    );
}
