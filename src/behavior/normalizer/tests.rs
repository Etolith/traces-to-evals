use std::collections::BTreeMap;

use serde_json::json;

use super::*;
use crate::DeterministicDetectorSet;
use crate::model::{PayloadIdentity, SpanEvent, SpanLink};

fn span_with_attributes(
    id: &str,
    name: &str,
    kind: SpanKind,
    attributes: BTreeMap<String, Value>,
) -> Span {
    Span {
        id: id.to_string(),
        trace_id: Some("trace-1".to_string()),
        parent_id: Some("root".to_string()),
        name: name.to_string(),
        kind,
        input: None,
        output: None,
        error: None,
        started_at: Some("2026-07-10T12:00:00Z".to_string()),
        ended_at: Some("2026-07-10T12:00:01Z".to_string()),
        attributes,
        ..Span::new("", "")
    }
}

#[test]
fn normalizes_structured_tool_failure_before_success_evidence() {
    let mut attributes = BTreeMap::new();
    attributes.insert("gen_ai.tool.name".to_string(), json!("cancel_card"));
    attributes.insert("agent.operation".to_string(), json!("cancel_card"));
    attributes.insert("agent.operation.effect".to_string(), json!("mutating"));
    attributes.insert(
        "agent.operation.retry_safety".to_string(),
        json!("non_idempotent"),
    );
    attributes.insert("agent.tool.requirement".to_string(), json!("required"));
    attributes.insert("tool.result.success".to_string(), json!(false));
    attributes.insert("state_delta".to_string(), json!(r#"{"changed":true}"#));
    let tool = span_with_attributes("tool-1", "cancel", SpanKind::Tool, attributes);
    let trace = Trace::new("trace-1").with_span(tool);

    let behavior = OpenInferenceBehaviorNormalizer::default()
        .normalize(&trace)
        .unwrap();

    assert_eq!(behavior.tool_calls[0].status, ToolCallStatus::Failed);
    assert_eq!(
        behavior.tool_calls[0].operation.as_deref(),
        Some("cancel_card")
    );
    assert_eq!(behavior.tool_calls[0].effect, OperationEffect::Mutating);
    assert_eq!(
        behavior.tool_calls[0].retry_safety,
        RetrySafety::NonIdempotent
    );
    assert_eq!(
        behavior.tool_calls[0].requirement,
        ToolRequirement::Required
    );
    assert_eq!(
        behavior.tool_calls[0]
            .state_change
            .as_ref()
            .unwrap()
            .observation,
        StateObservation::Unverified
    );
}

#[test]
fn does_not_copy_error_message_into_normalized_error() {
    let mut tool = Span::new("tool-1", "lookup").with_kind(SpanKind::Tool);
    tool.error = Some("customer secret caused network error".to_string());
    let trace = Trace::new("trace-1").with_span(tool);

    let behavior = OpenInferenceBehaviorNormalizer::default()
        .normalize(&trace)
        .unwrap();
    let error = behavior.tool_calls[0].error.as_ref().unwrap();

    assert_eq!(error.kind, "network");
    assert!(
        error
            .redacted_message_hash
            .as_deref()
            .unwrap()
            .starts_with("sha256:")
    );
    assert!(
        !serde_json::to_string(error)
            .unwrap()
            .contains("customer secret")
    );
}

#[test]
fn does_not_infer_safety_requiredness_or_claims_from_names_and_prose() {
    let root = Span {
        id: "root".to_string(),
        trace_id: Some("trace-1".to_string()),
        parent_id: None,
        name: "agent".to_string(),
        kind: SpanKind::Agent,
        input: Some("cancel it".to_string()),
        output: Some("Your card has been cancelled.".to_string()),
        error: None,
        started_at: None,
        ended_at: None,
        attributes: BTreeMap::new(),
        ..Span::new("", "")
    };
    let mut tool = Span::new("tool-1", "cancelCard").with_kind(SpanKind::Tool);
    tool.parent_id = Some("root".to_string());
    tool.attributes
        .insert("state_delta".to_string(), json!({"changed": true}));
    let trace = Trace::new("trace-1").with_span(root).with_span(tool);

    let behavior = OpenInferenceBehaviorNormalizer::default()
        .normalize(&trace)
        .unwrap();
    let call = &behavior.tool_calls[0];

    assert_eq!(call.operation, None);
    assert_eq!(call.effect, OperationEffect::Unknown);
    assert_eq!(call.retry_safety, RetrySafety::Unknown);
    assert_eq!(call.requirement, ToolRequirement::Unknown);
    assert_eq!(call.status, ToolCallStatus::Unknown);
    assert!(behavior.final_outcome.claims.is_empty());
    assert_eq!(behavior.final_outcome.status, FinalOutcomeStatus::Unknown);
}

#[test]
fn applies_versioned_adapter_mapping_without_tool_name_heuristics() {
    let adapter = BehaviorAdapterConfig::new("fintech_support", "7").with_tool_mapping(
        "cancelCard",
        ToolSemanticMapping::new("cancel_card")
            .with_effect(OperationEffect::Mutating)
            .with_retry_safety(RetrySafety::NonIdempotent)
            .with_requirement(ToolRequirement::Required),
    );
    let mut tool = Span::new("tool-1", "tool").with_kind(SpanKind::Tool);
    tool.attributes
        .insert("gen_ai.tool.name".to_string(), json!("cancelCard"));
    tool.attributes
        .insert("agent.tool.status".to_string(), json!("succeeded"));
    let trace = Trace::new("trace-1").with_span(tool);

    let behavior = OpenInferenceBehaviorNormalizer::from_adapter(adapter)
        .unwrap()
        .normalize(&trace)
        .unwrap();

    assert_eq!(
        behavior.tool_calls[0].operation.as_deref(),
        Some("cancel_card")
    );
    assert_eq!(behavior.metadata["traceeval.behavior_adapter.version"], "7");
}

#[test]
fn safe_input_preserves_status_fingerprint_time_events_and_links() {
    let mut tool = Span::new("tool-1", "browser.search").with_kind(SpanKind::Tool);
    tool.source_status = SourceSpanStatus::Ok;
    tool.start_time_unix_nano = Some(1_000_000_000);
    tool.end_time_unix_nano = Some(1_001_500_000);
    tool.duration_nano = Some(1_500_000);
    tool.attributes
        .insert("gen_ai.operation.name".into(), json!("search"));
    tool.payload_identities.insert(
        "gen_ai.tool.call.arguments".into(),
        PayloadIdentity {
            fingerprint: format!("sha256:{:064x}", 7),
            blob_id: Some("sha256:opaque-blob".into()),
            original_bytes: 91,
            quality: FactQuality::Explicit,
        },
    );
    tool.events.push(SpanEvent {
        identity: "event:tool-1:0".into(),
        name: "progress".into(),
        timestamp_unix_nano: 1_001_000_000,
        attributes: BTreeMap::new(),
    });
    tool.links.push(SpanLink {
        identity: "link:tool-1:0".into(),
        trace_id: "linked-trace".into(),
        span_id: "linked-span".into(),
        trace_state: String::new(),
        attributes: BTreeMap::new(),
    });
    let input = BehaviorInputV1::safe(
        Trace::new("trace-1").with_span(tool),
        super::super::input::BehaviorInputProvenanceV1 {
            projection_version: crate::behavior::SAFE_BEHAVIOR_PROJECTION_VERSION.into(),
            source_id: "source-1".into(),
            decoder_versions: ["otlp.v1".to_string()].into_iter().collect(),
            semantic_mapping_versions: ["otel-genai.v1".to_string()].into_iter().collect(),
        },
    )
    .unwrap();

    let behavior = OpenInferenceBehaviorNormalizer::default()
        .normalize_input(&input)
        .unwrap();
    let call = &behavior.tool_calls[0];

    assert_eq!(call.status, ToolCallStatus::Succeeded);
    assert_eq!(call.status_quality, FactQuality::Explicit);
    assert_eq!(call.operation.as_deref(), Some("search"));
    assert_eq!(call.operation_source_quality, FactQuality::Explicit);
    assert_eq!(
        call.invocation_fingerprint.as_deref(),
        Some(format!("sha256:{:064x}", 7).as_str())
    );
    assert_eq!(call.duration_nano, Some(1_500_000));
    assert_eq!(call.duration_ms, 2);
    assert!(
        call.evidence
            .iter()
            .any(|evidence| evidence.identity == "event:tool-1:0")
    );
    assert!(
        call.evidence
            .iter()
            .any(|evidence| evidence.identity == "link:tool-1:0")
    );
}

#[test]
fn explicit_error_status_normalizes_as_failure() {
    let mut tool = Span::new("tool-1", "browser.search").with_kind(SpanKind::Tool);
    tool.source_status = SourceSpanStatus::Error;
    let behavior = OpenInferenceBehaviorNormalizer::default()
        .normalize(&Trace::new("trace-1").with_span(tool))
        .unwrap();

    assert_eq!(behavior.tool_calls[0].status, ToolCallStatus::Failed);
    assert_eq!(behavior.tool_calls[0].status_quality, FactQuality::Explicit);
}

#[test]
fn conservative_v2_detector_v6_tolerates_multiple_roots_and_parent_cycles() {
    let mut root_a = Span::new("root-a", "planner").with_kind(SpanKind::Agent);
    root_a
        .attributes
        .insert("agent.final.status".into(), json!("completed"));
    let root_b = Span::new("root-b", "verifier").with_kind(SpanKind::Agent);
    let mut cycle_a = Span::new("cycle-a", "browser-a").with_kind(SpanKind::Tool);
    cycle_a.parent_id = Some("cycle-b".into());
    cycle_a.source_status = SourceSpanStatus::Ok;
    cycle_a
        .attributes
        .insert("gen_ai.tool.name".into(), json!("browser"));
    cycle_a
        .attributes
        .insert("gen_ai.operation.name".into(), json!("open_a"));
    let mut cycle_b = Span::new("cycle-b", "browser-b").with_kind(SpanKind::Tool);
    cycle_b.parent_id = Some("cycle-a".into());
    cycle_b.source_status = SourceSpanStatus::Ok;
    cycle_b
        .attributes
        .insert("gen_ai.tool.name".into(), json!("browser"));
    cycle_b
        .attributes
        .insert("gen_ai.operation.name".into(), json!("open_b"));

    let behavior = OpenInferenceBehaviorNormalizer::default()
        .normalize(
            &Trace::new("trace-topology-adversarial")
                .with_span(root_a)
                .with_span(root_b)
                .with_span(cycle_a)
                .with_span(cycle_b),
        )
        .unwrap();
    let report = DeterministicDetectorSet::default().detect_report(&behavior);

    assert_eq!(behavior.tool_calls.len(), 2);
    assert!(
        behavior
            .tool_calls
            .iter()
            .all(|call| call.status == ToolCallStatus::Succeeded)
    );
    assert!(report.findings.is_empty());
    assert_eq!(report.profile.profile_id, "traceeval.conservative");
    assert_eq!(report.profile.profile_version, "2");
    assert!(
        report
            .detector_versions
            .values()
            .all(|version| version == "6")
    );
}
