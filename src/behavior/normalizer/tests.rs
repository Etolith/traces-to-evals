use std::collections::BTreeMap;

use serde_json::json;

use super::*;

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
