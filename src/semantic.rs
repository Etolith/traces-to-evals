//! Shared semantic vocabulary for trace ingestion and analysis projection.

use crate::model::{Span, SpanKind};

pub const OPENINFERENCE_SPAN_KIND_ATTRIBUTE: &str = "openinference.span.kind";

const SEMANTIC_SPAN_KINDS: &[(&str, SpanKind)] = &[
    ("llm", SpanKind::Llm),
    ("agent", SpanKind::Agent),
    ("tool", SpanKind::Tool),
    ("chain", SpanKind::Chain),
    ("retrieval", SpanKind::Retriever),
    ("retriever", SpanKind::Retriever),
    ("reranker", SpanKind::Reranker),
    ("embedding", SpanKind::Embedding),
    ("guardrail", SpanKind::Guardrail),
    ("evaluator", SpanKind::Evaluator),
    ("prompt", SpanKind::Prompt),
];

/// Maps OpenInference values and persisted category aliases onto the portable
/// trace span kind.
pub fn semantic_span_kind(value: &str) -> SpanKind {
    let value = value.trim();
    SEMANTIC_SPAN_KINDS
        .iter()
        .find_map(|(candidate, kind)| value.eq_ignore_ascii_case(candidate).then_some(*kind))
        .unwrap_or(SpanKind::Other)
}

pub fn resolved_span_kind(span: &Span) -> SpanKind {
    if span.kind != SpanKind::Other {
        return span.kind;
    }
    span.attributes
        .get(OPENINFERENCE_SPAN_KIND_ATTRIBUTE)
        .and_then(|value| value.as_str())
        .map(semantic_span_kind)
        .unwrap_or(SpanKind::Other)
}

/// Semantic keys understood by the reusable behavior projection. Product
/// privacy policy must still decide whether a known key and its value may be
/// released to analysis.
pub fn is_known_semantic_attribute_key(key: &str) -> bool {
    key.eq_ignore_ascii_case(OPENINFERENCE_SPAN_KIND_ATTRIBUTE)
        || matches!(
            key.to_ascii_lowercase().as_str(),
            "gen_ai.operation.name"
                | "gen_ai.tool.name"
                | "gen_ai.tool.call.id"
                | "gen_ai.tool.status"
                | "agent.operation"
                | "agent.operation.effect"
                | "agent.operation.retry_safety"
                | "agent.tool.requirement"
                | "agent.tool.attempt"
                | "agent.tool.status"
                | "agent.approval.required"
                | "agent.approval.outcome"
                | "agent.state.observation"
                | "agent.state.predicate"
                | "agent.state.artifact.id"
                | "agent.final.status"
                | "final.status"
                | "agent.escalation.status"
                | "final.escalation.status"
                | "agent.outcome.claim.status"
                | "agent.outcome.claim.operation"
                | "agent.outcome.claim.call_id"
                | "final.outcome.claim.status"
                | "final.outcome.claim.operation"
                | "final.outcome.claim.call_id"
                | "agent.role"
                | "agent.policy.id"
                | "agent.policy.action"
                | "agent.policy.outcome"
                | "agent.policy.reason_code"
                | "tool.name"
                | "tool.call.id"
                | "tool_call_id"
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
                | "tool.state.predicate"
                | "tool.state.artifact.id"
                | "operation"
                | "operation.name"
                | "operation.effect"
                | "operation.retry_safety"
                | "operation.requirement"
                | "execution.status"
                | "execution.timeout"
                | "duration_ms"
                | "tool.duration_ms"
                | "gen_ai.tool.duration_ms"
                | "gen_ai.execute_tool.duration"
                | "tool.duration"
                | "execution.duration"
                | "error.type"
                | "error.code"
                | "error.retryable"
                | "tool.error.kind"
                | "tool.error.code"
                | "tool.error.retryable"
                | "exception.type"
                | "exception.escaped"
                | "exception.recorded"
                | "http.status_code"
                | "http.response.status_code"
                | "rpc.status_code"
                | "protocol.status_code"
                | "result.success"
                | "result.ok"
                | "policy.id"
                | "policy.version"
                | "policy.decision.id"
                | "policy.decision.outcome"
                | "policy.action"
                | "policy.outcome"
                | "policy.reason_code"
                | "guardrail.outcome"
                | "decision_id"
                | "reason_code"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_openinference_and_persisted_retrieval_aliases() {
        assert_eq!(semantic_span_kind("RETRIEVER"), SpanKind::Retriever);
        assert_eq!(semantic_span_kind("retrieval"), SpanKind::Retriever);
        assert_eq!(semantic_span_kind("RERANKER"), SpanKind::Reranker);
    }

    #[test]
    fn semantic_registry_excludes_payload_content() {
        assert!(is_known_semantic_attribute_key("tool.result.success"));
        assert!(!is_known_semantic_attribute_key("input.value"));
        assert!(!is_known_semantic_attribute_key("output.value"));
    }
}
