use serde_json::{Map, Value};

use crate::extractors::EvalCaseExtractor;
use crate::model::{EvalCase, Span, SpanKind, Trace};
use crate::{Result, TraceEvalError};

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenInferenceExtractor;

impl OpenInferenceExtractor {
    pub fn extract_trace(&self, trace: &Trace) -> Result<EvalCase> {
        trace_to_eval_case(trace)
    }

    pub fn extract_traces(&self, traces: &[Trace]) -> Result<Vec<EvalCase>> {
        self.extract_cases(traces)
    }

    pub fn span_kind_from_attribute(value: &str) -> SpanKind {
        openinference_span_kind(value)
    }
}

impl EvalCaseExtractor for OpenInferenceExtractor {
    fn extract_case(&self, trace: &Trace) -> Result<EvalCase> {
        self.extract_trace(trace)
    }
}

fn trace_to_eval_case(trace: &Trace) -> Result<EvalCase> {
    let input_span = trace
        .spans
        .iter()
        .find(|span| is_root_task_span(span) && span_input(span).is_some())
        .or_else(|| {
            trace
                .spans
                .iter()
                .find(|span| span_kind(span) == SpanKind::Llm && span_input(span).is_some())
        })
        .or_else(|| trace.spans.iter().find(|span| span_input(span).is_some()))
        .ok_or_else(|| TraceEvalError::MissingTraceInput {
            trace_id: trace.id.clone(),
            extractor: "openinference".to_string(),
        })?;

    let output_span = trace
        .spans
        .iter()
        .rev()
        .find(|span| is_root_task_span(span) && span_output(span).is_some())
        .or_else(|| {
            trace
                .spans
                .iter()
                .rev()
                .find(|span| span_kind(span) == SpanKind::Llm && span_output(span).is_some())
        })
        .or_else(|| {
            trace
                .spans
                .iter()
                .rev()
                .find(|span| span_output(span).is_some())
        });

    let mut metadata = trace.metadata.clone();
    merge_case_context(&mut metadata, input_span);
    if let Some(output_span) = output_span {
        merge_case_context(&mut metadata, output_span);
    }

    let tool_calls = trace
        .spans
        .iter()
        .filter(|span| span_kind(span) == SpanKind::Tool)
        .map(tool_call_context)
        .collect::<Vec<_>>();
    if !tool_calls.is_empty() {
        metadata.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }

    metadata.insert(
        "source_span_id".to_string(),
        Value::String(input_span.id.clone()),
    );
    metadata.insert(
        "extractor".to_string(),
        Value::String("openinference".to_string()),
    );

    if let Some(output_span) = output_span {
        metadata.insert(
            "output_span_id".to_string(),
            Value::String(output_span.id.clone()),
        );
    }

    Ok(EvalCase {
        id: format!("{}:{}", trace.id, input_span.id),
        trace_id: trace.id.clone(),
        input: span_input(input_span).expect("input span checked above"),
        actual_output: output_span.and_then(span_output),
        expected_output: None,
        rubric: None,
        metadata,
    })
}

fn is_root_task_span(span: &Span) -> bool {
    span.parent_id.is_none() && matches!(span_kind(span), SpanKind::Agent | SpanKind::Chain)
}

fn span_input(span: &Span) -> Option<String> {
    span.input
        .clone()
        .or_else(|| string_attribute(span, "input.value"))
}

fn span_output(span: &Span) -> Option<String> {
    span.output
        .clone()
        .or_else(|| string_attribute(span, "output.value"))
}

fn span_kind(span: &Span) -> SpanKind {
    if span.kind != SpanKind::Other {
        return span.kind;
    }

    string_attribute(span, "openinference.span.kind")
        .as_deref()
        .map(openinference_span_kind)
        .unwrap_or(SpanKind::Other)
}

fn string_attribute(span: &Span, key: &str) -> Option<String> {
    match span.attributes.get(key) {
        Some(Value::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn merge_case_context(metadata: &mut std::collections::BTreeMap<String, Value>, span: &Span) {
    for (key, value) in &span.attributes {
        if matches!(key.as_str(), "input.value" | "output.value") {
            continue;
        }
        metadata.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

fn tool_call_context(span: &Span) -> Value {
    let mut context = Map::new();
    context.insert("span_id".to_string(), Value::String(span.id.clone()));
    context.insert(
        "tool_name".to_string(),
        Value::String(
            string_attribute(span, "gen_ai.tool.name")
                .or_else(|| string_attribute(span, "tool_name"))
                .unwrap_or_else(|| span.name.clone()),
        ),
    );
    context.insert(
        "attributes".to_string(),
        Value::Object(
            span.attributes
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ),
    );

    Value::Object(context)
}

fn openinference_span_kind(value: &str) -> SpanKind {
    match value.to_ascii_uppercase().as_str() {
        "LLM" => SpanKind::Llm,
        "AGENT" => SpanKind::Agent,
        "TOOL" => SpanKind::Tool,
        "CHAIN" => SpanKind::Chain,
        "RETRIEVER" => SpanKind::Retriever,
        "RERANKER" => SpanKind::Reranker,
        "EMBEDDING" => SpanKind::Embedding,
        "GUARDRAIL" => SpanKind::Guardrail,
        "EVALUATOR" => SpanKind::Evaluator,
        "PROMPT" => SpanKind::Prompt,
        _ => SpanKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn extracts_root_chain_input_and_output_from_openinference_attributes() {
        let mut attributes = BTreeMap::new();
        attributes.insert(
            "openinference.span.kind".to_string(),
            Value::String("CHAIN".to_string()),
        );
        attributes.insert(
            "input.value".to_string(),
            Value::String("book a flight".to_string()),
        );
        attributes.insert(
            "output.value".to_string(),
            Value::String("flight booked".to_string()),
        );

        let trace = Trace::new("trace-1").with_span(Span {
            id: "span-1".to_string(),
            trace_id: Some("trace-1".to_string()),
            parent_id: None,
            name: "agent".to_string(),
            kind: SpanKind::Other,
            input: None,
            output: None,
            error: None,
            started_at: None,
            ended_at: None,
            attributes,
            ..Span::new("", "")
        });

        let case = OpenInferenceExtractor.extract_trace(&trace).unwrap();

        assert_eq!(case.input, "book a flight");
        assert_eq!(case.actual_output.as_deref(), Some("flight booked"));
        assert_eq!(
            case.metadata.get("extractor"),
            Some(&Value::String("openinference".to_string()))
        );
    }

    #[test]
    fn maps_openinference_span_kinds() {
        assert_eq!(
            OpenInferenceExtractor::span_kind_from_attribute("LLM"),
            SpanKind::Llm
        );
        assert_eq!(
            OpenInferenceExtractor::span_kind_from_attribute("agent"),
            SpanKind::Agent
        );
        assert_eq!(
            OpenInferenceExtractor::span_kind_from_attribute("unknown"),
            SpanKind::Other
        );
    }

    #[test]
    fn preserves_agent_context_and_tool_calls() {
        let mut agent_attributes = BTreeMap::new();
        agent_attributes.insert(
            "openinference.span.kind".to_string(),
            Value::String("AGENT".to_string()),
        );
        agent_attributes.insert(
            "input.value".to_string(),
            Value::String("replace my card".to_string()),
        );
        agent_attributes.insert(
            "output.value".to_string(),
            Value::String("card replaced".to_string()),
        );
        agent_attributes.insert(
            "state_delta".to_string(),
            Value::String(r#"{"resolved":true}"#.to_string()),
        );
        agent_attributes.insert(
            "retrieval_doc_ids".to_string(),
            Value::String("CARD_POLICY".to_string()),
        );
        agent_attributes.insert(
            "acme.routing_dimension".to_string(),
            Value::String("priority_support".to_string()),
        );

        let mut tool_attributes = BTreeMap::new();
        tool_attributes.insert(
            "gen_ai.tool.name".to_string(),
            Value::String("checkEligibility".to_string()),
        );
        tool_attributes.insert(
            "state_delta".to_string(),
            Value::String(r#"{"eligible":true}"#.to_string()),
        );
        tool_attributes.insert("approval_required".to_string(), Value::Bool(false));
        tool_attributes.insert(
            "acme.tool_region".to_string(),
            Value::String("eu-west".to_string()),
        );

        let trace = Trace::new("trace-1")
            .with_span(Span {
                id: "agent-span".to_string(),
                trace_id: Some("trace-1".to_string()),
                parent_id: None,
                name: "agent".to_string(),
                kind: SpanKind::Other,
                input: None,
                output: None,
                error: None,
                started_at: None,
                ended_at: None,
                attributes: agent_attributes,
                ..Span::new("", "")
            })
            .with_span(Span {
                id: "tool-span".to_string(),
                trace_id: Some("trace-1".to_string()),
                parent_id: Some("agent-span".to_string()),
                name: "execute_tool checkEligibility".to_string(),
                kind: SpanKind::Tool,
                input: None,
                output: None,
                error: None,
                started_at: None,
                ended_at: None,
                attributes: tool_attributes,
                ..Span::new("", "")
            });

        let case = OpenInferenceExtractor.extract_trace(&trace).unwrap();

        assert_eq!(case.metadata["state_delta"], r#"{"resolved":true}"#);
        assert_eq!(case.metadata["retrieval_doc_ids"], "CARD_POLICY");
        assert_eq!(case.metadata["acme.routing_dimension"], "priority_support");
        assert_eq!(
            case.metadata["tool_calls"][0]["tool_name"],
            "checkEligibility"
        );
        assert_eq!(
            case.metadata["tool_calls"][0]["attributes"]["state_delta"],
            r#"{"eligible":true}"#
        );
        assert_eq!(
            case.metadata["tool_calls"][0]["attributes"]["approval_required"],
            false
        );
        assert_eq!(
            case.metadata["tool_calls"][0]["attributes"]["acme.tool_region"],
            "eu-west"
        );
    }
}
