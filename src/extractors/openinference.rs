use serde_json::Value;

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
}
