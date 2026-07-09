use anyhow::{Result, anyhow};
use serde_json::Value;

use crate::extractors::EvalCaseExtractor;
use crate::model::{EvalCase, Span, SpanKind, Trace};

#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleExtractor;

impl SimpleExtractor {
    pub fn extract_trace(&self, trace: &Trace) -> Result<EvalCase> {
        trace_to_eval_case(trace)
    }

    pub fn extract_traces(&self, traces: &[Trace]) -> Result<Vec<EvalCase>> {
        self.extract_cases(traces)
    }
}

impl EvalCaseExtractor for SimpleExtractor {
    fn extract_case(&self, trace: &Trace) -> Result<EvalCase> {
        self.extract_trace(trace)
    }
}

fn trace_to_eval_case(trace: &Trace) -> Result<EvalCase> {
    let input_span = trace
        .spans
        .iter()
        .find(|span| span.kind == SpanKind::Llm && span.input.is_some())
        .or_else(|| trace.spans.iter().find(|span| span.input.is_some()))
        .ok_or_else(|| anyhow!("trace {} does not contain span input", trace.id))?;

    let output_span = trace
        .spans
        .iter()
        .rev()
        .find(|span| span.kind == SpanKind::Llm && span.output.is_some())
        .or_else(|| trace.spans.iter().rev().find(|span| span.output.is_some()));

    build_eval_case(trace, input_span, output_span)
}

fn build_eval_case(
    trace: &Trace,
    input_span: &Span,
    output_span: Option<&Span>,
) -> Result<EvalCase> {
    let input = input_span
        .input
        .clone()
        .ok_or_else(|| anyhow!("span {} has no input", input_span.id))?;

    let mut metadata = trace.metadata.clone();
    metadata.insert(
        "source_span_id".to_string(),
        Value::String(input_span.id.clone()),
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
        input,
        actual_output: output_span.and_then(|span| span.output.clone()),
        expected_output: None,
        rubric: None,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_eval_case_from_llm_spans() {
        let trace = Trace::new("trace-1")
            .with_span(Span::llm("span-1", "prompt").with_input("hello"))
            .with_span(Span::llm("span-2", "completion").with_output("world"));

        let case = SimpleExtractor.extract_trace(&trace).unwrap();

        assert_eq!(case.id, "trace-1:span-1");
        assert_eq!(case.trace_id, "trace-1");
        assert_eq!(case.input, "hello");
        assert_eq!(case.actual_output.as_deref(), Some("world"));
        assert_eq!(
            case.metadata.get("source_span_id"),
            Some(&Value::String("span-1".to_string()))
        );
    }

    #[test]
    fn requires_input_for_eval_case() {
        let trace = Trace::new("trace-1").with_span(Span::llm("span-1", "completion"));

        let error = SimpleExtractor
            .extract_trace(&trace)
            .unwrap_err()
            .to_string();

        assert!(error.contains("does not contain span input"));
    }
}
