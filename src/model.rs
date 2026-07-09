use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    pub id: String,
    #[serde(default)]
    pub spans: Vec<Span>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl Trace {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            spans: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.spans.push(span);
        self
    }

    pub fn to_eval_case(&self) -> Result<EvalCase> {
        EvalCase::try_from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub kind: SpanKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, Value>,
}

impl Span {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            trace_id: None,
            parent_id: None,
            name: name.into(),
            kind: SpanKind::Other,
            input: None,
            output: None,
            error: None,
            started_at: None,
            ended_at: None,
            attributes: BTreeMap::new(),
        }
    }

    pub fn llm(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            kind: SpanKind::Llm,
            ..Self::new(id, name)
        }
    }

    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.input = Some(input.into());
        self
    }

    pub fn with_output(mut self, output: impl Into<String>) -> Self {
        self.output = Some(output.into());
        self
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    Llm,
    Tool,
    Chain,
    Retriever,
    #[default]
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalCase {
    pub id: String,
    pub trace_id: String,
    pub input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rubric: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl EvalCase {
    pub fn new(
        id: impl Into<String>,
        trace_id: impl Into<String>,
        input: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            trace_id: trace_id.into(),
            input: input.into(),
            actual_output: None,
            expected_output: None,
            rubric: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_actual_output(mut self, actual_output: impl Into<String>) -> Self {
        self.actual_output = Some(actual_output.into());
        self
    }

    pub fn with_expected_output(mut self, expected_output: impl Into<String>) -> Self {
        self.expected_output = Some(expected_output.into());
        self
    }

    pub fn with_rubric(mut self, rubric: impl Into<String>) -> Self {
        self.rubric = Some(rubric.into());
        self
    }
}

impl TryFrom<&Trace> for EvalCase {
    type Error = anyhow::Error;

    fn try_from(trace: &Trace) -> Result<Self> {
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
            input: input_span.input.clone().expect("input checked above"),
            actual_output: output_span.and_then(|span| span.output.clone()),
            expected_output: None,
            rubric: None,
            metadata,
        })
    }
}

pub fn traces_to_eval_cases(traces: &[Trace]) -> Result<Vec<EvalCase>> {
    traces.iter().map(EvalCase::try_from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_eval_case_from_llm_spans() {
        let trace = Trace::new("trace-1")
            .with_span(Span::llm("span-1", "prompt").with_input("hello"))
            .with_span(Span::llm("span-2", "completion").with_output("world"));

        let case = trace.to_eval_case().unwrap();

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

        let error = trace.to_eval_case().unwrap_err().to_string();

        assert!(error.contains("does not contain span input"));
    }
}
