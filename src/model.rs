use std::collections::BTreeMap;

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

    pub fn with_kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    Llm,
    Agent,
    Tool,
    Chain,
    Retriever,
    Reranker,
    Embedding,
    Guardrail,
    Evaluator,
    Prompt,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_extended_span_kinds() {
        let span = Span::new("span-1", "agent").with_kind(SpanKind::Agent);

        assert_eq!(span.kind, SpanKind::Agent);
    }
}
