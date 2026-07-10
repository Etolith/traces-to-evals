use std::collections::{BTreeMap, HashMap};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::Result;
use crate::model::{Span, SpanKind, Trace};

use super::adapter::{BehaviorAdapterConfig, ToolSemanticMapping, is_valid_semantic_label};
use super::model::{
    AgentBehaviorTrace, AgentRole, AgentTurn, ApprovalOutcome, ClaimedOutcomeStatus,
    EscalationStatus, EvidenceRef, FinalOutcome, FinalOutcomeStatus, NormalizedToolError,
    OperationEffect, OutcomeClaim, PolicyDecision, PolicyDecisionOutcome, RetrySafety,
    StateChangeRef, StateObservation, ToolCallFact, ToolCallStatus, ToolRequirement,
};

mod attributes;
mod outcome;
mod tool;

use attributes::{
    bounded_text, explicit_operation, root_agent_span, span_input, span_kind, span_output,
    tool_name,
};
use outcome::{normalize_final_outcome, normalize_policy_decision};
use tool::normalize_tool_call;

pub trait AgentBehaviorNormalizer: Send + Sync {
    fn normalize(&self, trace: &Trace) -> Result<AgentBehaviorTrace>;

    fn normalize_traces(&self, traces: &[Trace]) -> Result<Vec<AgentBehaviorTrace>> {
        traces.iter().map(|trace| self.normalize(trace)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct OpenInferenceBehaviorNormalizer {
    max_summary_chars: usize,
    adapter: BehaviorAdapterConfig,
}

impl Default for OpenInferenceBehaviorNormalizer {
    fn default() -> Self {
        Self {
            max_summary_chars: 4_096,
            adapter: BehaviorAdapterConfig::default(),
        }
    }
}

impl OpenInferenceBehaviorNormalizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_summary_chars(mut self, max_summary_chars: usize) -> Self {
        self.max_summary_chars = max_summary_chars;
        self
    }

    pub fn from_adapter(adapter: BehaviorAdapterConfig) -> Result<Self> {
        adapter.validate()?;
        Ok(Self {
            adapter,
            ..Self::default()
        })
    }

    pub fn adapter(&self) -> &BehaviorAdapterConfig {
        &self.adapter
    }
}

impl AgentBehaviorNormalizer for OpenInferenceBehaviorNormalizer {
    fn normalize(&self, trace: &Trace) -> Result<AgentBehaviorTrace> {
        let root = root_agent_span(trace);
        let input = root
            .and_then(span_input)
            .or_else(|| trace.spans.iter().find_map(span_input));
        let output = root
            .and_then(span_output)
            .or_else(|| trace.spans.iter().rev().find_map(span_output));

        let mut behavior = AgentBehaviorTrace::new(&trace.id);
        behavior.metadata = trace.metadata.clone();
        behavior.metadata.insert(
            "traceeval.behavior_adapter.id".to_string(),
            Value::String(self.adapter.adapter_id.clone()),
        );
        behavior.metadata.insert(
            "traceeval.behavior_adapter.version".to_string(),
            Value::String(self.adapter.adapter_version.clone()),
        );
        behavior
            .evidence
            .push(EvidenceRef::new("trace", format!("trace:{}", trace.id)));

        if let Some((span, value)) = input {
            let value = bounded_text(&value, self.max_summary_chars);
            behavior.input_summary = Some(value.clone());
            behavior.turns.push(AgentTurn {
                turn_id: format!("{}:input", span.id),
                role: AgentRole::User,
                content_summary: Some(value),
                evidence: vec![EvidenceRef::span(&span.id)],
            });
        }
        if let Some((span, value)) = &output {
            behavior.turns.push(AgentTurn {
                turn_id: format!("{}:output", span.id),
                role: AgentRole::Assistant,
                content_summary: Some(bounded_text(value, self.max_summary_chars)),
                evidence: vec![EvidenceRef::span(&span.id)],
            });
        }

        let mut attempts: HashMap<(String, Option<String>), u32> = HashMap::new();
        for span in trace
            .spans
            .iter()
            .filter(|span| span_kind(span) == SpanKind::Tool)
        {
            let tool_name = tool_name(span);
            let mapping = self.adapter.mapping_for(&tool_name);
            let operation = explicit_operation(span).or_else(|| mapping.operation.clone());
            let inferred_attempt = {
                let value = attempts
                    .entry((tool_name.clone(), operation.clone()))
                    .or_default();
                *value += 1;
                *value
            };
            behavior.tool_calls.push(normalize_tool_call(
                span,
                tool_name,
                operation,
                mapping,
                inferred_attempt,
            ));
        }

        behavior.policy_decisions = trace
            .spans
            .iter()
            .filter_map(normalize_policy_decision)
            .collect();
        behavior.final_outcome = normalize_final_outcome(root);

        if let Some(observed_at) = root.and_then(|span| span.ended_at.as_ref()).or_else(|| {
            trace
                .spans
                .iter()
                .rev()
                .find_map(|span| span.ended_at.as_ref())
        }) {
            behavior.metadata.insert(
                "observed_at".to_string(),
                Value::String(observed_at.clone()),
            );
        }

        Ok(behavior)
    }
}

#[cfg(test)]
#[path = "normalizer/tests.rs"]
mod tests;
