use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use crate::behavior::{AgentBehaviorTrace, AgentRole, EvidenceRef};

use super::model::{
    SEMANTIC_BEHAVIOR_PROJECTION_SCHEMA_VERSION, SEMANTIC_BEHAVIOR_PROJECTION_VERSION,
    SemanticBehaviorFinalOutcome, SemanticBehaviorProjection, SemanticContentPolicy,
    SemanticEvidenceRef, SemanticFinalClaim, SemanticPolicyDecision, SemanticToolCall,
};

#[derive(Debug, Clone)]
pub struct SemanticBehaviorProjector {
    content_policy: SemanticContentPolicy,
    max_summary_chars: usize,
    max_tool_calls: usize,
    max_policy_decisions: usize,
}

impl Default for SemanticBehaviorProjector {
    fn default() -> Self {
        Self {
            content_policy: SemanticContentPolicy::StructuredOnly,
            max_summary_chars: 4_096,
            max_tool_calls: 64,
            max_policy_decisions: 64,
        }
    }
}

impl SemanticBehaviorProjector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_content_policy(mut self, content_policy: SemanticContentPolicy) -> Self {
        self.content_policy = content_policy;
        self
    }

    pub fn with_max_summary_chars(mut self, max_summary_chars: usize) -> Self {
        self.max_summary_chars = max_summary_chars.max(1);
        self
    }

    pub fn with_max_tool_calls(mut self, max_tool_calls: usize) -> Self {
        self.max_tool_calls = max_tool_calls.max(1);
        self
    }

    pub fn with_max_policy_decisions(mut self, max_policy_decisions: usize) -> Self {
        self.max_policy_decisions = max_policy_decisions.max(1);
        self
    }

    pub fn content_policy(&self) -> SemanticContentPolicy {
        self.content_policy
    }

    pub fn project(&self, trace: &AgentBehaviorTrace) -> SemanticBehaviorProjection {
        let included_calls = trace
            .tool_calls
            .iter()
            .take(self.max_tool_calls)
            .collect::<Vec<_>>();
        let included_decisions = trace
            .policy_decisions
            .iter()
            .take(self.max_policy_decisions)
            .collect::<Vec<_>>();
        let evidence = evidence_catalog(trace, &included_calls, &included_decisions);
        let evidence_keys = evidence
            .iter()
            .map(|item| (item.source.identity.as_str(), item.key.as_str()))
            .collect::<BTreeMap<_, _>>();

        let tool_calls = included_calls
            .iter()
            .enumerate()
            .map(|(sequence, call)| SemanticToolCall {
                sequence,
                tool_name: safe_label(&call.tool_name),
                operation: call.operation.as_deref().map(safe_label),
                effect: call.effect,
                retry_safety: call.retry_safety,
                requirement: call.requirement,
                attempt: call.attempt,
                duration_ms: call.duration_ms,
                status: call.status,
                error_kind: call.error.as_ref().map(|error| safe_label(&error.kind)),
                error_code: call
                    .error
                    .as_ref()
                    .and_then(|error| error.code.as_deref())
                    .map(safe_label),
                error_retryable: call.error.as_ref().and_then(|error| error.retryable),
                approval_required: call.approval_required,
                approval_outcome: call.approval_outcome,
                state_predicate: call
                    .state_change
                    .as_ref()
                    .and_then(|state| state.predicate.as_deref())
                    .map(safe_label),
                state_observation: call.state_change.as_ref().map(|state| state.observation),
                evidence_keys: keys_for(&call.evidence, &evidence_keys),
            })
            .collect::<Vec<_>>();

        let policy_decisions = included_decisions
            .iter()
            .map(|decision| SemanticPolicyDecision {
                policy_id: decision.policy_id.as_deref().map(safe_label),
                action: decision.action.as_deref().map(safe_label),
                outcome: decision.outcome,
                reason_code: decision.reason_code.as_deref().map(safe_label),
                evidence_keys: keys_for(&decision.evidence, &evidence_keys),
            })
            .collect::<Vec<_>>();

        let final_outcome = SemanticBehaviorFinalOutcome {
            status: trace.final_outcome.status,
            escalation: trace.final_outcome.escalation,
            claims: trace
                .final_outcome
                .claims
                .iter()
                .map(|claim| SemanticFinalClaim {
                    operation: claim.operation.as_deref().map(safe_label),
                    status: claim.status,
                    evidence_keys: keys_for(&claim.evidence, &evidence_keys),
                })
                .collect(),
            evidence_keys: keys_for(&trace.final_outcome.evidence, &evidence_keys),
        };
        let input_summary = self.projected_input(trace);
        let final_response_summary = self.projected_response(trace);
        let truncated = trace.tool_calls.len() > tool_calls.len()
            || trace.policy_decisions.len() > policy_decisions.len()
            || content_was_truncated(trace.input_summary.as_deref(), input_summary.as_deref())
            || content_was_truncated(
                final_assistant_summary(trace),
                final_response_summary.as_deref(),
            );

        let mut projection = SemanticBehaviorProjection {
            schema_version: SEMANTIC_BEHAVIOR_PROJECTION_SCHEMA_VERSION.to_string(),
            projection_id: String::new(),
            projection_version: SEMANTIC_BEHAVIOR_PROJECTION_VERSION.to_string(),
            projection_hash: String::new(),
            trace_id: trace.trace_id.clone(),
            content_policy: self.content_policy,
            input_summary,
            final_response_summary,
            tool_calls,
            policy_decisions,
            final_outcome,
            evidence,
            truncated,
        };
        projection.projection_hash = projection_content_hash(&projection);
        projection.projection_id = hash_parts([
            "semantic_behavior_projection",
            projection.trace_id.as_str(),
            projection.projection_hash.as_str(),
        ]);
        projection
    }

    fn projected_input(&self, trace: &AgentBehaviorTrace) -> Option<String> {
        match self.content_policy {
            SemanticContentPolicy::StructuredOnly => None,
            SemanticContentPolicy::PreRedactedSummaries => trace
                .input_summary
                .as_deref()
                .map(|value| bounded_text(value, self.max_summary_chars)),
        }
    }

    fn projected_response(&self, trace: &AgentBehaviorTrace) -> Option<String> {
        match self.content_policy {
            SemanticContentPolicy::StructuredOnly => None,
            SemanticContentPolicy::PreRedactedSummaries => final_assistant_summary(trace)
                .map(|value| bounded_text(value, self.max_summary_chars)),
        }
    }
}

fn evidence_catalog(
    trace: &AgentBehaviorTrace,
    calls: &[&crate::behavior::ToolCallFact],
    decisions: &[&crate::behavior::PolicyDecision],
) -> Vec<SemanticEvidenceRef> {
    let mut evidence = trace
        .evidence
        .iter()
        .cloned()
        .chain(calls.iter().flat_map(|call| call.evidence.iter().cloned()))
        .chain(
            decisions
                .iter()
                .flat_map(|decision| decision.evidence.iter().cloned()),
        )
        .chain(trace.final_outcome.evidence.iter().cloned())
        .chain(
            trace
                .final_outcome
                .claims
                .iter()
                .flat_map(|claim| claim.evidence.iter().cloned()),
        )
        .collect::<Vec<_>>();
    evidence.sort_by(|left, right| {
        left.identity
            .cmp(&right.identity)
            .then_with(|| left.kind.cmp(&right.kind))
    });
    evidence.dedup_by(|left, right| left.identity == right.identity);
    evidence
        .into_iter()
        .enumerate()
        .map(|(index, source)| SemanticEvidenceRef {
            key: format!("e{}", index + 1),
            source,
        })
        .collect()
}

fn keys_for(evidence: &[EvidenceRef], catalog: &BTreeMap<&str, &str>) -> Vec<String> {
    evidence
        .iter()
        .filter_map(|evidence| catalog.get(evidence.identity.as_str()).copied())
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn final_assistant_summary(trace: &AgentBehaviorTrace) -> Option<&str> {
    trace
        .turns
        .iter()
        .rev()
        .find(|turn| turn.role == AgentRole::Assistant)
        .and_then(|turn| turn.content_summary.as_deref())
}

fn safe_label(value: &str) -> String {
    let value = value.trim();
    if super::super::adapter::is_valid_semantic_label(value) {
        value.to_string()
    } else {
        format!("sha256:{:x}", Sha256::digest(value.as_bytes()))
    }
}

fn bounded_text(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn content_was_truncated(original: Option<&str>, projected: Option<&str>) -> bool {
    original
        .zip(projected)
        .is_some_and(|(original, projected)| original.chars().count() > projected.chars().count())
}

fn projection_content_hash(projection: &SemanticBehaviorProjection) -> String {
    #[derive(serde::Serialize)]
    struct Content<'a> {
        schema_version: &'a str,
        projection_version: &'a str,
        trace_id: &'a str,
        content_policy: SemanticContentPolicy,
        input_summary: &'a Option<String>,
        final_response_summary: &'a Option<String>,
        tool_calls: &'a [SemanticToolCall],
        policy_decisions: &'a [SemanticPolicyDecision],
        final_outcome: &'a SemanticBehaviorFinalOutcome,
        evidence: &'a [SemanticEvidenceRef],
        truncated: bool,
    }
    let content = Content {
        schema_version: &projection.schema_version,
        projection_version: &projection.projection_version,
        trace_id: &projection.trace_id,
        content_policy: projection.content_policy,
        input_summary: &projection.input_summary,
        final_response_summary: &projection.final_response_summary,
        tool_calls: &projection.tool_calls,
        policy_decisions: &projection.policy_decisions,
        final_outcome: &projection.final_outcome,
        evidence: &projection.evidence,
        truncated: projection.truncated,
    };
    let bytes = serde_json::to_vec(&content).expect("semantic projection serializes");
    format!("sha256:{:x}", Sha256::digest(bytes))
}

pub(super) fn hash_parts<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.len().to_be_bytes());
        hasher.update(part.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}
