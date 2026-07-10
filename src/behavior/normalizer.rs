use std::collections::{BTreeMap, HashMap};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::Result;
use crate::model::{Span, SpanKind, Trace};

use super::model::{
    AgentBehaviorTrace, AgentRole, AgentTurn, ApprovalOutcome, EvidenceRef, FinalOutcome,
    FinalOutcomeStatus, NormalizedToolError, PolicyDecision, PolicyDecisionOutcome, StateChangeRef,
    ToolCallFact, ToolCallStatus,
};

pub trait AgentBehaviorNormalizer {
    fn normalize(&self, trace: &Trace) -> Result<AgentBehaviorTrace>;

    fn normalize_traces(&self, traces: &[Trace]) -> Result<Vec<AgentBehaviorTrace>> {
        traces.iter().map(|trace| self.normalize(trace)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct OpenInferenceBehaviorNormalizer {
    max_summary_chars: usize,
}

impl Default for OpenInferenceBehaviorNormalizer {
    fn default() -> Self {
        Self {
            max_summary_chars: 4_096,
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
            let operation = operation(span, &tool_name);
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
                inferred_attempt,
            ));
        }

        behavior.policy_decisions = trace
            .spans
            .iter()
            .filter_map(normalize_policy_decision)
            .collect();
        add_escalation_decisions(trace, &mut behavior.policy_decisions);
        behavior.final_outcome = normalize_final_outcome(
            root,
            output.as_ref().map(|(_, value)| value.as_str()),
            &behavior.tool_calls,
            &behavior.policy_decisions,
            self.max_summary_chars,
        );

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

fn normalize_tool_call(
    span: &Span,
    tool_name: String,
    operation: Option<String>,
    inferred_attempt: u32,
) -> ToolCallFact {
    let status = tool_status(span);
    let error = normalized_error(span, status);
    let state_change = state_change(span, operation.as_deref());
    let mut evidence = vec![EvidenceRef::span(&span.id)];
    if let Some(state_change) = &state_change {
        evidence.push(state_change.evidence.clone());
    }
    let mutating = bool_attribute(span, &["agent.mutation", "tool.mutation", "tool.mutating"])
        .unwrap_or_else(|| operation.as_deref().is_some_and(is_mutating_operation));
    let idempotent = bool_attribute(
        span,
        &[
            "agent.idempotent",
            "tool.idempotent",
            "gen_ai.tool.idempotent",
        ],
    )
    .or_else(|| {
        operation
            .as_deref()
            .filter(|operation| matches!(*operation, "lookup" | "verify"))
            .map(|_| true)
    });

    ToolCallFact {
        call_id: string_attribute(
            span,
            &[
                "gen_ai.tool.call.id",
                "tool.call.id",
                "tool_call_id",
                "call_id",
            ],
        )
        .unwrap_or_else(|| span.id.clone()),
        tool_name,
        operation,
        attempt: u32_attribute(span, &["agent.tool.attempt", "tool.attempt", "attempt"])
            .unwrap_or(inferred_attempt),
        started_at: span
            .started_at
            .clone()
            .or_else(|| span.ended_at.clone())
            .unwrap_or_default(),
        duration_ms: duration_ms(span),
        status,
        error,
        approval_required: bool_attribute(
            span,
            &[
                "agent.approval.required",
                "approval_required",
                "tool.approval.required",
            ],
        )
        .unwrap_or(false),
        approval_outcome: approval_outcome(span),
        state_change,
        required: bool_attribute(span, &["agent.tool.required", "tool.required"]).unwrap_or(true),
        idempotent,
        mutating,
        evidence,
    }
}

fn tool_status(span: &Span) -> ToolCallStatus {
    // Failure precedence follows the architecture contract. Success is only
    // inferred after all bounded failure signals have been considered.
    if structured_result_bool(span) == Some(false) {
        return ToolCallStatus::Failed;
    }

    let status_text = string_attribute(
        span,
        &[
            "agent.tool.status",
            "tool.status",
            "gen_ai.tool.status",
            "execution.status",
        ],
    )
    .unwrap_or_default()
    .to_ascii_lowercase();
    let error_text = span
        .error
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if bool_attribute(span, &["tool.timeout", "execution.timeout"]) == Some(true)
        || status_text.contains("timeout")
        || error_text.contains("timeout")
        || error_text.contains("timed out")
    {
        return ToolCallStatus::TimedOut;
    }
    if bool_attribute(span, &["tool.cancelled", "execution.cancelled"]) == Some(true)
        || status_text.contains("cancel")
        || error_text.contains("cancel")
    {
        return ToolCallStatus::Cancelled;
    }
    if has_exception_event(span) {
        return ToolCallStatus::Failed;
    }
    if span.error.is_some() || matches!(status_text.as_str(), "failed" | "failure" | "error") {
        return ToolCallStatus::Failed;
    }
    if protocol_status(span).is_some_and(|status| status >= 400) {
        return ToolCallStatus::Failed;
    }
    if matches!(
        status_text.as_str(),
        "unknown" | "uncertain" | "ambiguous" | "incomplete"
    ) {
        return ToolCallStatus::Unknown;
    }

    if structured_result_bool(span) == Some(true)
        || matches!(
            status_text.as_str(),
            "succeeded" | "success" | "ok" | "completed"
        )
        || state_delta(span).is_some_and(|value| !is_empty_state_delta(&value))
        || span
            .output
            .as_ref()
            .is_some_and(|output| !output.trim().is_empty())
    {
        ToolCallStatus::Succeeded
    } else {
        ToolCallStatus::Unknown
    }
}

fn normalized_error(span: &Span, status: ToolCallStatus) -> Option<NormalizedToolError> {
    if !status.is_failure() && status != ToolCallStatus::Unknown {
        return None;
    }

    let message = span.error.as_deref().or_else(|| {
        span.attributes
            .get("exception.message")
            .or_else(|| span.attributes.get("error.message"))
            .or_else(|| span.attributes.get("tool.error.message"))
            .and_then(Value::as_str)
    });
    let code = string_attribute(
        span,
        &[
            "error.code",
            "tool.error.code",
            "exception.type",
            "http.status_code",
        ],
    )
    .or_else(|| protocol_status(span).map(|value| value.to_string()));
    let kind = string_attribute(span, &["error.type", "tool.error.kind", "exception.type"])
        .unwrap_or_else(|| match status {
            ToolCallStatus::TimedOut => "timeout".to_string(),
            ToolCallStatus::Cancelled => "cancelled".to_string(),
            ToolCallStatus::Unknown => "unknown".to_string(),
            ToolCallStatus::Failed => classify_error_kind(message.unwrap_or_default()).to_string(),
            ToolCallStatus::Succeeded => "none".to_string(),
        });
    let retryable =
        bool_attribute(span, &["error.retryable", "tool.error.retryable"]).or(match status {
            ToolCallStatus::TimedOut => Some(true),
            ToolCallStatus::Cancelled => Some(false),
            _ => None,
        });

    Some(NormalizedToolError {
        kind,
        code,
        retryable,
        redacted_message_hash: message.filter(|message| !message.is_empty()).map(hash_text),
    })
}

fn state_change(span: &Span, operation: Option<&str>) -> Option<StateChangeRef> {
    let value = state_delta(span)?;
    if is_empty_state_delta(&value) {
        return None;
    }
    let serialized = canonical_json(&value);
    let blocked = value
        .get("blocked")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let explicit_no_change = value
        .get("changed")
        .and_then(Value::as_bool)
        .is_some_and(|changed| !changed);
    let null_action = value.get("actionApplied").is_some_and(Value::is_null);
    let verified = value
        .get("verified")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || value
            .get("stateVerified")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || operation == Some("verify");
    let evidence = EvidenceRef {
        kind: "state_change".to_string(),
        identity: format!("state_change:{}:{}", span.id, hash_text(&serialized)),
        span_id: Some(span.id.clone()),
    };

    Some(StateChangeRef {
        evidence,
        changed: !(blocked || explicit_no_change || null_action),
        verified,
    })
}

fn normalize_policy_decision(span: &Span) -> Option<PolicyDecision> {
    let outcome = string_attribute(
        span,
        &[
            "agent.policy.outcome",
            "policy.decision.outcome",
            "policy.outcome",
            "guardrail.outcome",
        ],
    )?;
    let outcome = match outcome.to_ascii_lowercase().as_str() {
        "allow" | "allowed" | "approve" | "approved" => PolicyDecisionOutcome::Allowed,
        "deny" | "denied" | "disallow" | "disallowed" | "blocked" => PolicyDecisionOutcome::Denied,
        "require" | "required" => PolicyDecisionOutcome::Required,
        _ => PolicyDecisionOutcome::Unknown,
    };

    Some(PolicyDecision {
        decision_id: string_attribute(span, &["policy.decision.id", "decision_id"])
            .unwrap_or_else(|| format!("decision:{}", span.id)),
        policy_id: string_attribute(span, &["policy.id", "policy.version", "policy_version"]),
        action: string_attribute(span, &["policy.action", "action", "tool_name"]),
        outcome,
        reason_code: string_attribute(span, &["policy.reason_code", "reason_code"]),
        evidence: vec![EvidenceRef::span(&span.id)],
    })
}

fn add_escalation_decisions(trace: &Trace, decisions: &mut Vec<PolicyDecision>) {
    for span in &trace.spans {
        let Some(delta) = state_delta(span) else {
            continue;
        };
        if delta.get("requiresEscalation").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        let decision_id = format!("escalation:{}", span.id);
        if decisions
            .iter()
            .any(|decision| decision.decision_id == decision_id)
        {
            continue;
        }
        decisions.push(PolicyDecision {
            decision_id,
            policy_id: string_attribute(span, &["policy.id", "policy.version", "policy_version"]),
            action: Some("escalation".to_string()),
            outcome: PolicyDecisionOutcome::Required,
            reason_code: None,
            evidence: vec![EvidenceRef::span(&span.id)],
        });
    }
}

fn normalize_final_outcome(
    root: Option<&Span>,
    output: Option<&str>,
    tool_calls: &[ToolCallFact],
    policy_decisions: &[PolicyDecision],
    max_summary_chars: usize,
) -> FinalOutcome {
    let response = output.map(|value| bounded_text(value, max_summary_chars));
    let response_lower = output.unwrap_or_default().to_ascii_lowercase();
    let root_delta = root.and_then(state_delta);
    let resolution_outcome = root_delta
        .as_ref()
        .and_then(|delta| delta.get("finalResolution"))
        .filter(|value| !value.is_null())
        .and_then(|resolution| resolution.get("outcome"))
        .and_then(Value::as_str);
    let resolution_present = root
        .and_then(|span| bool_attribute(span, &["agent.resolution.present", "resolution_present"]))
        .unwrap_or_else(|| {
            root_delta
                .as_ref()
                .and_then(|delta| delta.get("finalResolution"))
                .is_some_and(|value| !value.is_null())
        });
    let escalation_required = root
        .and_then(|span| {
            bool_attribute(span, &["agent.escalation.required", "escalation_required"])
        })
        .unwrap_or_else(|| {
            policy_decisions.iter().any(|decision| {
                decision.outcome == PolicyDecisionOutcome::Required
                    && decision
                        .action
                        .as_deref()
                        .is_some_and(|action| action.contains("escalat"))
            })
        });
    let escalation_performed = root
        .and_then(|span| {
            bool_attribute(
                span,
                &["agent.escalation.performed", "escalation_performed"],
            )
        })
        .unwrap_or_else(|| {
            tool_calls.iter().any(|call| {
                call.status == ToolCallStatus::Succeeded
                    && (call.operation.as_deref() == Some("escalate")
                        || call.tool_name.to_ascii_lowercase().contains("escalat"))
            })
        });
    let claimed_success = root
        .and_then(|span| {
            bool_attribute(
                span,
                &["agent.final.claimed_success", "final.claimed_success"],
            )
        })
        .unwrap_or_else(|| claims_success(&response_lower));
    let failure_acknowledged = root
        .and_then(|span| {
            bool_attribute(
                span,
                &[
                    "agent.final.failure_acknowledged",
                    "final.failure_acknowledged",
                ],
            )
        })
        .unwrap_or_else(|| acknowledges_failure(&response_lower));
    let explicit_status = root.and_then(|span| {
        string_attribute(span, &["agent.final.status", "final.status", "outcome"])
    });

    let status = explicit_status
        .as_deref()
        .and_then(parse_final_status)
        .or_else(|| resolution_outcome.and_then(parse_final_status))
        .unwrap_or_else(|| {
            if root.is_some_and(|span| span.error.is_some()) {
                FinalOutcomeStatus::Failed
            } else if escalation_performed {
                FinalOutcomeStatus::Escalated
            } else if response.is_none() {
                FinalOutcomeStatus::Incomplete
            } else if resolution_present {
                FinalOutcomeStatus::Succeeded
            } else if failure_acknowledged {
                FinalOutcomeStatus::Failed
            } else {
                FinalOutcomeStatus::Unknown
            }
        });

    let evidence = root
        .map(|span| vec![EvidenceRef::span(&span.id)])
        .unwrap_or_default();
    FinalOutcome {
        status,
        response_summary: response,
        claimed_success,
        resolution_present,
        escalation_required,
        escalation_performed,
        failure_acknowledged,
        evidence,
    }
}

fn root_agent_span(trace: &Trace) -> Option<&Span> {
    trace
        .spans
        .iter()
        .find(|span| span.parent_id.is_none() && span_kind(span) == SpanKind::Agent)
        .or_else(|| {
            trace.spans.iter().find(|span| {
                span_kind(span) == SpanKind::Agent
                    && (span.input.is_some() || span.output.is_some())
            })
        })
        .or_else(|| {
            trace.spans.iter().find(|span| {
                span.parent_id.is_none()
                    && matches!(span_kind(span), SpanKind::Agent | SpanKind::Chain)
            })
        })
}

fn span_input(span: &Span) -> Option<(&Span, String)> {
    span.input
        .clone()
        .or_else(|| string_attribute(span, &["input.value"]))
        .map(|value| (span, value))
}

fn span_output(span: &Span) -> Option<(&Span, String)> {
    span.output
        .clone()
        .or_else(|| string_attribute(span, &["output.value"]))
        .map(|value| (span, value))
}

fn span_kind(span: &Span) -> SpanKind {
    if span.kind != SpanKind::Other {
        return span.kind;
    }
    match string_attribute(span, &["openinference.span.kind"])
        .unwrap_or_default()
        .to_ascii_uppercase()
        .as_str()
    {
        "LLM" => SpanKind::Llm,
        "AGENT" => SpanKind::Agent,
        "TOOL" => SpanKind::Tool,
        "CHAIN" => SpanKind::Chain,
        "GUARDRAIL" => SpanKind::Guardrail,
        _ => SpanKind::Other,
    }
}

fn tool_name(span: &Span) -> String {
    string_attribute(span, &["gen_ai.tool.name", "tool.name", "tool_name"])
        .unwrap_or_else(|| span.name.clone())
}

fn operation(span: &Span, tool_name: &str) -> Option<String> {
    if let Some(operation) =
        string_attribute(span, &["agent.operation", "tool.operation", "operation"])
    {
        return Some(operation.to_ascii_lowercase());
    }
    let name = tool_name.to_ascii_lowercase();
    let operation = if name.contains("refund") {
        "refund"
    } else if name.contains("cancel") {
        "cancel"
    } else if name.contains("verify") || name.contains("confirm") {
        "verify"
    } else if name.contains("escalat") {
        "escalate"
    } else if name.contains("create") || name.contains("order") {
        "create"
    } else if name.contains("record")
        || name.contains("update")
        || name.contains("lock")
        || name.contains("set")
        || name.contains("expedite")
    {
        "update"
    } else if name.contains("lookup")
        || name.contains("retrieve")
        || name.contains("get")
        || name.contains("check")
        || name.contains("classif")
        || name.contains("search")
    {
        "lookup"
    } else {
        return None;
    };
    Some(operation.to_string())
}

fn is_mutating_operation(operation: &str) -> bool {
    matches!(operation, "create" | "update" | "cancel" | "refund")
}

fn approval_outcome(span: &Span) -> Option<ApprovalOutcome> {
    let value = string_attribute(
        span,
        &[
            "agent.approval.outcome",
            "approval.outcome",
            "approval_outcome",
        ],
    )?;
    Some(match value.to_ascii_lowercase().as_str() {
        "approved" | "allow" | "allowed" => ApprovalOutcome::Approved,
        "denied" | "deny" | "rejected" => ApprovalOutcome::Denied,
        "cancelled" | "canceled" => ApprovalOutcome::Cancelled,
        "not_requested" | "not-required" | "not_required" => ApprovalOutcome::NotRequested,
        _ => ApprovalOutcome::Unknown,
    })
}

fn structured_result_bool(span: &Span) -> Option<bool> {
    for key in [
        "tool.result.success",
        "tool_result.success",
        "gen_ai.tool.result.success",
        "result.success",
        "result.ok",
    ] {
        if let Some(value) = span.attributes.get(key).and_then(value_as_bool) {
            return Some(value);
        }
    }
    let output = span.output.as_deref()?;
    let value: Value = serde_json::from_str(output).ok()?;
    value
        .get("success")
        .and_then(value_as_bool)
        .or_else(|| value.get("ok").and_then(value_as_bool))
        .or_else(|| value.get("error").map(|error| error.is_null()))
}

fn state_delta(span: &Span) -> Option<Value> {
    let value = span
        .attributes
        .get("state_delta")
        .or_else(|| span.attributes.get("agent.state_delta"))
        .or_else(|| span.attributes.get("state.delta"))?;
    match value {
        Value::String(value) => serde_json::from_str(value).ok(),
        value => Some(value.clone()),
    }
}

fn is_empty_state_delta(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Object(object) => object.is_empty(),
        Value::Array(array) => array.is_empty(),
        Value::String(value) => value.trim().is_empty() || value.trim() == "{}",
        _ => false,
    }
}

fn has_exception_event(span: &Span) -> bool {
    span.attributes.contains_key("exception.type")
        || span.attributes.contains_key("exception.message")
        || bool_attribute(span, &["exception.recorded"]) == Some(true)
}

fn protocol_status(span: &Span) -> Option<u16> {
    for key in [
        "http.status_code",
        "http.response.status_code",
        "rpc.status_code",
        "protocol.status_code",
    ] {
        let Some(value) = span.attributes.get(key) else {
            continue;
        };
        if let Some(value) = value.as_u64().and_then(|value| u16::try_from(value).ok()) {
            return Some(value);
        }
        if let Some(value) = value.as_str().and_then(|value| value.parse::<u16>().ok()) {
            return Some(value);
        }
    }
    None
}

fn duration_ms(span: &Span) -> u64 {
    for key in ["duration_ms", "tool.duration_ms", "gen_ai.tool.duration_ms"] {
        if let Some(value) = span.attributes.get(key).and_then(Value::as_u64) {
            return value;
        }
        if let Some(value) = span.attributes.get(key).and_then(Value::as_f64) {
            return value.max(0.0).round() as u64;
        }
    }
    for key in [
        "gen_ai.execute_tool.duration",
        "tool.duration",
        "execution.duration",
    ] {
        if let Some(seconds) = span.attributes.get(key).and_then(Value::as_f64) {
            return (seconds.max(0.0) * 1_000.0).round() as u64;
        }
    }
    0
}

fn string_attribute(span: &Span, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        span.attributes.get(*key).and_then(|value| match value {
            Value::String(value) => Some(value.clone()),
            Value::Number(value) => Some(value.to_string()),
            _ => None,
        })
    })
}

fn bool_attribute(span: &Span, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| span.attributes.get(*key).and_then(value_as_bool))
}

fn value_as_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::String(value) => match value.to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" => Some(true),
            "false" | "no" | "0" => Some(false),
            _ => None,
        },
        Value::Number(value) => value.as_u64().map(|value| value != 0),
        _ => None,
    }
}

fn u32_attribute(span: &Span, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|key| {
        span.attributes.get(*key).and_then(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        })
    })
}

fn classify_error_kind(message: &str) -> &'static str {
    let message = message.to_ascii_lowercase();
    if message.contains("rate limit") || message.contains("429") {
        "rate_limited"
    } else if message.contains("denied")
        || message.contains("forbidden")
        || message.contains("unauthorized")
    {
        "permission_denied"
    } else if message.contains("invalid") || message.contains("validation") {
        "invalid_request"
    } else if message.contains("network") || message.contains("connection") {
        "network"
    } else {
        "tool_error"
    }
}

fn parse_final_status(value: &str) -> Option<FinalOutcomeStatus> {
    Some(match value.to_ascii_lowercase().as_str() {
        "success" | "succeeded" | "resolved" | "complete" | "completed" => {
            FinalOutcomeStatus::Succeeded
        }
        "failure" | "failed" | "error" => FinalOutcomeStatus::Failed,
        "refused" | "denied" | "safe_refusal" => FinalOutcomeStatus::Refused,
        "escalated" | "escalation" => FinalOutcomeStatus::Escalated,
        "incomplete" | "missing" => FinalOutcomeStatus::Incomplete,
        "unknown" | "uncertain" => FinalOutcomeStatus::Unknown,
        _ => return None,
    })
}

fn claims_success(response: &str) -> bool {
    [
        "successfully",
        "has been cancelled",
        "has been canceled",
        "was cancelled",
        "was canceled",
        "has been refunded",
        "refund is complete",
        "completed your request",
        "action is complete",
        "we've submitted",
        "we have submitted",
        "is now locked",
        "has been updated",
    ]
    .iter()
    .any(|phrase| response.contains(phrase))
}

fn acknowledges_failure(response: &str) -> bool {
    [
        "could not complete",
        "couldn't complete",
        "was not completed",
        "has not been completed",
        "unable to complete",
        "i can't complete",
        "i cannot complete",
        "state is unknown",
        "could not verify",
    ]
    .iter()
    .any(|phrase| response.contains(phrase))
}

fn bounded_text(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    value.chars().take(max_chars).collect()
}

fn canonical_json(value: &Value) -> String {
    serde_json::to_string(&canonicalize_value(value)).unwrap_or_default()
}

fn canonicalize_value(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let ordered = object
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize_value(value)))
                .collect::<BTreeMap<_, _>>();
            Value::Object(ordered.into_iter().collect())
        }
        Value::Array(array) => Value::Array(array.iter().map(canonicalize_value).collect()),
        value => value.clone(),
    }
}

fn hash_text(value: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(value.as_bytes()))
}

#[cfg(test)]
mod tests {
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
        attributes.insert("tool.result.success".to_string(), json!(false));
        attributes.insert("state_delta".to_string(), json!(r#"{"changed":true}"#));
        let tool = span_with_attributes("tool-1", "cancel", SpanKind::Tool, attributes);
        let trace = Trace::new("trace-1").with_span(tool);

        let behavior = OpenInferenceBehaviorNormalizer::default()
            .normalize(&trace)
            .unwrap();

        assert_eq!(behavior.tool_calls[0].status, ToolCallStatus::Failed);
        assert_eq!(behavior.tool_calls[0].operation.as_deref(), Some("cancel"));
        assert!(behavior.tool_calls[0].mutating);
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
}
