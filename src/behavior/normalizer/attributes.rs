use super::*;

pub(super) fn root_agent_span(trace: &Trace) -> Option<&Span> {
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

pub(super) fn span_input(span: &Span) -> Option<(&Span, String)> {
    span.input
        .clone()
        .or_else(|| string_attribute(span, &["input.value"]))
        .map(|value| (span, value))
}

pub(super) fn span_output(span: &Span) -> Option<(&Span, String)> {
    span.output
        .clone()
        .or_else(|| string_attribute(span, &["output.value"]))
        .map(|value| (span, value))
}

pub(super) fn span_kind(span: &Span) -> SpanKind {
    crate::resolved_span_kind(span)
}

pub(super) fn tool_name(span: &Span) -> (String, FactQuality) {
    string_attribute(span, &["gen_ai.tool.name", "tool.name", "tool_name"])
        .map(|name| (name, FactQuality::Explicit))
        .unwrap_or_else(|| (span.name.clone(), FactQuality::Derived))
}

pub(super) fn explicit_operation(span: &Span) -> Option<String> {
    string_attribute(
        span,
        &[
            "gen_ai.operation.name",
            "agent.operation",
            "tool.operation",
            "operation",
            "operation.name",
        ],
    )
    .filter(|operation| is_valid_semantic_label(operation))
}

pub(super) fn explicit_effect(span: &Span) -> Option<OperationEffect> {
    string_attribute(
        span,
        &["agent.operation.effect", "tool.effect", "operation.effect"],
    )
    .as_deref()
    .map(parse_operation_effect)
}

pub(super) fn explicit_retry_safety(span: &Span) -> Option<RetrySafety> {
    string_attribute(
        span,
        &[
            "agent.operation.retry_safety",
            "tool.retry_safety",
            "operation.retry_safety",
        ],
    )
    .as_deref()
    .map(parse_retry_safety)
}

pub(super) fn explicit_requirement(span: &Span) -> Option<ToolRequirement> {
    string_attribute(
        span,
        &[
            "agent.tool.requirement",
            "tool.requirement",
            "operation.requirement",
        ],
    )
    .as_deref()
    .map(parse_tool_requirement)
}

pub(super) fn approval_outcome(span: &Span) -> Option<ApprovalOutcome> {
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

pub(super) fn structured_result_bool(span: &Span) -> Option<bool> {
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

pub(super) fn state_delta(span: &Span) -> Option<Value> {
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

pub(super) fn is_empty_state_delta(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Object(object) => object.is_empty(),
        Value::Array(array) => array.is_empty(),
        Value::String(value) => value.trim().is_empty() || value.trim() == "{}",
        _ => false,
    }
}

pub(super) fn has_exception_event(span: &Span) -> bool {
    span.attributes.contains_key("exception.type")
        || span.attributes.contains_key("exception.message")
        || bool_attribute(span, &["exception.recorded"]) == Some(true)
        || span
            .events
            .iter()
            .any(|event| event.name.eq_ignore_ascii_case("exception"))
}

pub(super) fn protocol_status(span: &Span) -> Option<u16> {
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

pub(super) fn duration_ms(span: &Span) -> u64 {
    if let Some(duration) = duration_nano(span) {
        return duration.saturating_add(500_000) / 1_000_000;
    }
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

pub(super) fn duration_nano(span: &Span) -> Option<u64> {
    span.duration_nano.or_else(|| {
        span.start_time_unix_nano
            .zip(span.end_time_unix_nano)
            .map(|(start, end)| end.saturating_sub(start))
    })
}

pub(super) fn string_attribute(span: &Span, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        span.attributes.get(*key).and_then(|value| match value {
            Value::String(value) => Some(value.clone()),
            Value::Number(value) => Some(value.to_string()),
            _ => None,
        })
    })
}

pub(super) fn bool_attribute(span: &Span, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| span.attributes.get(*key).and_then(value_as_bool))
}

pub(super) fn value_as_bool(value: &Value) -> Option<bool> {
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

pub(super) fn u32_attribute(span: &Span, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|key| {
        span.attributes.get(*key).and_then(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        })
    })
}

pub(super) fn classify_error_kind(message: &str) -> &'static str {
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

pub(super) fn parse_final_status(value: &str) -> Option<FinalOutcomeStatus> {
    Some(match value.to_ascii_lowercase().as_str() {
        "complete" | "completed" => FinalOutcomeStatus::Completed,
        "failure" | "failed" | "error" => FinalOutcomeStatus::Failed,
        "safely_refused" | "safe_refusal" => FinalOutcomeStatus::SafelyRefused,
        "escalated" | "escalation" => FinalOutcomeStatus::Escalated,
        "incomplete" | "missing" => FinalOutcomeStatus::Incomplete,
        "unknown" | "uncertain" => FinalOutcomeStatus::Unknown,
        _ => return None,
    })
}

pub(super) fn parse_claimed_outcome_status(value: &str) -> Option<ClaimedOutcomeStatus> {
    Some(match value.to_ascii_lowercase().as_str() {
        "success" | "succeeded" | "completed" => ClaimedOutcomeStatus::Succeeded,
        "failure" | "failed" | "error" => ClaimedOutcomeStatus::Failed,
        "not_completed" | "not-completed" => ClaimedOutcomeStatus::NotCompleted,
        "state_unknown" | "state-unknown" | "unknown" => ClaimedOutcomeStatus::StateUnknown,
        _ => return None,
    })
}

pub(super) fn parse_escalation_status(value: &str) -> EscalationStatus {
    match value.to_ascii_lowercase().as_str() {
        "not_required" | "not-required" => EscalationStatus::NotRequired,
        "required_and_performed" | "required-performed" => EscalationStatus::RequiredAndPerformed,
        "required_and_missing" | "required-missing" => EscalationStatus::RequiredAndMissing,
        _ => EscalationStatus::Unknown,
    }
}

pub(super) fn parse_operation_effect(value: &str) -> OperationEffect {
    match value.to_ascii_lowercase().as_str() {
        "read_only" | "read-only" | "readonly" => OperationEffect::ReadOnly,
        "mutating" | "mutation" => OperationEffect::Mutating,
        "verifying" | "verification" => OperationEffect::Verifying,
        "compensating" | "compensation" => OperationEffect::Compensating,
        "escalating" | "escalation" => OperationEffect::Escalating,
        _ => OperationEffect::Unknown,
    }
}

pub(super) fn parse_retry_safety(value: &str) -> RetrySafety {
    match value.to_ascii_lowercase().as_str() {
        "idempotent" => RetrySafety::Idempotent,
        "non_idempotent" | "non-idempotent" | "nonidempotent" => RetrySafety::NonIdempotent,
        _ => RetrySafety::Unknown,
    }
}

pub(super) fn parse_tool_requirement(value: &str) -> ToolRequirement {
    match value.to_ascii_lowercase().as_str() {
        "required" => ToolRequirement::Required,
        "optional" => ToolRequirement::Optional,
        _ => ToolRequirement::Unknown,
    }
}

pub(super) fn parse_state_observation(value: &str) -> StateObservation {
    match value.to_ascii_lowercase().as_str() {
        "verified_changed" | "verified-changed" => StateObservation::VerifiedChanged,
        "verified_unchanged" | "verified-unchanged" => StateObservation::VerifiedUnchanged,
        "unverified" => StateObservation::Unverified,
        "ambiguous" => StateObservation::Ambiguous,
        "conflicting" => StateObservation::Conflicting,
        _ => StateObservation::Unknown,
    }
}

pub(super) fn is_valid_identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value.chars().all(|character| character.is_ascii_graphic())
}

pub(super) fn bounded_text(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    value.chars().take(max_chars).collect()
}

pub(super) fn canonical_json(value: &Value) -> String {
    serde_json::to_string(&canonicalize_value(value)).unwrap_or_default()
}

pub(super) fn canonicalize_value(value: &Value) -> Value {
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

pub(super) fn hash_text(value: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(value.as_bytes()))
}
