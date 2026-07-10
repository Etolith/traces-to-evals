use super::attributes::*;
use super::*;

pub(super) fn normalize_tool_call(
    span: &Span,
    tool_name: String,
    operation: Option<String>,
    mapping: ToolSemanticMapping,
    inferred_attempt: u32,
) -> ToolCallFact {
    let status = tool_status(span);
    let error = normalized_error(span, status);
    let state_change = state_change(span);
    let mut evidence = vec![EvidenceRef::span(&span.id)];
    if let Some(state_change) = &state_change {
        evidence.push(state_change.artifact.clone());
    }
    let effect = explicit_effect(span).unwrap_or(mapping.effect);
    let retry_safety = explicit_retry_safety(span).unwrap_or(mapping.retry_safety);
    let requirement = explicit_requirement(span).unwrap_or(mapping.requirement);

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
        .filter(|call_id| is_valid_identity(call_id))
        .unwrap_or_else(|| span.id.clone()),
        tool_name,
        operation,
        effect,
        retry_safety,
        requirement,
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
            &["agent.approval.required", "tool.approval.required"],
        )
        .unwrap_or(false),
        approval_outcome: approval_outcome(span),
        state_change,
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

fn state_change(span: &Span) -> Option<StateChangeRef> {
    let observation = string_attribute(
        span,
        &[
            "agent.state.observation",
            "tool.state.observation",
            "state.observation",
        ],
    )
    .as_deref()
    .map(parse_state_observation);
    let value = state_delta(span);
    if observation.is_none() && value.as_ref().is_none_or(is_empty_state_delta) {
        return None;
    }
    let artifact_identity = string_attribute(
        span,
        &[
            "agent.state.artifact.id",
            "tool.state.artifact.id",
            "state.artifact.id",
        ],
    )
    .map(|identity| format!("state_artifact:{identity}"))
    .unwrap_or_else(|| {
        value
            .as_ref()
            .map(canonical_json)
            .map(|serialized| format!("state_change:{}:{}", span.id, hash_text(&serialized)))
            .unwrap_or_else(|| format!("state_change:{}", span.id))
    });
    let artifact = EvidenceRef {
        kind: "state_change".to_string(),
        identity: artifact_identity,
        span_id: Some(span.id.clone()),
    };

    Some(StateChangeRef {
        predicate: string_attribute(
            span,
            &[
                "agent.state.predicate",
                "tool.state.predicate",
                "state.predicate",
            ],
        )
        .filter(|predicate| is_valid_semantic_label(predicate)),
        observation: observation.unwrap_or(StateObservation::Unverified),
        artifact,
    })
}
