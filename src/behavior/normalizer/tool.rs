use super::attributes::*;
use super::*;

pub(super) struct ToolCallNormalizationContext {
    pub tool_name: String,
    pub tool_name_source_quality: FactQuality,
    pub operation: Option<String>,
    pub operation_source_quality: FactQuality,
    pub mapping: ToolSemanticMapping,
    pub inferred_attempt: u32,
    pub privacy: BehaviorInputPrivacyV1,
}

pub(super) fn normalize_tool_call(
    span: &Span,
    context: ToolCallNormalizationContext,
) -> ToolCallFact {
    let ToolCallNormalizationContext {
        tool_name,
        tool_name_source_quality,
        operation,
        operation_source_quality,
        mapping,
        inferred_attempt,
        privacy,
    } = context;
    let (status, status_quality) = tool_status(span);
    let error = normalized_error(span, status);
    let state_change = state_change(span);
    let mut evidence = vec![EvidenceRef::span(&span.id)];
    evidence.extend(span.events.iter().map(|event| EvidenceRef {
        kind: "span_event".into(),
        identity: event.identity.clone(),
        span_id: Some(span.id.clone()),
    }));
    evidence.extend(span.links.iter().map(|link| EvidenceRef {
        kind: "span_link".into(),
        identity: link.identity.clone(),
        span_id: Some(span.id.clone()),
    }));
    if let Some(state_change) = &state_change {
        evidence.push(state_change.artifact.clone());
    }
    let effect = explicit_effect(span).unwrap_or(mapping.effect);
    let retry_safety = explicit_retry_safety(span).unwrap_or(mapping.retry_safety);
    let requirement = explicit_requirement(span).unwrap_or(mapping.requirement);
    let (invocation_fingerprint, invocation_fingerprint_quality) =
        invocation_fingerprint(span, privacy);
    let (result_fingerprint, result_fingerprint_quality) = result_fingerprint(span, privacy);
    let exact_duration_nano = duration_nano(span);
    let legacy_duration_ms = duration_ms(span);

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
        tool_name_source_quality,
        operation,
        operation_source_quality,
        invocation_fingerprint,
        invocation_fingerprint_quality,
        result_fingerprint,
        result_fingerprint_quality,
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
        started_at_unix_nano: span.start_time_unix_nano,
        duration_ms: legacy_duration_ms,
        duration_nano: exact_duration_nano.or_else(|| {
            (legacy_duration_ms > 0).then(|| legacy_duration_ms.saturating_mul(1_000_000))
        }),
        status,
        status_quality,
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

fn tool_status(span: &Span) -> (ToolCallStatus, FactQuality) {
    // Failure precedence follows the architecture contract. Success is only
    // inferred after all bounded failure signals have been considered.
    if structured_result_bool(span) == Some(false) {
        return (
            ToolCallStatus::Failed,
            if span.source_status == SourceSpanStatus::Ok {
                FactQuality::Ambiguous
            } else {
                FactQuality::Explicit
            },
        );
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
        return (ToolCallStatus::TimedOut, FactQuality::Explicit);
    }
    if bool_attribute(span, &["tool.cancelled", "execution.cancelled"]) == Some(true)
        || status_text.contains("cancel")
        || error_text.contains("cancel")
    {
        return (ToolCallStatus::Cancelled, FactQuality::Explicit);
    }
    if span.source_status == SourceSpanStatus::Error {
        return (ToolCallStatus::Failed, FactQuality::Explicit);
    }
    if has_exception_event(span) {
        return (ToolCallStatus::Failed, FactQuality::Explicit);
    }
    if span.error.is_some() || matches!(status_text.as_str(), "failed" | "failure" | "error") {
        return (ToolCallStatus::Failed, FactQuality::Explicit);
    }
    if protocol_status(span).is_some_and(|status| status >= 400) {
        return (ToolCallStatus::Failed, FactQuality::Explicit);
    }
    if matches!(
        status_text.as_str(),
        "unknown" | "uncertain" | "ambiguous" | "incomplete"
    ) {
        return (ToolCallStatus::Unknown, FactQuality::Explicit);
    }

    if span.source_status == SourceSpanStatus::Ok {
        return (ToolCallStatus::Succeeded, FactQuality::Explicit);
    }
    if structured_result_bool(span) == Some(true)
        || matches!(
            status_text.as_str(),
            "succeeded" | "success" | "ok" | "completed"
        )
    {
        (ToolCallStatus::Succeeded, FactQuality::Explicit)
    } else {
        (ToolCallStatus::Unknown, FactQuality::Missing)
    }
}

fn invocation_fingerprint(
    span: &Span,
    privacy: BehaviorInputPrivacyV1,
) -> (Option<String>, FactQuality) {
    fingerprint(
        span,
        &[
            "traceeval.tool.invocation",
            "gen_ai.tool.call.arguments",
            "tool.arguments",
            "tool.parameters",
            "input.value",
            "input",
        ],
        span.input.as_deref(),
        privacy,
    )
}

fn result_fingerprint(
    span: &Span,
    privacy: BehaviorInputPrivacyV1,
) -> (Option<String>, FactQuality) {
    fingerprint(
        span,
        &[
            "traceeval.tool.result",
            "gen_ai.tool.call.result",
            "tool.result",
            "output.value",
            "output",
        ],
        span.output.as_deref(),
        privacy,
    )
}

fn fingerprint(
    span: &Span,
    keys: &[&str],
    legacy_body: Option<&str>,
    privacy: BehaviorInputPrivacyV1,
) -> (Option<String>, FactQuality) {
    let identities = keys
        .iter()
        .filter_map(|key| {
            span.payload_identities
                .get(*key)
                .map(|identity| ((*key).to_string(), identity))
        })
        .collect::<Vec<_>>();
    if !identities.is_empty() {
        let quality = identities
            .iter()
            .fold(FactQuality::Explicit, |quality, (_, identity)| {
                weaker_quality(quality, identity.quality)
            });
        if identities.len() == 1 {
            return (Some(identities[0].1.fingerprint.clone()), quality);
        }
        let value = Value::Array(
            identities
                .iter()
                .map(|(key, identity)| {
                    serde_json::json!({
                        "key": key,
                        "fingerprint": identity.fingerprint,
                        "bytes": identity.original_bytes,
                    })
                })
                .collect(),
        );
        return (Some(hash_text(&canonical_json(&value))), quality);
    }

    if privacy == BehaviorInputPrivacyV1::SafeIdentitiesOnly {
        return (None, FactQuality::Missing);
    }
    let values = keys
        .iter()
        .filter_map(|key| {
            span.attributes
                .get(*key)
                .map(|value| ((*key).to_string(), value))
        })
        .collect::<Vec<_>>();
    if !values.is_empty() {
        let value = Value::Array(
            values
                .iter()
                .map(|(key, value)| serde_json::json!({"key": key, "value": value}))
                .collect(),
        );
        return (
            Some(hash_text(&canonical_json(&value))),
            FactQuality::Derived,
        );
    }
    legacy_body
        .map(|body| {
            let canonical = serde_json::from_str(body)
                .map(|value| canonical_json(&value))
                .unwrap_or_else(|_| body.to_string());
            (Some(hash_text(&canonical)), FactQuality::Derived)
        })
        .unwrap_or((None, FactQuality::Missing))
}

fn weaker_quality(left: FactQuality, right: FactQuality) -> FactQuality {
    use FactQuality::{Ambiguous, Derived, Explicit, Inferred, Missing};
    match (left, right) {
        (Missing, _) | (_, Missing) => Missing,
        (Ambiguous, _) | (_, Ambiguous) => Ambiguous,
        (Inferred, _) | (_, Inferred) => Inferred,
        (Derived, _) | (_, Derived) => Derived,
        (Explicit, Explicit) => Explicit,
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
