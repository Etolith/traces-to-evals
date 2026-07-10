use super::attributes::*;
use super::*;

pub(super) fn normalize_policy_decision(span: &Span) -> Option<PolicyDecision> {
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

pub(super) fn normalize_final_outcome(root: Option<&Span>) -> FinalOutcome {
    let Some(root) = root else {
        return FinalOutcome::default();
    };
    let status = string_attribute(root, &["agent.final.status", "final.status"])
        .as_deref()
        .and_then(parse_final_status)
        .unwrap_or_else(|| {
            if root.error.is_some() {
                FinalOutcomeStatus::Failed
            } else {
                FinalOutcomeStatus::Unknown
            }
        });
    let escalation = string_attribute(
        root,
        &["agent.escalation.status", "final.escalation.status"],
    )
    .as_deref()
    .map(parse_escalation_status)
    .unwrap_or(EscalationStatus::Unknown);

    FinalOutcome {
        status,
        claims: outcome_claims(root),
        escalation,
        evidence: vec![EvidenceRef::span(&root.id)],
    }
}

fn outcome_claims(span: &Span) -> Vec<OutcomeClaim> {
    let value = span
        .attributes
        .get("agent.outcome.claims")
        .or_else(|| span.attributes.get("final.outcome.claims"));
    let parsed = value.and_then(|value| match value {
        Value::String(value) => serde_json::from_str::<Value>(value).ok(),
        value => Some(value.clone()),
    });
    let mut claims = parsed
        .as_ref()
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, value)| outcome_claim(span, index, value))
        .collect::<Vec<_>>();

    if claims.is_empty() {
        let status = string_attribute(
            span,
            &["agent.outcome.claim.status", "final.outcome.claim.status"],
        )
        .as_deref()
        .and_then(parse_claimed_outcome_status);
        if let Some(status) = status {
            claims.push(OutcomeClaim {
                operation: string_attribute(
                    span,
                    &[
                        "agent.outcome.claim.operation",
                        "final.outcome.claim.operation",
                    ],
                )
                .filter(|operation| is_valid_semantic_label(operation)),
                call_id: string_attribute(
                    span,
                    &["agent.outcome.claim.call_id", "final.outcome.claim.call_id"],
                )
                .filter(|call_id| is_valid_identity(call_id)),
                status,
                evidence: vec![claim_evidence(span, 0)],
            });
        }
    }
    claims
}

fn outcome_claim(span: &Span, index: usize, value: &Value) -> Option<OutcomeClaim> {
    let object = value.as_object()?;
    let status = object
        .get("status")
        .and_then(Value::as_str)
        .and_then(parse_claimed_outcome_status)?;
    Some(OutcomeClaim {
        operation: object
            .get("operation")
            .and_then(Value::as_str)
            .filter(|operation| is_valid_semantic_label(operation))
            .map(str::to_string),
        call_id: object
            .get("call_id")
            .and_then(Value::as_str)
            .filter(|call_id| is_valid_identity(call_id))
            .map(str::to_string),
        status,
        evidence: vec![claim_evidence(span, index)],
    })
}

fn claim_evidence(span: &Span, index: usize) -> EvidenceRef {
    EvidenceRef {
        kind: "outcome_claim".to_string(),
        identity: format!("outcome_claim:{}:{index}", span.id),
        span_id: Some(span.id.clone()),
    }
}
