use super::*;

#[derive(Debug, Default, Clone, Copy)]
pub struct RecoveryAnalyzer;

impl RecoveryAnalyzer {
    pub fn recovery_for_call(
        &self,
        trace: &AgentBehaviorTrace,
        call_index: usize,
    ) -> RecoveryStatus {
        let Some(call) = trace.tool_calls.get(call_index) else {
            return RecoveryStatus::Unknown;
        };
        if call.status == ToolCallStatus::Succeeded {
            return RecoveryStatus::Recovered;
        }

        let later_calls = &trace.tool_calls[call_index.saturating_add(1)..];
        let equivalent_success = later_calls.iter().any(|later| {
            retry_matches(call, later)
                && later.status == ToolCallStatus::Succeeded
                && call.retry_safety == RetrySafety::Idempotent
        });
        if equivalent_success {
            return RecoveryStatus::Recovered;
        }

        if later_calls
            .iter()
            .any(|later| reaches_required_state(call, later))
        {
            return RecoveryStatus::Recovered;
        }

        if call.effect == OperationEffect::Mutating
            && later_calls
                .iter()
                .any(|later| verifies_call_state(call, later))
        {
            return RecoveryStatus::Recovered;
        }

        if later_calls
            .iter()
            .any(|later| compensates_call(call, later))
        {
            return RecoveryStatus::Recovered;
        }

        if later_calls.iter().any(|later| {
            later.status == ToolCallStatus::Succeeded
                && later.tool_name == call.tool_name
                && later.operation == call.operation
                && later.effect == call.effect
        }) {
            // A later operation-level success is suggestive, but without a compatible
            // invocation identity or linked state proof it cannot clear this failure.
            return RecoveryStatus::Unknown;
        }

        let claims = trace
            .final_outcome
            .claims
            .iter()
            .filter(|claim| claim_matches_call(claim, call))
            .collect::<Vec<_>>();
        if claims
            .iter()
            .any(|claim| claim.status == ClaimedOutcomeStatus::Succeeded)
        {
            return RecoveryStatus::Unrecovered;
        }
        if claims.iter().any(|claim| {
            matches!(
                claim.status,
                ClaimedOutcomeStatus::Failed | ClaimedOutcomeStatus::NotCompleted
            )
        }) {
            return RecoveryStatus::Recovered;
        }
        if claims
            .iter()
            .any(|claim| claim.status == ClaimedOutcomeStatus::StateUnknown)
        {
            return RecoveryStatus::Unknown;
        }

        if trace.final_outcome.escalation == EscalationStatus::RequiredAndPerformed
            && trace.final_outcome.status == FinalOutcomeStatus::Escalated
        {
            return RecoveryStatus::Recovered;
        }

        if call.status == ToolCallStatus::Unknown {
            RecoveryStatus::Unknown
        } else {
            RecoveryStatus::Unrecovered
        }
    }
}

fn retry_matches(left: &ToolCallFact, right: &ToolCallFact) -> bool {
    left.operation == right.operation
        && left.tool_name == right.tool_name
        && left.invocation_fingerprint.is_some()
        && left.invocation_fingerprint == right.invocation_fingerprint
        && left.effect == right.effect
}

pub(super) fn equivalent_call_key(call: &ToolCallFact) -> Option<(String, String, String)> {
    if !matches!(
        call.invocation_fingerprint_quality,
        FactQuality::Explicit | FactQuality::Derived
    ) {
        return None;
    }
    let compatible_identity = if call.operation.is_some()
        && matches!(
            call.operation_source_quality,
            FactQuality::Explicit | FactQuality::Derived
        ) {
        format!("operation:{}", call.operation.as_ref()?)
    } else if matches!(
        call.tool_name_source_quality,
        FactQuality::Explicit | FactQuality::Derived
    ) {
        format!("tool:{}", call.tool_name)
    } else {
        return None;
    };
    Some((
        call.tool_name.clone(),
        compatible_identity,
        call.invocation_fingerprint.clone()?,
    ))
}

pub(super) fn has_material_progress(call: &ToolCallFact) -> bool {
    call.status == ToolCallStatus::Succeeded
        && (matches!(
            call.effect,
            OperationEffect::Verifying
                | OperationEffect::Compensating
                | OperationEffect::Escalating
        ) || call
            .state_change
            .as_ref()
            .is_some_and(|state| state.observation.is_verified()))
}

pub(super) fn claim_matches_call(claim: &OutcomeClaim, call: &ToolCallFact) -> bool {
    if let Some(call_id) = &claim.call_id {
        return call_id == &call.call_id;
    }
    if let Some(operation) = &claim.operation {
        return call.operation.as_ref() == Some(operation);
    }
    false
}

fn verifies_call_state(call: &ToolCallFact, later: &ToolCallFact) -> bool {
    if later.status != ToolCallStatus::Succeeded
        || later.effect != OperationEffect::Verifying
        || !later
            .state_change
            .as_ref()
            .is_some_and(|state| state.observation.is_verified())
    {
        return false;
    }
    match (
        call.state_change
            .as_ref()
            .and_then(|state| state.predicate.as_ref()),
        later
            .state_change
            .as_ref()
            .and_then(|state| state.predicate.as_ref()),
    ) {
        (Some(expected), Some(observed)) => expected == observed,
        _ => false,
    }
}

fn compensates_call(call: &ToolCallFact, later: &ToolCallFact) -> bool {
    if later.effect != OperationEffect::Compensating || later.status != ToolCallStatus::Succeeded {
        return false;
    }
    match (
        call.state_change
            .as_ref()
            .and_then(|state| state.predicate.as_ref()),
        later
            .state_change
            .as_ref()
            .and_then(|state| state.predicate.as_ref()),
    ) {
        (Some(expected), Some(compensated)) => expected == compensated,
        _ => false,
    }
}

fn reaches_required_state(call: &ToolCallFact, later: &ToolCallFact) -> bool {
    let Some(expected_predicate) = call
        .state_change
        .as_ref()
        .and_then(|state| state.predicate.as_ref())
    else {
        return false;
    };
    later.status == ToolCallStatus::Succeeded
        && later.state_change.as_ref().is_some_and(|state| {
            state.predicate.as_ref() == Some(expected_predicate)
                && state.observation == StateObservation::VerifiedChanged
        })
}

pub(super) fn claim_has_success_evidence(trace: &AgentBehaviorTrace, claim: &OutcomeClaim) -> bool {
    let matching_calls = trace
        .tool_calls
        .iter()
        .enumerate()
        .filter(|(_, call)| claim_matches_call(claim, call))
        .collect::<Vec<_>>();
    if matching_calls
        .iter()
        .any(|(_, call)| call.status == ToolCallStatus::Succeeded)
    {
        return true;
    }
    matching_calls.iter().any(|(index, call)| {
        trace.tool_calls[index.saturating_add(1)..]
            .iter()
            .any(|later| {
                verifies_call_state(call, later)
                    && later
                        .state_change
                        .as_ref()
                        .is_some_and(|state| state.observation == StateObservation::VerifiedChanged)
            })
    })
}
