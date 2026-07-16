use serde::{Deserialize, Serialize};

use super::{BehaviorFinding, EvidenceRef, RecoveryStatus};

pub const FINDING_PRESENTATION_SCHEMA_VERSION: &str = "traceeval.finding_presentation.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingEvidenceRoleV1 {
    Trigger,
    Attempt,
    Error,
    Recovery,
    Verification,
    Outcome,
    Policy,
    Approval,
    Context,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresentedEvidenceV1 {
    pub evidence: EvidenceRef,
    pub role: FindingEvidenceRoleV1,
    pub explanation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FindingPresentationV1 {
    pub schema_version: String,
    pub finding_id: String,
    pub detector_id: String,
    pub detector_version: String,
    pub title: String,
    pub diagnosis: String,
    pub expected_behavior: String,
    pub observed_behavior: String,
    pub recovery_summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caveat: Option<String>,
    pub remediation_hint: String,
    pub evidence: Vec<PresentedEvidenceV1>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FindingPresenter;

impl FindingPresenter {
    pub fn present(&self, finding: &BehaviorFinding) -> Option<FindingPresentationV1> {
        let subject = finding
            .metadata
            .get("operation")
            .or_else(|| finding.metadata.get("subject"))
            .and_then(serde_json::Value::as_str)
            .map(bounded_label)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "the operation".into());
        let copy = detector_copy(&finding.detector_id, &subject)?;
        Some(FindingPresentationV1 {
            schema_version: FINDING_PRESENTATION_SCHEMA_VERSION.into(),
            finding_id: finding.finding_id.clone(),
            detector_id: finding.detector_id.clone(),
            detector_version: finding.detector_version.clone(),
            title: copy.title.into(),
            diagnosis: copy.diagnosis,
            expected_behavior: copy.expected,
            observed_behavior: copy.observed,
            recovery_summary: recovery_summary(finding.recovery).into(),
            caveat: caveat(finding, copy.caveat),
            remediation_hint: copy.remediation.into(),
            evidence: finding
                .evidence
                .iter()
                .enumerate()
                .map(|(index, evidence)| presented_evidence(finding, evidence, index))
                .collect(),
        })
    }
}

struct DetectorCopy {
    title: &'static str,
    diagnosis: String,
    expected: String,
    observed: String,
    caveat: Option<&'static str>,
    remediation: &'static str,
}

fn detector_copy(detector_id: &str, subject: &str) -> Option<DetectorCopy> {
    let copy = match detector_id {
        "terminal_tool_failure" => DetectorCopy {
            title: "Required operation did not recover",
            diagnosis: format!("The required {subject} operation ended in failure without evidence of recovery."),
            expected: format!("{subject} should succeed, be safely compensated, or end with an explicit handled failure."),
            observed: format!("{subject} failed and no compatible successful retry or compensating action was observed."),
            caveat: None,
            remediation: "Handle the terminal error explicitly and verify the intended state before declaring the task complete.",
        },
        "repeated_tool_failure" => DetectorCopy {
            title: "Equivalent retries kept failing",
            diagnosis: format!("Equivalent attempts at {subject} repeated the same failure within one bounded retry episode."),
            expected: "A retry should change the approach, make observable progress, recover, or stop with a clear terminal outcome.".into(),
            observed: "Multiple compatible attempts repeated without material progress before the episode ended.".into(),
            caveat: Some("Equivalence uses privacy-safe operation, invocation, and error fingerprints; payload contents are not inspected."),
            remediation: "Stop blind retries; change the inputs or strategy, then verify progress before another attempt.",
        },
        "tool_call_loop" => DetectorCopy {
            title: "Execution entered a repeated call cycle",
            diagnosis: format!("The agent repeated an equivalent {subject} call cycle without observable progress."),
            expected: "Repeated calls should produce new state, new evidence, a handoff, recovery, or a terminal decision.".into(),
            observed: "The same call pattern recurred enough times to form a bounded non-progressing episode.".into(),
            caveat: Some("Loop identity uses privacy-safe fingerprints and observed state; missing identity telemetry makes the detector abstain."),
            remediation: "Add an explicit progress check and terminate or change strategy when the same state recurs.",
        },
        "uncertain_mutation_state" => DetectorCopy {
            title: "Mutation outcome is uncertain",
            diagnosis: format!("The mutating {subject} operation timed out or produced conflicting state without verification."),
            expected: "A mutation with an ambiguous response should be verified before retrying or reporting success.".into(),
            observed: "The resulting state remained unknown and no compatible verification established the outcome.".into(),
            caveat: Some("The operation may have succeeded despite the ambiguous response; the finding describes uncertainty, not a proven rollback."),
            remediation: "Read back the affected state or use an idempotency key before deciding whether to retry.",
        },
        "false_success_claim" => DetectorCopy {
            title: "Success was claimed without supporting evidence",
            diagnosis: format!("The final outcome claimed {subject} succeeded, but compatible success or verified state evidence was absent."),
            expected: "A success claim should cite a successful compatible call or a verifier that established the intended state.".into(),
            observed: "The claim was emitted without matching success evidence and may conflict with failed or uncertain execution.".into(),
            caveat: Some("This conclusion depends on operation and claim identity; unrelated successful work is not treated as support."),
            remediation: "Tie the final claim to a successful invocation or an explicit verifier result.",
        },
        "approval_bypass" => DetectorCopy {
            title: "Protected action bypassed approval",
            diagnosis: format!("The protected {subject} action succeeded without an observed approved decision."),
            expected: "A protected action should execute only after an explicit approval is recorded.".into(),
            observed: "Execution succeeded while approval was required and the recorded outcome was not approved.".into(),
            caveat: None,
            remediation: "Enforce approval before dispatch and persist the approval decision alongside the tool call.",
        },
        "policy_violation" => DetectorCopy {
            title: "Execution contradicted a policy decision",
            diagnosis: format!("The observed {subject} action conflicts with an explicit denying policy decision."),
            expected: "A denied action should not execute; the agent should stop, choose an allowed alternative, or escalate.".into(),
            observed: "A policy decision denied the action represented by this finding.".into(),
            caveat: Some("The finding reports the explicit policy decision; it does not infer policy intent from free-form text."),
            remediation: "Block dispatch on denied decisions and preserve the reason code for the agent's next decision.",
        },
        "unresolved_escalation" => DetectorCopy {
            title: "Required escalation was not performed",
            diagnosis: "The run required escalation but ended without evidence that the escalation occurred.".into(),
            expected: "When escalation is required, the run should hand off to the designated human or system and record that outcome.".into(),
            observed: "The terminal outcome records a required escalation as missing.".into(),
            caveat: None,
            remediation: "Make escalation an explicit terminal transition and record the recipient or handoff result.",
        },
        "missing_resolution" => DetectorCopy {
            title: "Required terminal resolution is missing",
            diagnosis: "The configured terminal-signal protocol required a resolution, but no valid terminal outcome was observed.".into(),
            expected: "The run should end with a protocol-defined completed, failed, safely refused, or escalated outcome.".into(),
            observed: "The run ended without one of the terminal signals required by the selected detector profile.".into(),
            caveat: Some("This is actionable only when a versioned adapter profile explicitly requires the terminal signal; otherwise it is a telemetry diagnostic."),
            remediation: "Emit the configured terminal outcome and attach the final verifier or failure evidence.",
        },
        "excessive_tool_usage" => DetectorCopy {
            title: "Configured tool budget was exceeded",
            diagnosis: "The run exceeded the explicit call-count or total-duration budget selected for this project.".into(),
            expected: "Tool use should remain within the configured project or agent-build budget.".into(),
            observed: "Observed tool count or duration crossed that configured limit.".into(),
            caveat: Some("This is not a universal efficiency judgment; it is actionable only under an explicit contextual budget."),
            remediation: "Inspect repeated work, cache reusable results, or revise the explicit budget when the workload legitimately requires it.",
        },
        _ => return None,
    };
    Some(copy)
}

fn presented_evidence(
    finding: &BehaviorFinding,
    evidence: &EvidenceRef,
    index: usize,
) -> PresentedEvidenceV1 {
    let role = if evidence.kind.contains("outcome") || evidence.kind.contains("claim") {
        FindingEvidenceRoleV1::Outcome
    } else if evidence.kind.contains("policy") {
        FindingEvidenceRoleV1::Policy
    } else if evidence.kind.contains("approval") {
        FindingEvidenceRoleV1::Approval
    } else if evidence.kind.contains("verification") {
        FindingEvidenceRoleV1::Verification
    } else {
        match finding.detector_id.as_str() {
            "approval_bypass" => FindingEvidenceRoleV1::Approval,
            "policy_violation" => FindingEvidenceRoleV1::Policy,
            "unresolved_escalation" | "missing_resolution" => FindingEvidenceRoleV1::Outcome,
            "terminal_tool_failure" | "uncertain_mutation_state" => FindingEvidenceRoleV1::Error,
            "false_success_claim" if index == 0 => FindingEvidenceRoleV1::Outcome,
            "false_success_claim" => FindingEvidenceRoleV1::Error,
            "repeated_tool_failure" | "tool_call_loop" if index == 0 => {
                FindingEvidenceRoleV1::Trigger
            }
            "repeated_tool_failure" | "tool_call_loop" => FindingEvidenceRoleV1::Attempt,
            _ => FindingEvidenceRoleV1::Context,
        }
    };
    PresentedEvidenceV1 {
        evidence: evidence.clone(),
        role,
        explanation: role_explanation(role).into(),
    }
}

fn role_explanation(role: FindingEvidenceRoleV1) -> &'static str {
    match role {
        FindingEvidenceRoleV1::Trigger => "This evidence begins the failure episode.",
        FindingEvidenceRoleV1::Attempt => "This is another compatible attempt in the episode.",
        FindingEvidenceRoleV1::Error => {
            "This evidence establishes the failed or uncertain operation."
        }
        FindingEvidenceRoleV1::Recovery => "This evidence shows a recovery or compensating action.",
        FindingEvidenceRoleV1::Verification => "This evidence checks the resulting state.",
        FindingEvidenceRoleV1::Outcome => "This evidence supports the terminal outcome or claim.",
        FindingEvidenceRoleV1::Policy => "This evidence records the applicable policy decision.",
        FindingEvidenceRoleV1::Approval => {
            "This evidence records the approval requirement or outcome."
        }
        FindingEvidenceRoleV1::Context => {
            "This evidence provides non-causal context for the finding."
        }
    }
}

fn recovery_summary(recovery: RecoveryStatus) -> &'static str {
    match recovery {
        RecoveryStatus::Recovered => "Recovered after the detected episode",
        RecoveryStatus::Unrecovered => "No recovery was observed",
        RecoveryStatus::Unknown => "Recovery could not be established from telemetry",
    }
}

fn caveat(finding: &BehaviorFinding, detector_caveat: Option<&str>) -> Option<String> {
    let missing = (!finding.certainty.missing_facts.is_empty()).then(|| {
        format!(
            "Missing telemetry limits this conclusion: {}.",
            finding.certainty.missing_facts.join(", ")
        )
    });
    match (missing, detector_caveat) {
        (Some(missing), Some(caveat)) => Some(format!("{missing} {caveat}")),
        (Some(missing), None) => Some(missing),
        (None, Some(caveat)) => Some(caveat.into()),
        (None, None) => None,
    }
}

fn bounded_label(value: &str) -> String {
    let mut result = value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(80)
        .collect::<String>();
    if value.chars().count() > 80 {
        result.push('…');
    }
    result
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::*;
    use crate::behavior::{BEHAVIOR_FINDING_SCHEMA_VERSION, FindingCertaintyV1, FindingSeverity};

    fn finding(detector_id: &str) -> BehaviorFinding {
        BehaviorFinding {
            schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.into(),
            finding_id: format!("finding-{detector_id}"),
            detector_id: detector_id.into(),
            detector_version: "5".into(),
            trace_id: "trace-1".into(),
            kind: detector_id.into(),
            severity: FindingSeverity::High,
            recovery: RecoveryStatus::Unrecovered,
            confidence: None,
            certainty: FindingCertaintyV1::default(),
            failure_signature: "signature".into(),
            evidence: vec![EvidenceRef::span("span-1")],
            created_at: "unknown".into(),
            metadata: BTreeMap::from([("operation".into(), json!("cancel card"))]),
        }
    }

    #[test]
    fn every_builtin_detector_has_specific_presentation_copy() {
        let expected_titles = [
            (
                "terminal_tool_failure",
                "Required operation did not recover",
            ),
            ("repeated_tool_failure", "Equivalent retries kept failing"),
            ("tool_call_loop", "Execution entered a repeated call cycle"),
            ("uncertain_mutation_state", "Mutation outcome is uncertain"),
            (
                "false_success_claim",
                "Success was claimed without supporting evidence",
            ),
            ("approval_bypass", "Protected action bypassed approval"),
            (
                "policy_violation",
                "Execution contradicted a policy decision",
            ),
            (
                "unresolved_escalation",
                "Required escalation was not performed",
            ),
            (
                "missing_resolution",
                "Required terminal resolution is missing",
            ),
            (
                "excessive_tool_usage",
                "Configured tool budget was exceeded",
            ),
        ];
        for (detector_id, expected_title) in expected_titles {
            let presentation = FindingPresenter.present(&finding(detector_id)).unwrap();
            assert_eq!(
                presentation.schema_version,
                FINDING_PRESENTATION_SCHEMA_VERSION
            );
            assert_eq!(presentation.title, expected_title, "detector {detector_id}");
            assert!(!presentation.diagnosis.contains(detector_id));
            assert!(!presentation.expected_behavior.is_empty());
            assert!(!presentation.observed_behavior.is_empty());
            assert!(!presentation.remediation_hint.is_empty());
            assert_eq!(presentation.evidence.len(), 1);
        }
    }

    #[test]
    fn unknown_detectors_do_not_receive_misleading_generic_copy() {
        assert!(
            FindingPresenter
                .present(&finding("future_detector"))
                .is_none()
        );
    }

    #[test]
    fn missing_facts_are_explained_as_a_local_caveat() {
        let mut finding = finding("terminal_tool_failure");
        finding.certainty.missing_facts = vec!["operation_identity".into()];
        let presentation = FindingPresenter.present(&finding).unwrap();
        assert!(presentation.caveat.unwrap().contains("operation_identity"));
    }
}
