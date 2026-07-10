use std::collections::BTreeSet;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::Result;
use crate::evaluation::EvaluationResult;

use super::BehaviorFinding;

mod finding;
mod model;
mod pairing;

pub use finding::{PairedFindingVerification, PairedFindingVerifier};
pub use model::{
    BudgetRegressionGate, ExecutionBudget, IncidentRegressionGate, PairedEvaluationComparison,
    PairedEvaluationKey, RemediationInputArtifacts, RemediationVerificationPolicy,
    RemediationVerificationReport, RemediationVerificationRequest, SuiteRegressionGate,
    VerificationArtifactDigest, VerificationGate, VerificationGateStatus,
};

use model::invalid_request;
use pairing::{EvaluationResultIndex, incident_gate, suite_gate};

pub const PAIRED_FINDING_VERIFICATION_SCHEMA_VERSION: &str =
    "traceeval.paired_finding_verification.v1";
pub const PAIRED_FINDING_VERIFIER_VERSION: &str = "1";
pub const REMEDIATION_VERIFICATION_SCHEMA_VERSION: &str = "traceeval.remediation_verification.v1";
pub const REMEDIATION_VERIFICATION_REQUEST_SCHEMA_VERSION: &str =
    "traceeval.remediation_verification_request.v1";
pub const REMEDIATION_VERIFIER_VERSION: &str = "1";

#[derive(Debug, Default, Clone, Copy)]
pub struct RemediationVerifier;

impl RemediationVerifier {
    pub fn new() -> Self {
        Self
    }

    pub fn verify_request(
        &self,
        request: RemediationVerificationRequest,
        baseline_findings: &[BehaviorFinding],
        candidate_findings: &[BehaviorFinding],
        baseline_results: &[EvaluationResult],
        candidate_results: &[EvaluationResult],
    ) -> Result<RemediationVerificationReport> {
        request.validate()?;
        let input_artifacts = request
            .input_artifacts
            .clone()
            .expect("validated remediation request has input artifacts");
        validate_record_count(
            "baseline_findings",
            input_artifacts.baseline_findings.record_count,
            baseline_findings.len(),
        )?;
        validate_record_count(
            "candidate_findings",
            input_artifacts.candidate_findings.record_count,
            candidate_findings.len(),
        )?;
        validate_record_count(
            "baseline_results",
            input_artifacts.baseline_results.record_count,
            baseline_results.len(),
        )?;
        validate_record_count(
            "candidate_results",
            input_artifacts.candidate_results.record_count,
            candidate_results.len(),
        )?;
        let finding_gate = PairedFindingVerifier::default()
            .with_severe_threshold(request.severe_threshold)
            .verify(
                request.case_id,
                request.target_failure_signatures,
                baseline_findings,
                candidate_findings,
            );
        Ok(self.verify(
            finding_gate,
            request.incident_case_id,
            request.suite_case_ids,
            baseline_results,
            candidate_results,
            request.policy_gate,
            request.approval_gate,
            request.baseline_budget,
            request.candidate_budget,
            input_artifacts,
            request.policy,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn verify(
        &self,
        finding_gate: PairedFindingVerification,
        incident_case_id: impl Into<String>,
        suite_case_ids: impl IntoIterator<Item = String>,
        baseline_results: &[EvaluationResult],
        candidate_results: &[EvaluationResult],
        policy_gate: VerificationGate,
        approval_gate: VerificationGate,
        baseline_budget: ExecutionBudget,
        candidate_budget: ExecutionBudget,
        input_artifacts: RemediationInputArtifacts,
        policy: RemediationVerificationPolicy,
    ) -> RemediationVerificationReport {
        let incident_case_id = incident_case_id.into();
        let suite_case_ids = suite_case_ids.into_iter().collect::<BTreeSet<_>>();
        let baseline_index = EvaluationResultIndex::new(baseline_results);
        let candidate_index = EvaluationResultIndex::new(candidate_results);
        let incident_regression_gate =
            incident_gate(&incident_case_id, &baseline_index, &candidate_index);
        let suite_regression_gate =
            suite_gate(&suite_case_ids, &baseline_index, &candidate_index, policy);
        let budget_gate = budget_gate(baseline_budget, candidate_budget, policy);
        let passed = finding_gate.finding_gate_passed
            && incident_regression_gate.passed
            && suite_regression_gate.passed
            && policy_gate.passed_gate()
            && approval_gate.passed_gate()
            && budget_gate.passed;
        let mut reasons = Vec::new();
        if !finding_gate.finding_gate_passed {
            reasons.push("finding verification gate failed".to_string());
        }
        if !incident_regression_gate.passed {
            reasons.push("incident regression case gate failed".to_string());
        }
        if !suite_regression_gate.passed {
            reasons.push("accepted suite regression gate failed".to_string());
        }
        if !policy_gate.passed_gate() {
            reasons.push("policy checks are failed or missing".to_string());
        }
        if !approval_gate.passed_gate() {
            reasons.push("approval checks are failed or missing".to_string());
        }
        if !budget_gate.passed {
            reasons.push("execution budget regression gate failed".to_string());
        }
        let verification_id = remediation_verification_id(
            &finding_gate.verification_id,
            &incident_regression_gate,
            &suite_regression_gate,
            &policy_gate,
            &approval_gate,
            &budget_gate,
            &input_artifacts,
            policy,
        );

        RemediationVerificationReport {
            schema_version: REMEDIATION_VERIFICATION_SCHEMA_VERSION.to_string(),
            verification_id,
            verifier_version: REMEDIATION_VERIFIER_VERSION.to_string(),
            finding_gate,
            incident_regression_gate,
            suite_regression_gate,
            policy_gate,
            approval_gate,
            budget_gate,
            input_artifacts,
            policy,
            passed,
            reasons,
        }
    }
}

fn validate_record_count(role: &str, declared: u64, actual: usize) -> Result<()> {
    if declared != actual as u64 {
        return Err(invalid_request(format!(
            "input_artifacts.{role}.record_count is {declared}, but {actual} records were supplied"
        )));
    }
    Ok(())
}

fn budget_gate(
    baseline: ExecutionBudget,
    candidate: ExecutionBudget,
    policy: RemediationVerificationPolicy,
) -> BudgetRegressionGate {
    let tool_call_increase = candidate
        .tool_call_count
        .saturating_sub(baseline.tool_call_count);
    let latency_increase_ms = candidate.latency_ms.saturating_sub(baseline.latency_ms);
    let cost_increase_microunits = candidate
        .cost_microunits
        .saturating_sub(baseline.cost_microunits);
    BudgetRegressionGate {
        passed: tool_call_increase <= policy.max_tool_call_increase
            && latency_increase_ms <= policy.max_latency_increase_ms
            && cost_increase_microunits <= policy.max_cost_increase_microunits,
        baseline,
        candidate,
        tool_call_increase,
        latency_increase_ms,
        cost_increase_microunits,
    }
}

#[allow(clippy::too_many_arguments)]
fn remediation_verification_id(
    finding_verification_id: &str,
    incident: &IncidentRegressionGate,
    suite: &SuiteRegressionGate,
    policy_gate: &VerificationGate,
    approval_gate: &VerificationGate,
    budget: &BudgetRegressionGate,
    input_artifacts: &RemediationInputArtifacts,
    policy: RemediationVerificationPolicy,
) -> String {
    #[derive(Serialize)]
    struct Identity<'a> {
        verifier_version: &'static str,
        finding_verification_id: &'a str,
        incident: &'a IncidentRegressionGate,
        suite: &'a SuiteRegressionGate,
        policy_gate: &'a VerificationGate,
        approval_gate: &'a VerificationGate,
        budget: &'a BudgetRegressionGate,
        input_artifacts: &'a RemediationInputArtifacts,
        policy: RemediationVerificationPolicy,
    }
    let bytes = serde_json::to_vec(&Identity {
        verifier_version: REMEDIATION_VERIFIER_VERSION,
        finding_verification_id,
        incident,
        suite,
        policy_gate,
        approval_gate,
        budget,
        input_artifacts,
        policy,
    })
    .expect("remediation verification identity serializes");
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
#[path = "verification/tests.rs"]
mod tests;
