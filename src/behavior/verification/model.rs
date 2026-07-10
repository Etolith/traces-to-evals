use serde::{Deserialize, Serialize};

use super::{PairedFindingVerification, REMEDIATION_VERIFICATION_REQUEST_SCHEMA_VERSION};
use crate::behavior::{EvidenceRef, FindingSeverity};
use crate::{Result, TraceEvalError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationGateStatus {
    Passed,
    Failed,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationGate {
    pub status: VerificationGateStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl VerificationGate {
    pub fn passed(evidence: Vec<EvidenceRef>) -> Self {
        Self {
            status: VerificationGateStatus::Passed,
            evidence,
            detail: None,
        }
    }

    pub fn failed(detail: impl Into<String>, evidence: Vec<EvidenceRef>) -> Self {
        Self {
            status: VerificationGateStatus::Failed,
            evidence,
            detail: Some(detail.into()),
        }
    }

    pub fn missing(detail: impl Into<String>) -> Self {
        Self {
            status: VerificationGateStatus::Missing,
            evidence: Vec::new(),
            detail: Some(detail.into()),
        }
    }

    pub fn passed_gate(&self) -> bool {
        self.status == VerificationGateStatus::Passed
            && !self.evidence.is_empty()
            && self.evidence.iter().all(|evidence| {
                !evidence.kind.trim().is_empty() && !evidence.identity.trim().is_empty()
            })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionBudget {
    pub tool_call_count: u64,
    pub latency_ms: u64,
    pub cost_microunits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationArtifactDigest {
    pub content_hash: String,
    pub byte_count: u64,
    pub record_count: u64,
}

impl VerificationArtifactDigest {
    pub fn validate(&self, role: &str) -> Result<()> {
        let Some(hex) = self.content_hash.strip_prefix("sha256:") else {
            return Err(invalid_request(format!(
                "input_artifacts.{role}.content_hash must use sha256"
            )));
        };
        if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(invalid_request(format!(
                "input_artifacts.{role}.content_hash must contain 64 hexadecimal characters"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemediationInputArtifacts {
    pub baseline_findings: VerificationArtifactDigest,
    pub candidate_findings: VerificationArtifactDigest,
    pub baseline_results: VerificationArtifactDigest,
    pub candidate_results: VerificationArtifactDigest,
}

impl RemediationInputArtifacts {
    pub fn validate(&self) -> Result<()> {
        self.baseline_findings.validate("baseline_findings")?;
        self.candidate_findings.validate("candidate_findings")?;
        self.baseline_results.validate("baseline_results")?;
        self.candidate_results.validate("candidate_results")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemediationVerificationPolicy {
    pub max_new_suite_failures: usize,
    pub max_suite_score_drop: f32,
    pub max_tool_call_increase: u64,
    pub max_latency_increase_ms: u64,
    pub max_cost_increase_microunits: u64,
}

impl RemediationVerificationPolicy {
    pub fn strict() -> Self {
        Self {
            max_new_suite_failures: 0,
            max_suite_score_drop: 0.0,
            max_tool_call_increase: 0,
            max_latency_increase_ms: 0,
            max_cost_increase_microunits: 0,
        }
    }
}

impl Default for RemediationVerificationPolicy {
    fn default() -> Self {
        Self::strict()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemediationVerificationRequest {
    pub schema_version: String,
    pub case_id: String,
    pub target_failure_signatures: Vec<String>,
    pub incident_case_id: String,
    pub suite_case_ids: Vec<String>,
    #[serde(default = "default_severe_threshold")]
    pub severe_threshold: FindingSeverity,
    pub policy_gate: VerificationGate,
    pub approval_gate: VerificationGate,
    pub baseline_budget: ExecutionBudget,
    pub candidate_budget: ExecutionBudget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_artifacts: Option<RemediationInputArtifacts>,
    #[serde(default)]
    pub policy: RemediationVerificationPolicy,
}

impl RemediationVerificationRequest {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != REMEDIATION_VERIFICATION_REQUEST_SCHEMA_VERSION {
            return Err(invalid_request(format!(
                "unsupported schema_version {}; expected {}",
                self.schema_version, REMEDIATION_VERIFICATION_REQUEST_SCHEMA_VERSION
            )));
        }
        if self.case_id.trim().is_empty() {
            return Err(invalid_request("case_id must not be empty"));
        }
        if self.incident_case_id.trim().is_empty() {
            return Err(invalid_request("incident_case_id must not be empty"));
        }
        if self.target_failure_signatures.is_empty()
            || self
                .target_failure_signatures
                .iter()
                .any(|value| value.trim().is_empty())
        {
            return Err(invalid_request(
                "target_failure_signatures must contain non-empty values",
            ));
        }
        if self.suite_case_ids.is_empty()
            || self
                .suite_case_ids
                .iter()
                .any(|value| value.trim().is_empty())
        {
            return Err(invalid_request(
                "suite_case_ids must contain non-empty values",
            ));
        }
        if !self.policy.max_suite_score_drop.is_finite()
            || !(0.0..=1.0).contains(&self.policy.max_suite_score_drop)
        {
            return Err(invalid_request(
                "policy.max_suite_score_drop must be finite and between 0 and 1",
            ));
        }
        self.input_artifacts
            .as_ref()
            .ok_or_else(|| invalid_request("input_artifacts are required"))?
            .validate()?;
        Ok(())
    }
}

fn default_severe_threshold() -> FindingSeverity {
    FindingSeverity::High
}

pub(super) fn invalid_request(message: impl Into<String>) -> TraceEvalError {
    TraceEvalError::InvalidRemediationVerificationRequest {
        message: message.into(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PairedEvaluationKey {
    pub case_id: String,
    pub evaluator_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedEvaluationComparison {
    pub key: PairedEvaluationKey,
    pub baseline_passed: bool,
    pub candidate_passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_drop: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IncidentRegressionGate {
    pub case_id: String,
    pub passed: bool,
    pub paired_result_count: usize,
    pub paired_results: Vec<PairedEvaluationComparison>,
    pub missing_candidate_results: Vec<PairedEvaluationKey>,
    pub unexpected_candidate_results: Vec<PairedEvaluationKey>,
    pub failed_candidate_results: Vec<PairedEvaluationKey>,
    pub unversioned_results: Vec<PairedEvaluationKey>,
    pub invalid_identity_results: Vec<PairedEvaluationKey>,
    pub invalid_score_results: Vec<PairedEvaluationKey>,
    pub duplicate_baseline_results: Vec<PairedEvaluationKey>,
    pub duplicate_candidate_results: Vec<PairedEvaluationKey>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuiteRegressionGate {
    pub passed: bool,
    pub suite_case_count: usize,
    pub suite_case_ids: Vec<String>,
    pub paired_result_count: usize,
    pub paired_results: Vec<PairedEvaluationComparison>,
    pub missing_baseline_cases: Vec<String>,
    pub missing_candidate_results: Vec<PairedEvaluationKey>,
    pub unexpected_candidate_results: Vec<PairedEvaluationKey>,
    pub new_failure_results: Vec<PairedEvaluationKey>,
    pub score_drop_results: Vec<PairedEvaluationKey>,
    pub unversioned_results: Vec<PairedEvaluationKey>,
    pub invalid_identity_results: Vec<PairedEvaluationKey>,
    pub invalid_score_results: Vec<PairedEvaluationKey>,
    pub duplicate_baseline_results: Vec<PairedEvaluationKey>,
    pub duplicate_candidate_results: Vec<PairedEvaluationKey>,
    pub maximum_score_drop: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetRegressionGate {
    pub passed: bool,
    pub baseline: ExecutionBudget,
    pub candidate: ExecutionBudget,
    pub tool_call_increase: u64,
    pub latency_increase_ms: u64,
    pub cost_increase_microunits: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemediationVerificationReport {
    pub schema_version: String,
    pub verification_id: String,
    pub verifier_version: String,
    pub finding_gate: PairedFindingVerification,
    pub incident_regression_gate: IncidentRegressionGate,
    pub suite_regression_gate: SuiteRegressionGate,
    pub policy_gate: VerificationGate,
    pub approval_gate: VerificationGate,
    pub budget_gate: BudgetRegressionGate,
    pub input_artifacts: RemediationInputArtifacts,
    pub policy: RemediationVerificationPolicy,
    pub passed: bool,
    pub reasons: Vec<String>,
}
