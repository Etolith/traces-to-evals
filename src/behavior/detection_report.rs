use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::input::BehaviorInputCoverageV1;
use super::model::{BehaviorFinding, EvidenceRef};

pub const DETECTION_REPORT_SCHEMA_VERSION: &str = "traceeval.detection_report.v1";
pub const CONSERVATIVE_DETECTOR_PROFILE_ID: &str = "traceeval.conservative";
pub const CONSERVATIVE_DETECTOR_PROFILE_VERSION: &str = "2";
pub const DETECTOR_PROFILE_SCHEMA_VERSION: &str = "traceeval.detector_profile.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectorProfileIdentityV1 {
    pub profile_id: String,
    pub profile_version: String,
}

impl DetectorProfileIdentityV1 {
    pub fn conservative() -> Self {
        Self {
            profile_id: CONSERVATIVE_DETECTOR_PROFILE_ID.into(),
            profile_version: CONSERVATIVE_DETECTOR_PROFILE_VERSION.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepeatedFailurePolicyV1 {
    pub minimum_failures: usize,
    pub maximum_call_window: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolLoopPolicyV1 {
    pub minimum_equivalent_calls: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolUsageBudgetV1 {
    pub maximum_calls: usize,
    pub maximum_total_duration_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectorProfileV1 {
    pub schema_version: String,
    pub identity: DetectorProfileIdentityV1,
    pub enabled_detectors: BTreeSet<String>,
    pub repeated_failure: RepeatedFailurePolicyV1,
    pub tool_loop: ToolLoopPolicyV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_usage_budget: Option<ToolUsageBudgetV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_signal_protocol: Option<String>,
}

impl DetectorProfileV1 {
    pub fn conservative() -> Self {
        Self {
            schema_version: DETECTOR_PROFILE_SCHEMA_VERSION.into(),
            identity: DetectorProfileIdentityV1::conservative(),
            enabled_detectors: [
                "terminal_tool_failure",
                "uncertain_mutation_state",
                "false_success_claim",
                "approval_bypass",
                "policy_violation",
                "unresolved_escalation",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            repeated_failure: RepeatedFailurePolicyV1 {
                minimum_failures: 3,
                maximum_call_window: 8,
            },
            tool_loop: ToolLoopPolicyV1 {
                minimum_equivalent_calls: 4,
            },
            tool_usage_budget: None,
            terminal_signal_protocol: None,
        }
    }

    pub fn coding_agent() -> Self {
        let mut profile = Self::conservative();
        profile.identity = DetectorProfileIdentityV1 {
            profile_id: "traceeval.coding_agent".into(),
            profile_version: "2".into(),
        };
        profile
            .enabled_detectors
            .insert("missing_resolution".into());
        profile.terminal_signal_protocol = Some("traceeval.coding_agent.terminal.v1".into());
        profile
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != DETECTOR_PROFILE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported detector profile schema {}",
                self.schema_version
            ));
        }
        if self.identity.profile_id.trim().is_empty()
            || self.identity.profile_version.trim().is_empty()
        {
            return Err("profile identity must be non-empty".into());
        }
        if self.repeated_failure.minimum_failures < 2
            || self.repeated_failure.maximum_call_window < self.repeated_failure.minimum_failures
        {
            return Err("repeated-failure threshold/window is invalid".into());
        }
        if self.tool_loop.minimum_equivalent_calls < 2 {
            return Err("tool-loop threshold must be at least two".into());
        }
        if self.enabled_detectors.contains("missing_resolution")
            && self.terminal_signal_protocol.is_none()
        {
            return Err(
                "missing_resolution requires an explicit versioned terminal-signal protocol".into(),
            );
        }
        if self.enabled_detectors.contains("excessive_tool_usage")
            && self.tool_usage_budget.is_none()
        {
            return Err("excessive_tool_usage requires an explicit tool-usage budget".into());
        }
        if self.tool_usage_budget.as_ref().is_some_and(|budget| {
            budget.maximum_calls == 0 || budget.maximum_total_duration_ms == 0
        }) {
            return Err("tool-usage budgets must be non-zero".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectorEvaluationStatusV1 {
    Evaluated,
    Inconclusive,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectorCoverageV1 {
    pub detector_id: String,
    pub status: DetectorEvaluationStatusV1,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub required_facts: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub observed_facts: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub missing_facts: BTreeSet<String>,
    pub semantic_coverage: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryDiagnosticSeverityV1 {
    Info,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryDiagnosticV1 {
    pub code: String,
    pub severity: TelemetryDiagnosticSeverityV1,
    pub message: String,
    pub trace_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detector_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionReportV1 {
    pub schema_version: String,
    pub trace_id: String,
    pub input_schema_version: String,
    pub profile: DetectorProfileIdentityV1,
    pub detector_versions: BTreeMap<String, String>,
    pub input_coverage: BehaviorInputCoverageV1,
    pub detector_coverage: BTreeMap<String, DetectorCoverageV1>,
    #[serde(default)]
    pub findings: Vec<BehaviorFinding>,
    #[serde(default)]
    pub telemetry_diagnostics: Vec<TelemetryDiagnosticV1>,
}
