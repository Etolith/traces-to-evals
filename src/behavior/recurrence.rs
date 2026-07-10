use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Result, TraceEvalError};

use super::{BEHAVIOR_FINDING_SCHEMA_VERSION, BehaviorFinding, FindingSeverity};

pub const FINDING_RECURRENCE_COMPARISON_SCHEMA_VERSION: &str =
    "traceeval.finding_recurrence_comparison.v1";
pub const FINDING_RECURRENCE_REQUEST_SCHEMA_VERSION: &str =
    "traceeval.finding_recurrence_request.v1";
pub const FINDING_RECURRENCE_COMPARATOR_VERSION: &str = "2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PopulationBasis {
    Exact,
    Sampled,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FindingWindow {
    pub window_id: String,
    pub observed_trace_count: u64,
    pub population_basis: PopulationBasis,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FindingRecurrenceRequest {
    pub schema_version: String,
    pub target_failure_signatures: Vec<String>,
    pub baseline_window: FindingWindow,
    pub candidate_window: FindingWindow,
    #[serde(default = "default_severe_threshold")]
    pub severe_threshold: FindingSeverity,
}

impl FindingRecurrenceRequest {
    pub fn validate(&self) -> Result<()> {
        validate_request(
            &self.schema_version,
            &self.target_failure_signatures,
            &self.baseline_window,
            &self.candidate_window,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FindingKindRate {
    pub kind: String,
    pub occurrence_count: u64,
    pub affected_trace_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub affected_trace_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FindingRecurrenceComparison {
    pub schema_version: String,
    pub comparison_id: String,
    pub comparator_version: String,
    pub severe_threshold: FindingSeverity,
    pub target_failure_signatures: Vec<String>,
    pub baseline_window: FindingWindow,
    pub candidate_window: FindingWindow,
    pub baseline_occurrence_count: u64,
    pub candidate_occurrence_count: u64,
    pub baseline_affected_trace_count: u64,
    pub candidate_affected_trace_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_recurrence_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_recurrence_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recurrence_rate_delta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recurrence_rate_ratio: Option<f64>,
    pub severe_novel_finding_ids: Vec<String>,
    pub baseline_finding_rates_by_kind: Vec<FindingKindRate>,
    pub candidate_finding_rates_by_kind: Vec<FindingKindRate>,
    pub evidence_complete: bool,
    pub evidence_gaps: Vec<String>,
    pub interpretation: String,
}

#[derive(Debug, Clone, Copy)]
pub struct FindingRecurrenceComparator {
    severe_threshold: FindingSeverity,
}

impl Default for FindingRecurrenceComparator {
    fn default() -> Self {
        Self {
            severe_threshold: FindingSeverity::High,
        }
    }
}

impl FindingRecurrenceComparator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_severe_threshold(mut self, threshold: FindingSeverity) -> Self {
        self.severe_threshold = threshold;
        self
    }

    pub fn compare_request(
        &self,
        request: FindingRecurrenceRequest,
        baseline_findings: &[BehaviorFinding],
        candidate_findings: &[BehaviorFinding],
    ) -> Result<FindingRecurrenceComparison> {
        request.validate()?;
        self.with_severe_threshold(request.severe_threshold)
            .compare(
                request.target_failure_signatures,
                request.baseline_window,
                baseline_findings,
                request.candidate_window,
                candidate_findings,
            )
    }

    pub fn compare(
        &self,
        target_failure_signatures: impl IntoIterator<Item = String>,
        baseline_window: FindingWindow,
        baseline_findings: &[BehaviorFinding],
        candidate_window: FindingWindow,
        candidate_findings: &[BehaviorFinding],
    ) -> Result<FindingRecurrenceComparison> {
        let targets = target_failure_signatures
            .into_iter()
            .collect::<BTreeSet<_>>();
        let target_failure_signatures = targets.iter().cloned().collect::<Vec<_>>();
        validate_request(
            FINDING_RECURRENCE_REQUEST_SCHEMA_VERSION,
            &target_failure_signatures,
            &baseline_window,
            &candidate_window,
        )?;
        let baseline_findings = unique_findings("baseline", baseline_findings)?;
        let candidate_findings = unique_findings("candidate", candidate_findings)?;
        validate_observed_population("baseline", &baseline_window, &baseline_findings)?;
        validate_observed_population("candidate", &candidate_window, &candidate_findings)?;
        let baseline_target_findings = matching_findings(&baseline_findings, &targets);
        let candidate_target_findings = matching_findings(&candidate_findings, &targets);
        let baseline_occurrence_count = baseline_target_findings.len() as u64;
        let candidate_occurrence_count = candidate_target_findings.len() as u64;
        let baseline_affected_trace_count = affected_trace_count(&baseline_target_findings);
        let candidate_affected_trace_count = affected_trace_count(&candidate_target_findings);
        let baseline_recurrence_rate = rate(
            baseline_affected_trace_count,
            baseline_window.observed_trace_count,
        );
        let candidate_recurrence_rate = rate(
            candidate_affected_trace_count,
            candidate_window.observed_trace_count,
        );
        let recurrence_rate_delta = baseline_recurrence_rate
            .zip(candidate_recurrence_rate)
            .map(|(baseline, candidate)| candidate - baseline);
        let recurrence_rate_ratio = baseline_recurrence_rate
            .zip(candidate_recurrence_rate)
            .and_then(|(baseline, candidate)| {
                if baseline > 0.0 {
                    Some(candidate / baseline)
                } else {
                    None
                }
            });
        let baseline_signatures = baseline_findings
            .iter()
            .map(|finding| finding.failure_signature.as_str())
            .collect::<BTreeSet<_>>();
        let severe_novel_finding_ids = candidate_findings
            .iter()
            .filter(|finding| {
                finding.severity >= self.severe_threshold
                    && !baseline_signatures.contains(finding.failure_signature.as_str())
            })
            .map(|finding| finding.finding_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let comparison_id = comparison_id(
            self.severe_threshold,
            &target_failure_signatures,
            &baseline_window,
            &candidate_window,
            &baseline_findings,
            &candidate_findings,
        );
        let baseline_finding_rates_by_kind =
            finding_rates_by_kind(&baseline_findings, baseline_window.observed_trace_count);
        let candidate_finding_rates_by_kind =
            finding_rates_by_kind(&candidate_findings, candidate_window.observed_trace_count);
        let mut evidence_gaps = Vec::new();
        if baseline_window.observed_trace_count == 0 {
            evidence_gaps.push("baseline_window_has_no_observed_traces".to_string());
        }
        if candidate_window.observed_trace_count == 0 {
            evidence_gaps.push("candidate_window_has_no_observed_traces".to_string());
        }
        if baseline_window.population_basis == PopulationBasis::Unknown {
            evidence_gaps.push("baseline_population_basis_unknown".to_string());
        }
        if candidate_window.population_basis == PopulationBasis::Unknown {
            evidence_gaps.push("candidate_population_basis_unknown".to_string());
        }
        let evidence_complete = evidence_gaps.is_empty();
        let interpretation = if !evidence_complete {
            "Evidence is incomplete; deployment policy must pause rather than interpret absence as success."
                .to_string()
        } else if matches!(
            (
                baseline_window.population_basis,
                candidate_window.population_basis
            ),
            (PopulationBasis::Exact, PopulationBasis::Exact)
        ) {
            "Rates describe the exact observed windows; deployment policy decides promotion or rollback."
                .to_string()
        } else {
            "Rates describe observed sampled traces and must not be presented as exact customer prevalence; deployment policy decides promotion or rollback."
                .to_string()
        };

        Ok(FindingRecurrenceComparison {
            schema_version: FINDING_RECURRENCE_COMPARISON_SCHEMA_VERSION.to_string(),
            comparison_id,
            comparator_version: FINDING_RECURRENCE_COMPARATOR_VERSION.to_string(),
            severe_threshold: self.severe_threshold,
            target_failure_signatures,
            baseline_window,
            candidate_window,
            baseline_occurrence_count,
            candidate_occurrence_count,
            baseline_affected_trace_count,
            candidate_affected_trace_count,
            baseline_recurrence_rate,
            candidate_recurrence_rate,
            recurrence_rate_delta,
            recurrence_rate_ratio,
            severe_novel_finding_ids,
            baseline_finding_rates_by_kind,
            candidate_finding_rates_by_kind,
            evidence_complete,
            evidence_gaps,
            interpretation,
        })
    }
}

fn matching_findings<'a>(
    findings: &[&'a BehaviorFinding],
    targets: &BTreeSet<String>,
) -> Vec<&'a BehaviorFinding> {
    findings
        .iter()
        .copied()
        .filter(|finding| targets.contains(&finding.failure_signature))
        .collect()
}

fn affected_trace_count(findings: &[&BehaviorFinding]) -> u64 {
    findings
        .iter()
        .map(|finding| finding.trace_id.as_str())
        .collect::<BTreeSet<_>>()
        .len() as u64
}

fn unique_findings<'a>(
    window_name: &str,
    findings: &'a [BehaviorFinding],
) -> Result<Vec<&'a BehaviorFinding>> {
    let mut unique = BTreeMap::<&str, &BehaviorFinding>::new();
    for finding in findings {
        if finding.schema_version != BEHAVIOR_FINDING_SCHEMA_VERSION {
            return Err(invalid_request(format!(
                "{window_name} finding {} has unsupported schema_version {}",
                finding.finding_id, finding.schema_version
            )));
        }
        if finding.finding_id.trim().is_empty()
            || finding.trace_id.trim().is_empty()
            || finding.kind.trim().is_empty()
            || finding.failure_signature.trim().is_empty()
        {
            return Err(invalid_request(format!(
                "{window_name} findings require non-empty finding_id, trace_id, kind, and failure_signature"
            )));
        }
        if let Some(previous) = unique.insert(finding.finding_id.as_str(), finding)
            && previous != finding
        {
            return Err(invalid_request(format!(
                "{window_name} finding_id {} has conflicting records",
                finding.finding_id
            )));
        }
    }
    Ok(unique.into_values().collect())
}

fn validate_observed_population(
    window_name: &str,
    window: &FindingWindow,
    findings: &[&BehaviorFinding],
) -> Result<()> {
    let affected_trace_count = findings
        .iter()
        .map(|finding| finding.trace_id.as_str())
        .collect::<BTreeSet<_>>()
        .len() as u64;
    if affected_trace_count > window.observed_trace_count {
        return Err(invalid_request(format!(
            "{window_name} has {affected_trace_count} affected traces but only {} observed traces",
            window.observed_trace_count
        )));
    }
    Ok(())
}

fn finding_rates_by_kind(
    findings: &[&BehaviorFinding],
    observed_trace_count: u64,
) -> Vec<FindingKindRate> {
    let mut by_kind = BTreeMap::<&str, (u64, BTreeSet<&str>)>::new();
    for finding in findings {
        let entry = by_kind
            .entry(finding.kind.as_str())
            .or_insert_with(|| (0, BTreeSet::new()));
        entry.0 += 1;
        entry.1.insert(finding.trace_id.as_str());
    }
    by_kind
        .into_iter()
        .map(|(kind, (occurrence_count, traces))| {
            let affected_trace_count = traces.len() as u64;
            FindingKindRate {
                kind: kind.to_string(),
                occurrence_count,
                affected_trace_count,
                affected_trace_rate: rate(affected_trace_count, observed_trace_count),
            }
        })
        .collect()
}

fn rate(affected_trace_count: u64, observed_trace_count: u64) -> Option<f64> {
    (observed_trace_count > 0).then_some(affected_trace_count as f64 / observed_trace_count as f64)
}

fn default_severe_threshold() -> FindingSeverity {
    FindingSeverity::High
}

fn validate_request(
    schema_version: &str,
    targets: &[String],
    baseline_window: &FindingWindow,
    candidate_window: &FindingWindow,
) -> Result<()> {
    if schema_version != FINDING_RECURRENCE_REQUEST_SCHEMA_VERSION {
        return Err(invalid_request(format!(
            "unsupported schema_version {schema_version}; expected {FINDING_RECURRENCE_REQUEST_SCHEMA_VERSION}"
        )));
    }
    if targets.is_empty() || targets.iter().any(|target| target.trim().is_empty()) {
        return Err(invalid_request(
            "target_failure_signatures must contain non-empty values",
        ));
    }
    if baseline_window.window_id.trim().is_empty() || candidate_window.window_id.trim().is_empty() {
        return Err(invalid_request("window_id values must not be empty"));
    }
    if baseline_window.window_id == candidate_window.window_id {
        return Err(invalid_request(
            "baseline and candidate window_id values must differ",
        ));
    }
    Ok(())
}

fn invalid_request(message: impl Into<String>) -> TraceEvalError {
    TraceEvalError::InvalidFindingRecurrenceRequest {
        message: message.into(),
    }
}

fn comparison_id(
    severe_threshold: FindingSeverity,
    targets: &[String],
    baseline_window: &FindingWindow,
    candidate_window: &FindingWindow,
    baseline_findings: &[&BehaviorFinding],
    candidate_findings: &[&BehaviorFinding],
) -> String {
    #[derive(Serialize)]
    struct Identity<'a> {
        comparator_version: &'static str,
        severe_threshold: FindingSeverity,
        targets: &'a [String],
        baseline_window: &'a FindingWindow,
        candidate_window: &'a FindingWindow,
        baseline_finding_ids: Vec<&'a str>,
        candidate_finding_ids: Vec<&'a str>,
    }
    let baseline_finding_ids = baseline_findings
        .iter()
        .map(|finding| finding.finding_id.as_str())
        .collect::<Vec<_>>();
    let candidate_finding_ids = candidate_findings
        .iter()
        .map(|finding| finding.finding_id.as_str())
        .collect::<Vec<_>>();
    let bytes = serde_json::to_vec(&Identity {
        comparator_version: FINDING_RECURRENCE_COMPARATOR_VERSION,
        severe_threshold,
        targets,
        baseline_window,
        candidate_window,
        baseline_finding_ids,
        candidate_finding_ids,
    })
    .expect("finding recurrence comparison identity serializes");
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
#[path = "recurrence/tests.rs"]
mod tests;
