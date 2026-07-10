use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{PAIRED_FINDING_VERIFICATION_SCHEMA_VERSION, PAIRED_FINDING_VERIFIER_VERSION};
use crate::behavior::{BehaviorFinding, FindingSeverity};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairedFindingVerification {
    pub schema_version: String,
    pub verification_id: String,
    pub verifier_version: String,
    pub severe_threshold: FindingSeverity,
    pub case_id: String,
    pub target_failure_signatures: Vec<String>,
    pub baseline_finding_ids: Vec<String>,
    pub candidate_finding_ids: Vec<String>,
    pub baseline_reproduced: bool,
    pub candidate_resolved: bool,
    pub no_severe_novel_findings: bool,
    pub recurring_target_signatures: Vec<String>,
    pub severe_novel_finding_ids: Vec<String>,
    pub finding_gate_passed: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct PairedFindingVerifier {
    severe_threshold: FindingSeverity,
}

impl Default for PairedFindingVerifier {
    fn default() -> Self {
        Self {
            severe_threshold: FindingSeverity::High,
        }
    }
}

impl PairedFindingVerifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_severe_threshold(mut self, threshold: FindingSeverity) -> Self {
        self.severe_threshold = threshold;
        self
    }

    pub fn verify(
        &self,
        case_id: impl Into<String>,
        target_failure_signatures: impl IntoIterator<Item = String>,
        baseline: &[BehaviorFinding],
        candidate: &[BehaviorFinding],
    ) -> PairedFindingVerification {
        let case_id = case_id.into();
        let targets = target_failure_signatures
            .into_iter()
            .collect::<BTreeSet<_>>();
        let baseline_signatures = baseline
            .iter()
            .map(|finding| finding.failure_signature.as_str())
            .collect::<BTreeSet<_>>();
        let candidate_signatures = candidate
            .iter()
            .map(|finding| finding.failure_signature.as_str())
            .collect::<BTreeSet<_>>();
        let baseline_reproduced = !targets.is_empty()
            && targets
                .iter()
                .all(|target| baseline_signatures.contains(target.as_str()));
        let recurring_target_signatures = targets
            .iter()
            .filter(|target| candidate_signatures.contains(target.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let candidate_resolved = recurring_target_signatures.is_empty();
        let severe_novel_finding_ids = candidate
            .iter()
            .filter(|finding| {
                finding.severity >= self.severe_threshold
                    && !baseline_signatures.contains(finding.failure_signature.as_str())
            })
            .map(|finding| finding.finding_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let no_severe_novel_findings = severe_novel_finding_ids.is_empty();
        let finding_gate_passed =
            baseline_reproduced && candidate_resolved && no_severe_novel_findings;
        let mut reasons = Vec::new();
        if targets.is_empty() {
            reasons.push("no target failure signatures were supplied".to_string());
        } else if !baseline_reproduced {
            reasons.push("baseline did not reproduce every target failure signature".to_string());
        }
        if !candidate_resolved {
            reasons.push("candidate still emits a target failure signature".to_string());
        }
        if !no_severe_novel_findings {
            reasons.push("candidate emits a severe novel finding".to_string());
        }

        let target_failure_signatures = targets.into_iter().collect::<Vec<_>>();
        let baseline_finding_ids = finding_ids(baseline);
        let candidate_finding_ids = finding_ids(candidate);
        let verification_id = verification_id(
            &case_id,
            self.severe_threshold,
            &target_failure_signatures,
            &baseline_finding_ids,
            &candidate_finding_ids,
        );
        PairedFindingVerification {
            schema_version: PAIRED_FINDING_VERIFICATION_SCHEMA_VERSION.to_string(),
            verification_id,
            verifier_version: PAIRED_FINDING_VERIFIER_VERSION.to_string(),
            severe_threshold: self.severe_threshold,
            case_id,
            target_failure_signatures,
            baseline_finding_ids,
            candidate_finding_ids,
            baseline_reproduced,
            candidate_resolved,
            no_severe_novel_findings,
            recurring_target_signatures,
            severe_novel_finding_ids,
            finding_gate_passed,
            reasons,
        }
    }
}

fn finding_ids(findings: &[BehaviorFinding]) -> Vec<String> {
    findings
        .iter()
        .map(|finding| finding.finding_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn verification_id(
    case_id: &str,
    severe_threshold: FindingSeverity,
    targets: &[String],
    baseline_ids: &[String],
    candidate_ids: &[String],
) -> String {
    let mut hasher = Sha256::new();
    let threshold = serde_json::to_string(&severe_threshold).expect("finding severity serializes");
    for value in std::iter::once(case_id)
        .chain(std::iter::once(PAIRED_FINDING_VERIFIER_VERSION))
        .chain(std::iter::once(threshold.as_str()))
        .chain(targets.iter().map(String::as_str))
        .chain(baseline_ids.iter().map(String::as_str))
        .chain(candidate_ids.iter().map(String::as_str))
    {
        hasher.update(value.len().to_be_bytes());
        hasher.update(value.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}
