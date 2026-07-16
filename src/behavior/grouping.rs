use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{BehaviorFinding, FindingSeverity, RecoveryStatus};

pub const KNOWN_SIGNATURE_GROUP_SCHEMA_VERSION: &str = "traceeval.known_signature_group.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnownSignatureGroup {
    pub schema_version: String,
    pub group_id: String,
    pub failure_signature: String,
    pub detector_ids: Vec<String>,
    pub finding_ids: Vec<String>,
    pub trace_ids: Vec<String>,
    pub occurrence_count: u64,
    pub severity: FindingSeverity,
    pub first_seen_at: String,
    pub last_seen_at: String,
    pub recovery_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct KnownSignatureGrouper;

impl KnownSignatureGrouper {
    pub fn group(&self, findings: &[BehaviorFinding]) -> Vec<KnownSignatureGroup> {
        let unique_findings = findings
            .iter()
            .map(|finding| (finding.finding_id.as_str(), finding))
            .collect::<BTreeMap<_, _>>();
        let mut groups: BTreeMap<&str, Vec<&BehaviorFinding>> = BTreeMap::new();
        for finding in unique_findings.into_values() {
            groups
                .entry(finding.failure_signature.as_str())
                .or_default()
                .push(finding);
        }

        groups
            .into_iter()
            .map(|(failure_signature, findings)| {
                let detector_ids = findings
                    .iter()
                    .map(|finding| finding.detector_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let finding_ids = findings
                    .iter()
                    .map(|finding| finding.finding_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let trace_ids = findings
                    .iter()
                    .map(|finding| finding.trace_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let severity = findings
                    .iter()
                    .map(|finding| finding.severity)
                    .max()
                    .unwrap_or(FindingSeverity::Info);
                let first_seen_at = findings
                    .iter()
                    .map(|finding| finding.created_at.as_str())
                    .min()
                    .unwrap_or("unknown")
                    .to_string();
                let last_seen_at = findings
                    .iter()
                    .map(|finding| finding.created_at.as_str())
                    .max()
                    .unwrap_or("unknown")
                    .to_string();
                let mut recovery_counts = BTreeMap::new();
                for finding in &findings {
                    *recovery_counts
                        .entry(recovery_name(finding.recovery).to_string())
                        .or_default() += 1;
                }

                KnownSignatureGroup {
                    schema_version: KNOWN_SIGNATURE_GROUP_SCHEMA_VERSION.to_string(),
                    group_id: group_id(failure_signature),
                    failure_signature: failure_signature.to_string(),
                    detector_ids,
                    occurrence_count: finding_ids.len() as u64,
                    finding_ids,
                    trace_ids,
                    severity,
                    first_seen_at,
                    last_seen_at,
                    recovery_counts,
                }
            })
            .collect()
    }
}

fn group_id(failure_signature: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"known_signature");
    hasher.update(failure_signature.len().to_be_bytes());
    hasher.update(failure_signature.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn recovery_name(recovery: RecoveryStatus) -> &'static str {
    match recovery {
        RecoveryStatus::Recovered => "recovered",
        RecoveryStatus::Unrecovered => "unrecovered",
        RecoveryStatus::Unknown => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::behavior::{BEHAVIOR_FINDING_SCHEMA_VERSION, EvidenceRef};

    fn finding(id: &str, trace_id: &str, created_at: &str) -> BehaviorFinding {
        BehaviorFinding {
            schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
            finding_id: id.to_string(),
            detector_id: "terminal_tool_failure".to_string(),
            detector_version: "2".to_string(),
            trace_id: trace_id.to_string(),
            kind: "terminal_tool_failure".to_string(),
            severity: FindingSeverity::High,
            recovery: RecoveryStatus::Unrecovered,
            confidence: Some(1.0),
            certainty: crate::behavior::FindingCertaintyV1::default(),
            failure_signature: "sha256:same".to_string(),
            evidence: vec![EvidenceRef::span(format!("{id}-span"))],
            created_at: created_at.to_string(),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn groups_exact_signatures_and_deduplicates_finding_delivery() {
        let first = finding("finding-1", "trace-1", "2026-07-10T12:00:00Z");
        let second = finding("finding-2", "trace-2", "2026-07-10T12:01:00Z");

        let groups = KnownSignatureGrouper.group(&[first.clone(), first, second]);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].occurrence_count, 2);
        assert_eq!(groups[0].trace_ids, ["trace-1", "trace-2"]);
        assert_eq!(groups[0].first_seen_at, "2026-07-10T12:00:00Z");
        assert_eq!(groups[0].last_seen_at, "2026-07-10T12:01:00Z");
        assert!(groups[0].group_id.starts_with("sha256:"));
    }
}
