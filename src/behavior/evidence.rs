use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    AgentBehaviorTrace, BehaviorFinding, EscalationStatus, EvidenceRef, FinalOutcomeStatus,
    OperationEffect, RetrySafety, ToolCallStatus, ToolRequirement,
};

pub const EVIDENCE_PACKET_SCHEMA_VERSION: &str = "traceeval.evidence_packet.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidencePacket {
    pub schema_version: String,
    pub packet_id: String,
    pub content_hash: String,
    pub trace_ids: Vec<String>,
    pub finding_ids: Vec<String>,
    pub evidence: Vec<EvidenceRef>,
    pub detector_versions: BTreeMap<String, Vec<String>>,
    pub adapter_versions: BTreeMap<String, Vec<String>>,
    pub telemetry_gaps: Vec<String>,
    pub created_at: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EvidencePacketBuilder;

impl EvidencePacketBuilder {
    pub fn build(
        &self,
        traces: &[AgentBehaviorTrace],
        findings: &[BehaviorFinding],
    ) -> EvidencePacket {
        let trace_ids = traces
            .iter()
            .map(|trace| trace.trace_id.clone())
            .chain(findings.iter().map(|finding| finding.trace_id.clone()))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let finding_ids = findings
            .iter()
            .map(|finding| finding.finding_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut evidence = traces
            .iter()
            .flat_map(trace_evidence)
            .chain(findings.iter().flat_map(|finding| finding.evidence.clone()))
            .collect::<Vec<_>>();
        evidence.sort_by(|left, right| left.identity.cmp(&right.identity));
        evidence.dedup_by(|left, right| left.identity == right.identity);
        let mut detector_version_sets: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for finding in findings {
            detector_version_sets
                .entry(finding.detector_id.clone())
                .or_default()
                .insert(finding.detector_version.clone());
        }
        let detector_versions = detector_version_sets
            .into_iter()
            .map(|(detector_id, versions)| (detector_id, versions.into_iter().collect()))
            .collect::<BTreeMap<_, _>>();
        let mut adapter_version_sets: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for trace in traces {
            let Some((adapter_id, adapter_version)) = trace
                .metadata
                .get("traceeval.behavior_adapter.id")
                .and_then(|value| value.as_str())
                .zip(
                    trace
                        .metadata
                        .get("traceeval.behavior_adapter.version")
                        .and_then(|value| value.as_str()),
                )
            else {
                continue;
            };
            adapter_version_sets
                .entry(adapter_id.to_string())
                .or_default()
                .insert(adapter_version.to_string());
        }
        let adapter_versions = adapter_version_sets
            .into_iter()
            .map(|(adapter_id, versions)| (adapter_id, versions.into_iter().collect()))
            .collect::<BTreeMap<_, _>>();
        let telemetry_gaps = traces
            .iter()
            .flat_map(telemetry_gaps)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let created_at = findings
            .iter()
            .map(|finding| finding.created_at.as_str())
            .max()
            .unwrap_or("unknown")
            .to_string();

        let content = EvidencePacketContent {
            trace_ids: &trace_ids,
            finding_ids: &finding_ids,
            evidence: &evidence,
            detector_versions: &detector_versions,
            adapter_versions: &adapter_versions,
            telemetry_gaps: &telemetry_gaps,
            created_at: &created_at,
        };
        let serialized = serde_json::to_vec(&content).expect("evidence packet content serializes");
        let content_hash = format!("sha256:{:x}", Sha256::digest(&serialized));
        let packet_id = hash_parts(["evidence_packet", content_hash.as_str()]);

        EvidencePacket {
            schema_version: EVIDENCE_PACKET_SCHEMA_VERSION.to_string(),
            packet_id,
            content_hash,
            trace_ids,
            finding_ids,
            evidence,
            detector_versions,
            adapter_versions,
            telemetry_gaps,
            created_at,
        }
    }
}

#[derive(Serialize)]
struct EvidencePacketContent<'a> {
    trace_ids: &'a [String],
    finding_ids: &'a [String],
    evidence: &'a [EvidenceRef],
    detector_versions: &'a BTreeMap<String, Vec<String>>,
    adapter_versions: &'a BTreeMap<String, Vec<String>>,
    telemetry_gaps: &'a [String],
    created_at: &'a str,
}

fn trace_evidence(trace: &AgentBehaviorTrace) -> Vec<EvidenceRef> {
    trace
        .evidence
        .iter()
        .cloned()
        .chain(
            trace
                .tool_calls
                .iter()
                .flat_map(|call| call.evidence.clone()),
        )
        .chain(
            trace
                .policy_decisions
                .iter()
                .flat_map(|decision| decision.evidence.clone()),
        )
        .chain(trace.final_outcome.evidence.clone())
        .collect()
}

fn telemetry_gaps(trace: &AgentBehaviorTrace) -> Vec<String> {
    let mut gaps = Vec::new();
    for call in &trace.tool_calls {
        let prefix = format!("trace:{}:call:{}", trace.trace_id, call.call_id);
        if call.status == ToolCallStatus::Unknown {
            gaps.push(format!("{prefix}:status_unknown"));
        }
        if call.effect == OperationEffect::Unknown {
            gaps.push(format!("{prefix}:effect_unknown"));
        }
        if call.retry_safety == RetrySafety::Unknown {
            gaps.push(format!("{prefix}:retry_safety_unknown"));
        }
        if call.requirement == ToolRequirement::Unknown {
            gaps.push(format!("{prefix}:requirement_unknown"));
        }
    }
    if trace.final_outcome.status == FinalOutcomeStatus::Unknown {
        gaps.push(format!("trace:{}:final_outcome_unknown", trace.trace_id));
    }
    if trace.final_outcome.escalation == EscalationStatus::Unknown {
        gaps.push(format!("trace:{}:escalation_unknown", trace.trace_id));
    }
    gaps
}

fn hash_parts<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.len().to_be_bytes());
        hasher.update(part.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::{
        AgentBehaviorTrace, BEHAVIOR_FINDING_SCHEMA_VERSION, BehaviorFinding, FindingSeverity,
        RecoveryStatus,
    };

    fn finding() -> BehaviorFinding {
        BehaviorFinding {
            schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
            finding_id: "finding-1".to_string(),
            detector_id: "missing_resolution".to_string(),
            detector_version: "2".to_string(),
            trace_id: "trace-1".to_string(),
            kind: "missing_resolution".to_string(),
            severity: FindingSeverity::Medium,
            recovery: RecoveryStatus::Unrecovered,
            confidence: Some(1.0),
            failure_signature: "sha256:signature".to_string(),
            evidence: vec![EvidenceRef::span("root")],
            created_at: "2026-07-10T12:00:00Z".to_string(),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn packet_is_deterministic_and_does_not_copy_trace_content() {
        let mut trace = AgentBehaviorTrace::new("trace-1");
        trace.input_summary = Some("private customer prompt".to_string());
        trace.evidence.push(EvidenceRef::span("root"));
        let finding = finding();

        let first = EvidencePacketBuilder
            .build(std::slice::from_ref(&trace), std::slice::from_ref(&finding));
        let second = EvidencePacketBuilder.build(&[trace], &[finding]);

        assert_eq!(first, second);
        assert!(first.packet_id.starts_with("sha256:"));
        assert!(first.content_hash.starts_with("sha256:"));
        assert!(
            !serde_json::to_string(&first)
                .unwrap()
                .contains("private customer prompt")
        );
        assert!(
            first
                .telemetry_gaps
                .contains(&"trace:trace-1:final_outcome_unknown".to_string())
        );
    }

    #[test]
    fn packet_preserves_every_detector_version() {
        let trace = AgentBehaviorTrace::new("trace-1");
        let current = finding();
        let mut older = current.clone();
        older.finding_id = "finding-older".to_string();
        older.detector_version = "1".to_string();

        let packet = EvidencePacketBuilder.build(&[trace], &[current, older]);

        assert_eq!(
            packet.detector_versions["missing_resolution"],
            ["1".to_string(), "2".to_string()]
        );
    }
}
