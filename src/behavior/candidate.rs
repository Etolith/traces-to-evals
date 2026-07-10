use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use super::EvidencePacket;
use super::model::{
    AgentBehaviorTrace, BehaviorFinding, CandidateGenerator, CandidateReview,
    CandidateReviewDecision, EVAL_CANDIDATE_SCHEMA_VERSION, EvalCandidate, EvalCandidateStatus,
    RedactedCandidateInput,
};
use crate::{Result, TraceEvalError};

mod expectations;

use expectations::expected_behavior;

pub trait EvalCandidateGenerator {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn prompt_version(&self) -> Option<&str> {
        None
    }
    fn model(&self) -> Option<&str> {
        None
    }
    fn generate(&self, trace: &AgentBehaviorTrace, finding: &BehaviorFinding) -> EvalCandidate;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FindingEvalCandidateGenerator;

impl FindingEvalCandidateGenerator {
    pub fn generate_all(
        &self,
        traces: &[AgentBehaviorTrace],
        findings: &[BehaviorFinding],
    ) -> Vec<EvalCandidate> {
        let traces = traces
            .iter()
            .map(|trace| (trace.trace_id.as_str(), trace))
            .collect::<BTreeMap<_, _>>();
        findings
            .iter()
            .filter_map(|finding| {
                traces
                    .get(finding.trace_id.as_str())
                    .map(|trace| self.generate(trace, finding))
            })
            .collect()
    }

    pub fn generate_with_evidence_packet(
        &self,
        trace: &AgentBehaviorTrace,
        finding: &BehaviorFinding,
        evidence_packet: &EvidencePacket,
    ) -> EvalCandidate {
        self.generate_internal(trace, finding, Some(evidence_packet), None)
    }

    pub fn generate_with_redacted_input(
        &self,
        trace: &AgentBehaviorTrace,
        finding: &BehaviorFinding,
        proposed_input: RedactedCandidateInput,
    ) -> Result<EvalCandidate> {
        proposed_input.validate()?;
        Ok(self.generate_internal(trace, finding, None, Some(proposed_input)))
    }

    pub fn generate_with_evidence_packet_and_redacted_input(
        &self,
        trace: &AgentBehaviorTrace,
        finding: &BehaviorFinding,
        evidence_packet: &EvidencePacket,
        proposed_input: RedactedCandidateInput,
    ) -> Result<EvalCandidate> {
        proposed_input.validate()?;
        Ok(self.generate_internal(trace, finding, Some(evidence_packet), Some(proposed_input)))
    }

    pub fn generate_all_with_evidence_packet(
        &self,
        traces: &[AgentBehaviorTrace],
        findings: &[BehaviorFinding],
        evidence_packet: &EvidencePacket,
    ) -> Vec<EvalCandidate> {
        let traces = traces
            .iter()
            .map(|trace| (trace.trace_id.as_str(), trace))
            .collect::<BTreeMap<_, _>>();
        findings
            .iter()
            .filter_map(|finding| {
                traces.get(finding.trace_id.as_str()).map(|trace| {
                    self.generate_with_evidence_packet(trace, finding, evidence_packet)
                })
            })
            .collect()
    }

    fn generate_internal(
        &self,
        trace: &AgentBehaviorTrace,
        finding: &BehaviorFinding,
        evidence_packet: Option<&EvidencePacket>,
        proposed_input: Option<RedactedCandidateInput>,
    ) -> EvalCandidate {
        let expected = expected_behavior(&finding.detector_id);
        let proposed_rubric = expected.join(" ");
        let generator = CandidateGenerator {
            name: self.name().to_string(),
            version: self.version().to_string(),
            prompt_version: self.prompt_version().map(str::to_string),
            model: self.model().map(str::to_string),
        };
        let evidence_packet_id = evidence_packet.map(|packet| packet.packet_id.clone());
        let definition_hash = candidate_definition_hash(&CandidateDefinition {
            source_trace_id: &trace.trace_id,
            source_finding_ids: std::slice::from_ref(&finding.finding_id),
            source_cluster_refs: &[],
            evidence_packet_id: evidence_packet_id.as_deref(),
            proposed_input: proposed_input.as_ref(),
            proposed_expected_behavior: &expected,
            proposed_rubric: &proposed_rubric,
            proposed_grader: "agent_behavior_policy",
            generator: &generator,
        });
        let candidate_id = hash_parts([
            finding.finding_id.as_str(),
            self.name(),
            self.version(),
            definition_hash.as_str(),
        ]);
        let mut metadata = BTreeMap::new();
        metadata.insert("detector_id".to_string(), json!(finding.detector_id));
        metadata.insert(
            "detector_version".to_string(),
            json!(finding.detector_version),
        );
        metadata.insert("severity".to_string(), json!(finding.severity));
        metadata.insert("recovery".to_string(), json!(finding.recovery));
        metadata.insert(
            "failure_signature".to_string(),
            json!(finding.failure_signature),
        );
        for key in [
            "traceeval.behavior_adapter.id",
            "traceeval.behavior_adapter.version",
        ] {
            if let Some(value) = finding.metadata.get(key) {
                metadata.insert(key.to_string(), value.clone());
            }
        }

        let mut evidence = finding.evidence.clone();
        if let Some(input) = &proposed_input {
            evidence.extend(input.evidence().iter().cloned());
            evidence.sort_by(|left, right| left.identity.cmp(&right.identity));
            evidence.dedup_by(|left, right| left.identity == right.identity);
        }

        EvalCandidate {
            schema_version: EVAL_CANDIDATE_SCHEMA_VERSION.to_string(),
            candidate_id,
            definition_hash,
            status: EvalCandidateStatus::Candidate,
            source_trace_id: trace.trace_id.clone(),
            source_finding_ids: vec![finding.finding_id.clone()],
            source_cluster_refs: Vec::new(),
            evidence_packet_id,
            proposed_input,
            proposed_expected_behavior: expected,
            proposed_rubric,
            proposed_grader: "agent_behavior_policy".to_string(),
            generator,
            review: None,
            evidence,
            metadata,
        }
    }
}

impl EvalCandidateGenerator for FindingEvalCandidateGenerator {
    fn name(&self) -> &str {
        "finding_to_eval_candidate"
    }

    fn version(&self) -> &str {
        "2"
    }

    fn prompt_version(&self) -> Option<&str> {
        Some("deterministic_behavior_policy.v1")
    }

    fn generate(&self, trace: &AgentBehaviorTrace, finding: &BehaviorFinding) -> EvalCandidate {
        self.generate_internal(trace, finding, None, None)
    }
}

impl EvalCandidate {
    pub fn record_review(mut self, review: CandidateReview) -> Result<Self> {
        self.require_status(EvalCandidateStatus::Candidate, "record review")?;
        self.validate_definition_hash()?;
        self.status = EvalCandidateStatus::Reviewed;
        self.review = Some(review);
        Ok(self)
    }

    pub fn resolve_review(mut self) -> Result<Self> {
        self.require_status(EvalCandidateStatus::Reviewed, "resolve review")?;
        self.validate_definition_hash()?;
        let review =
            self.review
                .as_ref()
                .ok_or_else(|| TraceEvalError::InvalidCandidateTransition {
                    candidate_id: self.candidate_id.clone(),
                    message: "reviewed candidate is missing review provenance".to_string(),
                })?;
        self.status = match review.decision {
            CandidateReviewDecision::Approve => EvalCandidateStatus::Accepted,
            CandidateReviewDecision::Reject => EvalCandidateStatus::Rejected,
        };
        Ok(self)
    }

    pub fn supersede(mut self) -> Result<Self> {
        if matches!(
            self.status,
            EvalCandidateStatus::Rejected | EvalCandidateStatus::Superseded
        ) {
            return Err(TraceEvalError::InvalidCandidateTransition {
                candidate_id: self.candidate_id.clone(),
                message: format!("cannot supersede candidate in {:?} state", self.status),
            });
        }
        self.validate_definition_hash()?;
        self.status = EvalCandidateStatus::Superseded;
        Ok(self)
    }

    pub fn validate_definition_hash(&self) -> Result<()> {
        if let Some(proposed_input) = &self.proposed_input {
            proposed_input.validate()?;
        }
        let actual = candidate_definition_hash(&CandidateDefinition {
            source_trace_id: &self.source_trace_id,
            source_finding_ids: &self.source_finding_ids,
            source_cluster_refs: &self.source_cluster_refs,
            evidence_packet_id: self.evidence_packet_id.as_deref(),
            proposed_input: self.proposed_input.as_ref(),
            proposed_expected_behavior: &self.proposed_expected_behavior,
            proposed_rubric: &self.proposed_rubric,
            proposed_grader: &self.proposed_grader,
            generator: &self.generator,
        });
        if actual != self.definition_hash {
            return Err(TraceEvalError::InvalidCandidateTransition {
                candidate_id: self.candidate_id.clone(),
                message: "candidate definition changed after generation".to_string(),
            });
        }
        Ok(())
    }

    fn require_status(&self, expected: EvalCandidateStatus, action: &str) -> Result<()> {
        if self.status != expected {
            return Err(TraceEvalError::InvalidCandidateTransition {
                candidate_id: self.candidate_id.clone(),
                message: format!(
                    "cannot {action} from {:?}; expected {:?}",
                    self.status, expected
                ),
            });
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct CandidateDefinition<'a> {
    source_trace_id: &'a str,
    source_finding_ids: &'a [String],
    source_cluster_refs: &'a [String],
    evidence_packet_id: Option<&'a str>,
    proposed_input: Option<&'a RedactedCandidateInput>,
    proposed_expected_behavior: &'a [String],
    proposed_rubric: &'a str,
    proposed_grader: &'a str,
    generator: &'a CandidateGenerator,
}

fn candidate_definition_hash(definition: &CandidateDefinition<'_>) -> String {
    let bytes = serde_json::to_vec(definition).expect("candidate definition serializes");
    format!("sha256:{:x}", Sha256::digest(bytes))
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
    use crate::behavior::model::{
        BEHAVIOR_FINDING_SCHEMA_VERSION, BehaviorFinding, FindingSeverity, RecoveryStatus,
    };

    #[test]
    fn generated_candidate_remains_unreviewed() {
        let mut trace = AgentBehaviorTrace::new("trace-1");
        trace.input_summary = Some("cancel the card".to_string());
        let finding = BehaviorFinding {
            schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
            finding_id: "finding-1".to_string(),
            detector_id: "uncertain_mutation_state".to_string(),
            detector_version: "2".to_string(),
            trace_id: "trace-1".to_string(),
            kind: "uncertain_mutation_state".to_string(),
            severity: FindingSeverity::High,
            recovery: RecoveryStatus::Unrecovered,
            confidence: Some(1.0),
            failure_signature: "signature-1".to_string(),
            evidence: Vec::new(),
            created_at: "2026-07-10T12:00:00Z".to_string(),
            metadata: BTreeMap::new(),
        };

        let candidate = FindingEvalCandidateGenerator.generate(&trace, &finding);

        assert_eq!(candidate.status, EvalCandidateStatus::Candidate);
        assert!(candidate.proposed_input.is_none());
        assert!(
            !serde_json::to_string(&candidate)
                .unwrap()
                .contains("cancel the card")
        );
        assert!(
            candidate
                .proposed_expected_behavior
                .iter()
                .any(|value| value.contains("Verify final state"))
        );
        assert!(candidate.definition_hash.starts_with("sha256:"));
        assert_eq!(
            candidate.generator.prompt_version.as_deref(),
            Some("deterministic_behavior_policy.v1")
        );
    }

    #[test]
    fn explicit_redacted_input_carries_hashed_provenance() {
        let trace = AgentBehaviorTrace::new("trace-1");
        let finding = BehaviorFinding {
            schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
            finding_id: "finding-1".to_string(),
            detector_id: "missing_resolution".to_string(),
            detector_version: "2".to_string(),
            trace_id: "trace-1".to_string(),
            kind: "missing_resolution".to_string(),
            severity: FindingSeverity::Medium,
            recovery: RecoveryStatus::Unrecovered,
            confidence: Some(1.0),
            failure_signature: "signature-1".to_string(),
            evidence: Vec::new(),
            created_at: "2026-07-10T12:00:00Z".to_string(),
            metadata: BTreeMap::new(),
        };
        let input = RedactedCandidateInput::new(
            "synthetic cancellation request",
            "redaction-policy-v3",
            vec![crate::behavior::EvidenceRef::new(
                "redaction_record",
                "sha256:redaction",
            )],
        )
        .unwrap();

        let candidate = FindingEvalCandidateGenerator
            .generate_with_redacted_input(&trace, &finding, input)
            .unwrap();

        let proposed_input = candidate.proposed_input.as_ref().unwrap();
        assert_eq!(proposed_input.summary(), "synthetic cancellation request");
        assert_eq!(
            proposed_input.redaction_policy_version(),
            "redaction-policy-v3"
        );
        assert!(candidate.validate_definition_hash().is_ok());

        let mut tampered = candidate;
        tampered.proposed_input = Some(
            RedactedCandidateInput::new(
                "different request",
                "redaction-policy-v3",
                vec![crate::behavior::EvidenceRef::new(
                    "redaction_record",
                    "sha256:redaction",
                )],
            )
            .unwrap(),
        );
        assert!(tampered.validate_definition_hash().is_err());
    }

    #[test]
    fn review_transition_preserves_definition_and_provenance() {
        let trace = AgentBehaviorTrace::new("trace-1");
        let finding = BehaviorFinding {
            schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
            finding_id: "finding-1".to_string(),
            detector_id: "missing_resolution".to_string(),
            detector_version: "2".to_string(),
            trace_id: "trace-1".to_string(),
            kind: "missing_resolution".to_string(),
            severity: FindingSeverity::Medium,
            recovery: RecoveryStatus::Unrecovered,
            confidence: Some(1.0),
            failure_signature: "signature-1".to_string(),
            evidence: Vec::new(),
            created_at: "2026-07-10T12:00:00Z".to_string(),
            metadata: BTreeMap::new(),
        };
        let candidate = FindingEvalCandidateGenerator.generate(&trace, &finding);
        let reviewed = candidate
            .clone()
            .record_review(CandidateReview {
                reviewer_ref: "reviewer-1".to_string(),
                reviewed_at: "2026-07-10T13:00:00Z".to_string(),
                decision: CandidateReviewDecision::Approve,
                reason: Some("matches policy".to_string()),
            })
            .unwrap();
        let accepted = reviewed.resolve_review().unwrap();

        assert_eq!(accepted.status, EvalCandidateStatus::Accepted);
        assert_eq!(accepted.review.as_ref().unwrap().reviewer_ref, "reviewer-1");

        let mut tampered = candidate;
        tampered
            .proposed_expected_behavior
            .push("silently changed expectation".to_string());
        assert!(
            tampered
                .record_review(CandidateReview {
                    reviewer_ref: "reviewer-2".to_string(),
                    reviewed_at: "2026-07-10T13:01:00Z".to_string(),
                    decision: CandidateReviewDecision::Approve,
                    reason: None,
                })
                .is_err()
        );
    }
}
