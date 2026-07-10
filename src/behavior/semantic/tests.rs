use std::collections::BTreeMap;

use futures::executor::block_on;
use serde_json::json;

use crate::Result;
use crate::behavior::{
    AgentBehaviorTrace, AgentRole, AgentTurn, EvidenceRef, FinalOutcomeStatus, FindingSeverity,
    SemanticBehaviorEvaluator, SemanticBehaviorJudgment, SemanticContentPolicy, SemanticVerdict,
    StateChangeRef, StateObservation, ToolCallFact, ToolCallStatus,
};
use crate::evaluation::EvaluationCriteria;

use super::{SemanticBehaviorDetector, SemanticBehaviorPolicy, SemanticBehaviorProjector};

#[derive(Clone)]
struct FakeEvaluator {
    judgment: SemanticBehaviorJudgment,
}

#[async_trait::async_trait]
impl SemanticBehaviorEvaluator for FakeEvaluator {
    fn evaluator_id(&self) -> String {
        "fake/semantic".to_string()
    }

    fn evaluator_version(&self) -> String {
        "1".to_string()
    }

    async fn evaluate(
        &self,
        _projection: &super::SemanticBehaviorProjection,
        _policy: &SemanticBehaviorPolicy,
    ) -> Result<SemanticBehaviorJudgment> {
        Ok(self.judgment.clone())
    }
}

fn criteria(completeness: bool) -> EvaluationCriteria {
    EvaluationCriteria {
        relevance: true,
        correctness: true,
        completeness,
        safety: true,
    }
}

fn trace() -> AgentBehaviorTrace {
    let mut trace = AgentBehaviorTrace::new("trace-1");
    trace.input_summary = Some("private user account 123".to_string());
    trace.turns.push(AgentTurn {
        turn_id: "turn-1".to_string(),
        role: AgentRole::Assistant,
        content_summary: Some("private assistant response".to_string()),
        evidence: vec![EvidenceRef::span("root")],
    });
    trace.tool_calls.push(ToolCallFact {
        call_id: "call-private".to_string(),
        tool_name: "unsafe tool name with customer 123".to_string(),
        operation: Some("update".to_string()),
        effect: crate::behavior::OperationEffect::Mutating,
        retry_safety: crate::behavior::RetrySafety::NonIdempotent,
        requirement: crate::behavior::ToolRequirement::Required,
        attempt: 1,
        started_at: "2026-07-10T12:00:00Z".to_string(),
        duration_ms: 15,
        status: ToolCallStatus::Succeeded,
        error: None,
        approval_required: false,
        approval_outcome: None,
        state_change: Some(StateChangeRef {
            predicate: Some("account_updated".to_string()),
            observation: StateObservation::Unverified,
            artifact: EvidenceRef::new("state_change", "state:private"),
        }),
        evidence: vec![EvidenceRef::span("tool-1")],
    });
    trace.final_outcome.status = FinalOutcomeStatus::Incomplete;
    trace.final_outcome.evidence = vec![EvidenceRef::span("root")];
    trace.evidence.push(EvidenceRef::span("root"));
    trace.metadata.insert(
        "credential".to_string(),
        json!("must-never-enter-projection"),
    );
    trace
}

#[test]
fn structured_projection_omits_content_and_arbitrary_metadata() {
    let projection = SemanticBehaviorProjector::new().project(&trace());
    let serialized = serde_json::to_string(&projection).unwrap();

    assert_eq!(
        projection.content_policy,
        SemanticContentPolicy::StructuredOnly
    );
    assert!(projection.input_summary.is_none());
    assert!(projection.final_response_summary.is_none());
    assert!(!serialized.contains("private user account"));
    assert!(!serialized.contains("private assistant response"));
    assert!(!serialized.contains("must-never-enter-projection"));
    assert!(!serialized.contains("unsafe tool name"));
    assert!(projection.tool_calls[0].tool_name.starts_with("sha256:"));
    assert!(projection.projection_hash.starts_with("sha256:"));
    assert!(!projection.evidence.is_empty());
}

#[test]
fn pre_redacted_content_requires_explicit_policy_and_is_bounded() {
    let projection = SemanticBehaviorProjector::new()
        .with_content_policy(SemanticContentPolicy::PreRedactedSummaries)
        .with_max_summary_chars(7)
        .project(&trace());

    assert_eq!(projection.input_summary.as_deref(), Some("private"));
    assert_eq!(
        projection.final_response_summary.as_deref(),
        Some("private")
    );
    assert!(projection.truncated);
}

#[test]
fn thresholded_failure_with_pre_redacted_content_becomes_specific_finding() {
    let evaluator = FakeEvaluator {
        judgment: SemanticBehaviorJudgment {
            verdict: SemanticVerdict::Fail,
            score: 2,
            failure_kind: Some("incomplete_resolution".to_string()),
            severity: Some(FindingSeverity::Medium),
            confidence: 0.94,
            summary: "The behavior ends without resolution.".to_string(),
            criteria: criteria(false),
            evidence_keys: vec!["e1".to_string()],
        },
    };

    let detector = SemanticBehaviorDetector::new().with_projector(
        SemanticBehaviorProjector::new()
            .with_content_policy(SemanticContentPolicy::PreRedactedSummaries),
    );
    let run = block_on(detector.detect_traces(&[trace()], &evaluator)).unwrap();

    assert_eq!(run.projections.len(), 1);
    assert_eq!(run.evaluations.len(), 1);
    assert_eq!(run.results.len(), 1);
    assert_eq!(run.findings.len(), 1);
    let finding = &run.findings[0];
    assert_eq!(finding.kind, "incomplete_resolution");
    assert_eq!(finding.severity, FindingSeverity::Medium);
    assert_eq!(finding.confidence, Some(0.94));
    assert_eq!(finding.metadata["semantic_threshold_met"], true);
    assert_eq!(finding.metadata["semantic_content_eligible"], true);
    assert_eq!(finding.metadata["semantic_actionable"], true);
    assert_eq!(finding.metadata["requires_human_review"], true);
    assert!(
        finding
            .evidence
            .iter()
            .any(|evidence| evidence.kind == "semantic_projection")
    );
    assert!(!run.results[0].passed);
    assert!(
        run.results[0].metadata["evaluator_spec_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );
}

#[test]
fn abstention_becomes_informational_review_finding() {
    let evaluator = FakeEvaluator {
        judgment: SemanticBehaviorJudgment {
            verdict: SemanticVerdict::Abstain,
            score: 2,
            failure_kind: None,
            severity: None,
            confidence: 0.4,
            summary: "The user intent was omitted.".to_string(),
            criteria: criteria(false),
            evidence_keys: Vec::new(),
        },
    };

    let run =
        block_on(SemanticBehaviorDetector::new().detect_traces(&[trace()], &evaluator)).unwrap();

    assert_eq!(run.findings.len(), 1);
    assert_eq!(run.findings[0].kind, "semantic_review_required");
    assert_eq!(run.findings[0].severity, FindingSeverity::Info);
    assert_eq!(run.findings[0].metadata["semantic_actionable"], false);
}

#[test]
fn structured_only_failure_is_review_only_even_above_confidence_threshold() {
    let evaluator = FakeEvaluator {
        judgment: SemanticBehaviorJudgment {
            verdict: SemanticVerdict::Fail,
            score: 1,
            failure_kind: Some("unsafe_action".to_string()),
            severity: Some(FindingSeverity::High),
            confidence: 0.99,
            summary: "The structured facts may require review.".to_string(),
            criteria: criteria(false),
            evidence_keys: vec!["e1".to_string()],
        },
    };

    let run =
        block_on(SemanticBehaviorDetector::new().detect_traces(&[trace()], &evaluator)).unwrap();

    assert_eq!(run.findings[0].kind, "semantic_review_required");
    assert_eq!(run.findings[0].severity, FindingSeverity::Info);
    assert_eq!(run.findings[0].metadata["semantic_threshold_met"], true);
    assert_eq!(run.findings[0].metadata["semantic_content_eligible"], false);
    assert_eq!(run.findings[0].metadata["semantic_actionable"], false);
}

#[test]
fn invalid_model_evidence_is_rejected_instead_of_hallucinated() {
    let evaluator = FakeEvaluator {
        judgment: SemanticBehaviorJudgment {
            verdict: SemanticVerdict::Fail,
            score: 1,
            failure_kind: Some("unsafe_action".to_string()),
            severity: Some(FindingSeverity::High),
            confidence: 0.99,
            summary: "Unsupported.".to_string(),
            criteria: criteria(false),
            evidence_keys: vec!["not-in-projection".to_string()],
        },
    };

    let error = block_on(SemanticBehaviorDetector::new().detect_traces(&[trace()], &evaluator))
        .unwrap_err();

    assert!(error.to_string().contains("evidence_keys"));
}

#[test]
fn pass_does_not_create_a_finding() {
    let evaluator = FakeEvaluator {
        judgment: SemanticBehaviorJudgment {
            verdict: SemanticVerdict::Pass,
            score: 4,
            failure_kind: None,
            severity: None,
            confidence: 0.9,
            summary: "The projected behavior satisfies the rubric.".to_string(),
            criteria: criteria(true),
            evidence_keys: Vec::new(),
        },
    };

    let run =
        block_on(SemanticBehaviorDetector::new().detect_traces(&[trace()], &evaluator)).unwrap();

    assert!(run.findings.is_empty());
    assert!(run.results[0].passed);
}

#[test]
fn semantic_policy_rejects_invalid_confidence_threshold() {
    let policy = SemanticBehaviorPolicy {
        minimum_failure_confidence: f32::NAN,
        ..SemanticBehaviorPolicy::default()
    };

    assert!(policy.validate().is_err());
}

#[test]
fn projection_is_deterministic() {
    let projector = SemanticBehaviorProjector::new();
    let first = projector.project(&trace());
    let second = projector.project(&trace());

    assert_eq!(first, second);
    assert_eq!(
        first
            .evidence
            .iter()
            .map(|item| &item.key)
            .collect::<Vec<_>>(),
        second
            .evidence
            .iter()
            .map(|item| &item.key)
            .collect::<Vec<_>>()
    );
}

#[test]
fn arbitrary_metadata_does_not_change_projection_identity() {
    let projector = SemanticBehaviorProjector::new();
    let first = projector.project(&trace());
    let mut changed = trace();
    changed.metadata = BTreeMap::from([("different_secret".to_string(), json!("value"))]);
    let second = projector.project(&changed);

    assert_eq!(first.projection_hash, second.projection_hash);
}
