use std::collections::BTreeMap;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::Result;
use crate::behavior::{
    AgentBehaviorTrace, BEHAVIOR_FINDING_SCHEMA_VERSION, BehaviorFinding, EvidenceRef,
    FindingSeverity, RecoveryStatus,
};

use super::model::{
    SEMANTIC_BEHAVIOR_DETECTOR_ID, SEMANTIC_BEHAVIOR_DETECTOR_VERSION,
    SEMANTIC_BEHAVIOR_EVALUATION_SCHEMA_VERSION, SemanticBehaviorDetectionRun,
    SemanticBehaviorEvaluation, SemanticBehaviorEvaluator, SemanticBehaviorPolicy, SemanticVerdict,
};
use super::projection::{SemanticBehaviorProjector, hash_parts};

#[derive(Debug, Default, Clone)]
pub struct SemanticBehaviorDetector {
    projector: SemanticBehaviorProjector,
    policy: SemanticBehaviorPolicy,
}

impl SemanticBehaviorDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_projector(mut self, projector: SemanticBehaviorProjector) -> Self {
        self.projector = projector;
        self
    }

    pub fn with_policy(mut self, policy: SemanticBehaviorPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn policy(&self) -> &SemanticBehaviorPolicy {
        &self.policy
    }

    pub async fn detect_traces<E>(
        &self,
        traces: &[AgentBehaviorTrace],
        evaluator: &E,
    ) -> Result<SemanticBehaviorDetectionRun>
    where
        E: SemanticBehaviorEvaluator + ?Sized,
    {
        self.policy.validate()?;
        let evaluator_id = evaluator.evaluator_id();
        let evaluator_version = evaluator.evaluator_version();
        let rubric_hash = hash_text(&self.policy.rubric);
        let evaluator_spec_hash = evaluator_spec_hash(
            &evaluator_id,
            &evaluator_version,
            &self.policy,
            self.projector.content_policy(),
        );
        let mut projections = Vec::with_capacity(traces.len());
        let mut evaluations = Vec::with_capacity(traces.len());
        let mut results = Vec::with_capacity(traces.len());
        let mut findings = Vec::new();

        for trace in traces {
            let projection = self.projector.project(trace);
            let judgment = evaluator.evaluate(&projection, &self.policy).await?;
            judgment.validate(&projection)?;
            let evaluation = SemanticBehaviorEvaluation {
                schema_version: SEMANTIC_BEHAVIOR_EVALUATION_SCHEMA_VERSION.to_string(),
                projection_id: projection.projection_id.clone(),
                projection_hash: projection.projection_hash.clone(),
                trace_id: trace.trace_id.clone(),
                evaluator_id: evaluator_id.clone(),
                evaluator_version: evaluator_version.clone(),
                evaluator_spec_hash: evaluator_spec_hash.clone(),
                rubric_version: self.policy.rubric_version.clone(),
                rubric_hash: rubric_hash.clone(),
                judgment,
            };
            if let Some(finding) = finding_from_evaluation(
                trace,
                &projection,
                &evaluation,
                self.policy.minimum_failure_confidence,
                self.policy.emit_abstentions,
            ) {
                findings.push(finding);
            }
            results.push(evaluation.to_evaluation_result());
            evaluations.push(evaluation);
            projections.push(projection);
        }

        Ok(SemanticBehaviorDetectionRun {
            projections,
            evaluations,
            results,
            findings,
        })
    }
}

fn finding_from_evaluation(
    trace: &AgentBehaviorTrace,
    projection: &super::model::SemanticBehaviorProjection,
    evaluation: &SemanticBehaviorEvaluation,
    minimum_confidence: f32,
    emit_abstentions: bool,
) -> Option<BehaviorFinding> {
    let judgment = &evaluation.judgment;
    if judgment.verdict == SemanticVerdict::Pass
        || (judgment.verdict == SemanticVerdict::Abstain && !emit_abstentions)
    {
        return None;
    }
    let threshold_met =
        judgment.verdict == SemanticVerdict::Fail && judgment.confidence >= minimum_confidence;
    let content_eligible =
        projection.content_policy == super::model::SemanticContentPolicy::PreRedactedSummaries;
    let actionable = threshold_met && content_eligible;
    let kind = if actionable {
        judgment
            .failure_kind
            .as_deref()
            .expect("validated failed judgment has failure kind")
    } else {
        "semantic_review_required"
    };
    let severity = if actionable {
        judgment
            .severity
            .expect("validated failed judgment has severity")
    } else {
        FindingSeverity::Info
    };
    let evaluation_hash = evaluation_hash(evaluation);
    let mut evidence = projection.source_evidence(&judgment.evidence_keys);
    evidence.push(EvidenceRef::new(
        "semantic_projection",
        projection.projection_hash.clone(),
    ));
    evidence.push(EvidenceRef::new(
        "semantic_evaluation",
        evaluation_hash.clone(),
    ));
    evidence.sort_by(|left, right| left.identity.cmp(&right.identity));
    evidence.dedup_by(|left, right| left.identity == right.identity);
    let finding_id = hash_parts(
        [
            trace.trace_id.as_str(),
            SEMANTIC_BEHAVIOR_DETECTOR_ID,
            SEMANTIC_BEHAVIOR_DETECTOR_VERSION,
            evaluation.evaluator_id.as_str(),
            evaluation.evaluator_version.as_str(),
            evaluation.rubric_hash.as_str(),
        ]
        .into_iter()
        .chain(evidence.iter().map(|evidence| evidence.identity.as_str())),
    );
    let failure_signature = hash_parts([
        SEMANTIC_BEHAVIOR_DETECTOR_ID,
        kind,
        evaluation.rubric_version.as_str(),
    ]);
    let mut metadata = BTreeMap::from([
        ("semantic_threshold_met".to_string(), json!(threshold_met)),
        (
            "semantic_content_eligible".to_string(),
            json!(content_eligible),
        ),
        ("semantic_actionable".to_string(), json!(actionable)),
        ("requires_human_review".to_string(), json!(true)),
        ("semantic_verdict".to_string(), json!(judgment.verdict)),
        ("semantic_score".to_string(), json!(judgment.score)),
        (
            "semantic_confidence".to_string(),
            json!(judgment.confidence),
        ),
        ("semantic_criteria".to_string(), json!(judgment.criteria)),
        (
            "semantic_evaluator_id".to_string(),
            json!(evaluation.evaluator_id),
        ),
        (
            "semantic_evaluator_version".to_string(),
            json!(evaluation.evaluator_version),
        ),
        (
            "evaluator_spec_hash".to_string(),
            json!(evaluation.evaluator_spec_hash),
        ),
        (
            "semantic_projection_hash".to_string(),
            json!(projection.projection_hash),
        ),
        (
            "semantic_content_policy".to_string(),
            json!(projection.content_policy),
        ),
        (
            "semantic_rubric_version".to_string(),
            json!(evaluation.rubric_version),
        ),
        (
            "semantic_rubric_hash".to_string(),
            json!(evaluation.rubric_hash),
        ),
        (
            "semantic_evaluation_hash".to_string(),
            json!(evaluation_hash),
        ),
        (
            "semantic_cited_evidence_count".to_string(),
            json!(judgment.evidence_keys.len()),
        ),
    ]);
    if let Some(reported_kind) = &judgment.failure_kind {
        metadata.insert(
            "semantic_reported_failure_kind".to_string(),
            json!(reported_kind),
        );
    }
    copy_adapter_provenance(trace, &mut metadata);

    Some(BehaviorFinding {
        schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
        finding_id,
        detector_id: SEMANTIC_BEHAVIOR_DETECTOR_ID.to_string(),
        detector_version: SEMANTIC_BEHAVIOR_DETECTOR_VERSION.to_string(),
        trace_id: trace.trace_id.clone(),
        kind: kind.to_string(),
        severity,
        recovery: RecoveryStatus::Unknown,
        confidence: Some(judgment.confidence),
        failure_signature,
        evidence,
        created_at: observed_at(trace),
        metadata,
    })
}

fn evaluator_spec_hash(
    evaluator_id: &str,
    evaluator_version: &str,
    policy: &SemanticBehaviorPolicy,
    content_policy: super::model::SemanticContentPolicy,
) -> String {
    let minimum_confidence = policy.minimum_failure_confidence.to_bits().to_string();
    hash_parts([
        SEMANTIC_BEHAVIOR_DETECTOR_VERSION,
        evaluator_id,
        evaluator_version,
        policy.rubric_version.as_str(),
        policy.rubric.as_str(),
        minimum_confidence.as_str(),
        if policy.emit_abstentions {
            "true"
        } else {
            "false"
        },
        match content_policy {
            super::model::SemanticContentPolicy::StructuredOnly => "structured_only",
            super::model::SemanticContentPolicy::PreRedactedSummaries => "pre_redacted_summaries",
        },
    ])
}

fn evaluation_hash(evaluation: &SemanticBehaviorEvaluation) -> String {
    let bytes = serde_json::to_vec(evaluation).expect("semantic evaluation serializes");
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn hash_text(value: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(value.as_bytes()))
}

fn copy_adapter_provenance(trace: &AgentBehaviorTrace, metadata: &mut BTreeMap<String, Value>) {
    for key in [
        "traceeval.behavior_adapter.id",
        "traceeval.behavior_adapter.version",
    ] {
        if let Some(value) = trace.metadata.get(key) {
            metadata.insert(key.to_string(), value.clone());
        }
    }
}

fn observed_at(trace: &AgentBehaviorTrace) -> String {
    trace
        .metadata
        .get("observed_at")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            trace
                .tool_calls
                .last()
                .map(|call| call.started_at.clone())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string())
}
