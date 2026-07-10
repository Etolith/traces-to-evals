use std::collections::BTreeMap;

use anyhow::{Result, anyhow};

use crate::behavior::{
    AgentBehaviorTrace, ApprovalBypassDetector, BehaviorAdapterConfig, BehaviorFinding,
    DeterministicDetectorSet, EvidencePacketBuilder, ExcessiveToolUsageDetector,
    FalseSuccessClaimDetector, FindingEvalCandidateGenerator, KnownSignatureGrouper,
    MissingResolutionDetector, OpenInferenceBehaviorNormalizer, PolicyViolationDetector,
    RepeatedToolFailureDetector, SafeFindingProjector, TerminalToolFailureDetector,
    ToolCallLoopDetector, UncertainMutationStateDetector, UnresolvedEscalationDetector,
    finding_projection_cases,
};
#[cfg(feature = "llm-judge-openai")]
use crate::behavior::{
    DEFAULT_SEMANTIC_BEHAVIOR_RUBRIC, OpenAiSemanticBehaviorEvaluator, SemanticBehaviorDetector,
    SemanticBehaviorEvaluator, SemanticBehaviorPolicy, SemanticBehaviorProjector,
    SemanticContentPolicy,
};
use crate::cli::{BehaviorTraceFormat, DetectArgs};
#[cfg(feature = "llm-judge-openai")]
use crate::cli::{JudgeProviderName, SemanticContentPolicyName};
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlFile;
use crate::model::Trace;
use crate::{AgentBehaviorNormalizer, TraceDetector};

#[cfg(feature = "llm-judge-openai")]
pub async fn run(args: DetectArgs) -> Result<()> {
    if let Some(provider) = args.semantic_judge {
        let model = args
            .semantic_model
            .as_deref()
            .ok_or_else(|| anyhow!("--semantic-model is required for --semantic-judge"))?
            .to_string();
        return match provider {
            JudgeProviderName::OpenaiDive => {
                run_with_semantic_evaluator(args, &OpenAiSemanticBehaviorEvaluator::from_env(model))
                    .await
            }
        };
    }

    let normalized = normalize(&args)?;
    let findings = deterministic_findings(&args, &normalized);
    write_outputs(&args, &normalized, &findings)
}

#[cfg(feature = "llm-judge-openai")]
async fn run_with_semantic_evaluator<E>(args: DetectArgs, evaluator: &E) -> Result<()>
where
    E: SemanticBehaviorEvaluator + ?Sized,
{
    let normalized = normalize(&args)?;
    let mut findings = deterministic_findings(&args, &normalized);
    let rubric = match &args.semantic_rubric_file {
        Some(path) => std::fs::read_to_string(path)?,
        None => DEFAULT_SEMANTIC_BEHAVIOR_RUBRIC.to_string(),
    };
    let policy = SemanticBehaviorPolicy {
        rubric_version: args.semantic_rubric_version.clone(),
        rubric,
        minimum_failure_confidence: args.semantic_min_confidence,
        emit_abstentions: !args.semantic_ignore_abstentions,
    };
    let content_policy = match args.semantic_content {
        SemanticContentPolicyName::StructuredOnly => SemanticContentPolicy::StructuredOnly,
        SemanticContentPolicyName::PreRedactedSummaries => {
            SemanticContentPolicy::PreRedactedSummaries
        }
    };
    let semantic_run = SemanticBehaviorDetector::new()
        .with_projector(SemanticBehaviorProjector::new().with_content_policy(content_policy))
        .with_policy(policy)
        .detect_traces(&normalized, evaluator)
        .await?;
    if let Some(path) = &args.semantic_results_out {
        JsonlFile::new(path).write_all(&semantic_run.results)?;
    }
    if let Some(path) = &args.semantic_projections_out {
        JsonlFile::new(path).write_all(&semantic_run.projections)?;
    }
    findings.extend(semantic_run.findings);
    write_outputs(&args, &normalized, &findings)
}

#[cfg(not(feature = "llm-judge-openai"))]
pub fn run(args: DetectArgs) -> Result<()> {
    if args.semantic_judge.is_some() {
        return Err(anyhow!(
            "semantic behavior judging requires rebuilding with --features llm-judge-openai"
        ));
    }
    let normalized = normalize(&args)?;
    let findings = deterministic_findings(&args, &normalized);
    write_outputs(&args, &normalized, &findings)
}

fn normalize(args: &DetectArgs) -> Result<Vec<AgentBehaviorTrace>> {
    let traces: Vec<Trace> = JsonlFile::new(&args.traces).read_all()?;
    match args.format {
        BehaviorTraceFormat::OpenInference => match &args.adapter_config {
            Some(path) => {
                let adapter: BehaviorAdapterConfig = JsonFile::new(path).read()?;
                Ok(OpenInferenceBehaviorNormalizer::from_adapter(adapter)?
                    .normalize_traces(&traces)?)
            }
            None => Ok(OpenInferenceBehaviorNormalizer::default().normalize_traces(&traces)?),
        },
    }
}

fn deterministic_findings(
    args: &DetectArgs,
    normalized: &[AgentBehaviorTrace],
) -> Vec<BehaviorFinding> {
    let detectors = DeterministicDetectorSet::new(vec![
        Box::new(TerminalToolFailureDetector) as Box<dyn TraceDetector>,
        Box::new(RepeatedToolFailureDetector::new(args.max_repeated_failures)),
        Box::new(ToolCallLoopDetector::new(args.max_equivalent_calls)),
        Box::new(UncertainMutationStateDetector),
        Box::new(FalseSuccessClaimDetector),
        Box::new(ApprovalBypassDetector),
        Box::new(PolicyViolationDetector),
        Box::new(ExcessiveToolUsageDetector::new(
            args.max_tool_calls,
            args.max_total_tool_duration_ms,
        )),
        Box::new(UnresolvedEscalationDetector),
        Box::new(MissingResolutionDetector),
    ]);
    detectors.detect_traces(normalized)
}

fn write_outputs(
    args: &DetectArgs,
    normalized: &[AgentBehaviorTrace],
    findings: &[BehaviorFinding],
) -> Result<()> {
    JsonlFile::new(&args.out).write_all(findings)?;
    if let Some(path) = &args.normalized_out {
        JsonlFile::new(path).write_all(normalized)?;
    }
    let evidence_packet = (args.candidates_out.is_some() || args.evidence_packet_out.is_some())
        .then(|| EvidencePacketBuilder.build(normalized, findings));
    if let (Some(path), Some(packet)) = (&args.evidence_packet_out, &evidence_packet) {
        JsonFile::new(path).write_pretty(packet)?;
    }
    if let Some(path) = &args.candidates_out {
        let candidates = FindingEvalCandidateGenerator.generate_all_with_evidence_packet(
            normalized,
            findings,
            evidence_packet
                .as_ref()
                .expect("candidate output creates an evidence packet"),
        );
        JsonlFile::new(path).write_all(&candidates)?;
    }
    write_projections(args, normalized, findings)?;
    if let Some(path) = &args.signature_groups_out {
        let groups = KnownSignatureGrouper.group(findings);
        JsonlFile::new(path).write_all(&groups)?;
    }
    Ok(())
}

fn write_projections(
    args: &DetectArgs,
    normalized: &[AgentBehaviorTrace],
    findings: &[BehaviorFinding],
) -> Result<()> {
    if args.projections_out.is_none() && args.projection_cases_out.is_none() {
        return Ok(());
    }
    let projector = args.projection_metadata_keys.iter().fold(
        SafeFindingProjector::default().with_max_bytes(args.projection_max_bytes),
        |projector, field| projector.with_allowed_metadata_field(field),
    );
    let contexts = normalized
        .iter()
        .map(|trace| (trace.trace_id.as_str(), &trace.metadata))
        .collect::<BTreeMap<_, _>>();
    let empty_context = BTreeMap::new();
    let projections = findings
        .iter()
        .map(|finding| {
            projector.project_with_context(
                finding,
                contexts
                    .get(finding.trace_id.as_str())
                    .copied()
                    .unwrap_or(&empty_context),
            )
        })
        .collect::<Vec<_>>();
    if let Some(path) = &args.projections_out {
        JsonlFile::new(path).write_all(&projections)?;
    }
    if let Some(path) = &args.projection_cases_out {
        JsonlFile::new(path).write_all(&finding_projection_cases(&projections))?;
    }
    Ok(())
}

#[cfg(all(test, feature = "llm-judge-openai"))]
mod tests {
    use clap::Parser;
    use tempfile::tempdir;

    use crate::behavior::{FindingSeverity, SemanticBehaviorJudgment, SemanticVerdict};
    use crate::cli::{Cli, Command};
    use crate::evaluation::EvaluationCriteria;

    use super::*;

    struct FakeSemanticEvaluator;

    #[async_trait::async_trait]
    impl SemanticBehaviorEvaluator for FakeSemanticEvaluator {
        fn evaluator_id(&self) -> String {
            "fake/semantic".to_string()
        }

        fn evaluator_version(&self) -> String {
            "1".to_string()
        }

        async fn evaluate(
            &self,
            _projection: &crate::behavior::SemanticBehaviorProjection,
            _policy: &SemanticBehaviorPolicy,
        ) -> crate::Result<SemanticBehaviorJudgment> {
            Ok(SemanticBehaviorJudgment {
                verdict: SemanticVerdict::Fail,
                score: 2,
                failure_kind: Some("incomplete_resolution".to_string()),
                severity: Some(FindingSeverity::Medium),
                confidence: 0.95,
                summary: "A concrete incomplete resolution is present.".to_string(),
                criteria: EvaluationCriteria {
                    relevance: true,
                    correctness: true,
                    completeness: false,
                    safety: true,
                },
                evidence_keys: vec!["e1".to_string()],
            })
        }
    }

    #[test]
    fn semantic_cli_path_merges_findings_and_writes_auditable_artifacts() {
        let dir = tempdir().unwrap();
        let findings = dir.path().join("findings.jsonl");
        let results = dir.path().join("semantic-results.jsonl");
        let projections = dir.path().join("semantic-projections.jsonl");
        let candidates = dir.path().join("candidates.jsonl");
        let cli = Cli::parse_from([
            "traceeval",
            "detect",
            "--traces",
            "fixtures/behavior/traces.jsonl",
            "--adapter-config",
            "fixtures/behavior/adapter.json",
            "--out",
            findings.to_str().unwrap(),
            "--candidates-out",
            candidates.to_str().unwrap(),
            "--semantic-judge",
            "openai-dive",
            "--semantic-model",
            "fake",
            "--semantic-results-out",
            results.to_str().unwrap(),
            "--semantic-projections-out",
            projections.to_str().unwrap(),
        ]);
        let Command::Detect(args) = cli.command else {
            panic!("expected detect command");
        };

        futures::executor::block_on(run_with_semantic_evaluator(args, &FakeSemanticEvaluator))
            .unwrap();

        let finding_rows = std::fs::read_to_string(findings).unwrap();
        assert_eq!(finding_rows.lines().count(), 8);
        assert!(finding_rows.contains("semantic_behavior_judge"));
        assert_eq!(std::fs::read_to_string(results).unwrap().lines().count(), 4);
        assert_eq!(
            std::fs::read_to_string(projections)
                .unwrap()
                .lines()
                .count(),
            4
        );
        assert_eq!(
            std::fs::read_to_string(candidates).unwrap().lines().count(),
            8
        );
    }
}
