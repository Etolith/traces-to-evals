use anyhow::Result;

use crate::behavior::{
    ApprovalBypassDetector, DeterministicDetectorSet, ExcessiveToolUsageDetector,
    FalseSuccessClaimDetector, FindingEvalCandidateGenerator, MissingResolutionDetector,
    OpenInferenceBehaviorNormalizer, PolicyViolationDetector, RepeatedToolFailureDetector,
    TerminalToolFailureDetector, ToolCallLoopDetector, UncertainMutationStateDetector,
    UnresolvedEscalationDetector,
};
use crate::cli::{BehaviorTraceFormat, DetectArgs};
use crate::io::jsonl::JsonlFile;
use crate::model::Trace;
use crate::{AgentBehaviorNormalizer, TraceDetector};

pub fn run(args: DetectArgs) -> Result<()> {
    let traces: Vec<Trace> = JsonlFile::new(&args.traces).read_all()?;
    let normalized = match args.format {
        BehaviorTraceFormat::OpenInference => {
            OpenInferenceBehaviorNormalizer::default().normalize_traces(&traces)?
        }
    };
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
    let findings = detectors.detect_traces(&normalized);

    JsonlFile::new(&args.out).write_all(&findings)?;
    if let Some(path) = &args.normalized_out {
        JsonlFile::new(path).write_all(&normalized)?;
    }
    if let Some(path) = &args.candidates_out {
        let candidates = FindingEvalCandidateGenerator.generate_all(&normalized, &findings);
        JsonlFile::new(path).write_all(&candidates)?;
    }
    Ok(())
}
