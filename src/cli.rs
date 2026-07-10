use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::commands;

mod cluster;

pub use cluster::*;

#[derive(Debug, Parser)]
#[command(about = "Turn traces into eval cases and score them")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Extract eval cases from trace JSONL.
    Extract(ExtractArgs),
    /// Normalize agent behavior and emit deterministic and optional semantic findings.
    Detect(DetectArgs),
    /// Compare paired baseline and candidate behavior findings.
    VerifyFindings(VerifyFindingsArgs),
    /// Verify all paired offline remediation gates.
    VerifyRemediation(VerifyRemediationArgs),
    /// Compare baseline and candidate finding recurrence windows.
    CompareRecurrence(CompareRecurrenceArgs),
    /// Grade eval cases with a deterministic grader or judge provider.
    Grade(GradeArgs),
    /// Validate eval cases and evaluation results.
    Validate(ValidateArgs),
    /// Fit a calibration model from historical results and human ratings.
    Calibrate(CalibrateArgs),
    /// Assign cases and optional results to clusters.
    Cluster(ClusterArgs),
    /// Build an aggregate evaluation report.
    Report(ReportArgs),
}

#[derive(Debug, Clone, Args)]
pub struct VerifyFindingsArgs {
    /// Stable regression case identity.
    #[arg(long = "case-id")]
    pub case_id: String,
    /// Baseline behavior findings JSONL.
    #[arg(long)]
    pub baseline: PathBuf,
    /// Candidate behavior findings JSONL.
    #[arg(long)]
    pub candidate: PathBuf,
    /// Expected baseline failure signature that the candidate must remove.
    #[arg(long = "target-signature", required = true)]
    pub target_signatures: Vec<String>,
    /// Severity at which a novel candidate finding fails the gate.
    #[arg(long, value_enum, default_value = "high")]
    pub severe_threshold: FindingSeverityName,
    /// Paired finding verification JSON output.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct VerifyRemediationArgs {
    /// Versioned remediation verification request JSON.
    #[arg(long)]
    pub request: PathBuf,
    /// Baseline behavior findings JSONL.
    #[arg(long = "baseline-findings")]
    pub baseline_findings: PathBuf,
    /// Candidate behavior findings JSONL.
    #[arg(long = "candidate-findings")]
    pub candidate_findings: PathBuf,
    /// Baseline evaluation results JSONL.
    #[arg(long = "baseline-results")]
    pub baseline_results: PathBuf,
    /// Candidate evaluation results JSONL.
    #[arg(long = "candidate-results")]
    pub candidate_results: PathBuf,
    /// Combined remediation verification JSON output.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct CompareRecurrenceArgs {
    /// Versioned recurrence comparison request JSON.
    #[arg(long)]
    pub request: PathBuf,
    /// Baseline-window behavior findings JSONL.
    #[arg(long = "baseline-findings")]
    pub baseline_findings: PathBuf,
    /// Candidate-window behavior findings JSONL.
    #[arg(long = "candidate-findings")]
    pub candidate_findings: PathBuf,
    /// Finding recurrence comparison JSON output.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct DetectArgs {
    /// Input trace JSONL file.
    #[arg(long)]
    pub traces: PathBuf,
    /// Input trace format to normalize.
    #[arg(long, value_enum, default_value = "openinference")]
    pub format: BehaviorTraceFormat,
    /// Optional versioned application tool-semantics adapter JSON.
    #[arg(long = "adapter-config")]
    pub adapter_config: Option<PathBuf>,
    /// Output behavior findings JSONL file.
    #[arg(long)]
    pub out: PathBuf,
    /// Optional semantic behavior judge provider.
    #[arg(long = "semantic-judge", value_enum, requires = "semantic_model")]
    pub semantic_judge: Option<JudgeProviderName>,
    /// Model name for semantic behavior judging.
    #[arg(long = "semantic-model", requires = "semantic_judge")]
    pub semantic_model: Option<String>,
    /// Optional semantic behavior evaluation results JSONL output.
    #[arg(long = "semantic-results-out", requires = "semantic_judge")]
    pub semantic_results_out: Option<PathBuf>,
    /// Optional bounded semantic behavior projections JSONL output.
    #[arg(long = "semantic-projections-out", requires = "semantic_judge")]
    pub semantic_projections_out: Option<PathBuf>,
    /// Optional UTF-8 rubric file for semantic behavior judging.
    #[arg(long = "semantic-rubric-file", requires = "semantic_judge")]
    pub semantic_rubric_file: Option<PathBuf>,
    /// Stable version for the configured semantic rubric.
    #[arg(
        long = "semantic-rubric-version",
        default_value = "traceeval.semantic_behavior_rubric.v1"
    )]
    pub semantic_rubric_version: String,
    /// Confidence required to classify an eligible semantic failure as actionable.
    #[arg(long = "semantic-min-confidence", default_value_t = 0.8)]
    pub semantic_min_confidence: f32,
    /// Content allowed into the semantic projection.
    #[arg(
        long = "semantic-content",
        value_enum,
        default_value = "structured-only"
    )]
    pub semantic_content: SemanticContentPolicyName,
    /// Do not emit informational findings when the semantic judge abstains.
    #[arg(long = "semantic-ignore-abstentions")]
    pub semantic_ignore_abstentions: bool,
    /// Optional normalized agent behavior JSONL output.
    #[arg(long = "normalized-out")]
    pub normalized_out: Option<PathBuf>,
    /// Optional unreviewed eval candidate JSONL output.
    #[arg(long = "candidates-out")]
    pub candidates_out: Option<PathBuf>,
    /// Optional immutable evidence packet JSON output.
    #[arg(long = "evidence-packet-out")]
    pub evidence_packet_out: Option<PathBuf>,
    /// Optional safe semantic finding projections JSONL output.
    #[arg(long = "projections-out")]
    pub projections_out: Option<PathBuf>,
    /// Optional safe projection eval cases JSONL for embedding/clustering.
    #[arg(long = "projection-cases-out")]
    pub projection_cases_out: Option<PathBuf>,
    /// Optional exact failure-signature groups JSONL output.
    #[arg(long = "signature-groups-out")]
    pub signature_groups_out: Option<PathBuf>,
    /// Additional trace/finding metadata field allowed into projections.
    #[arg(long = "projection-metadata-key")]
    pub projection_metadata_keys: Vec<String>,
    /// Maximum UTF-8 bytes in each finding projection.
    #[arg(long, default_value_t = 4_096)]
    pub projection_max_bytes: usize,
    /// Failed equivalent calls that trigger repeated-failure detection.
    #[arg(long, default_value_t = 3)]
    pub max_repeated_failures: usize,
    /// Equivalent calls without progress that trigger loop detection.
    #[arg(long, default_value_t = 4)]
    pub max_equivalent_calls: usize,
    /// Maximum tool calls allowed per trace.
    #[arg(long, default_value_t = 25)]
    pub max_tool_calls: usize,
    /// Maximum total tool latency allowed per trace.
    #[arg(long, default_value_t = 60_000)]
    pub max_total_tool_duration_ms: u64,
}

#[derive(Debug, Clone, Args)]
pub struct ExtractArgs {
    /// Input trace JSONL file.
    #[arg(long)]
    pub traces: PathBuf,
    /// Trace format to extract.
    #[arg(long, value_enum, default_value = "simple")]
    pub format: ExtractFormat,
    /// Output eval cases JSONL file.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct GradeArgs {
    /// Input eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Deterministic grader to run.
    #[arg(long, value_enum, conflicts_with = "judge")]
    pub grader: Option<DeterministicGraderName>,
    /// Phrase required by the `contains` grader.
    #[arg(long, requires = "grader")]
    pub contains: Option<String>,
    /// Judge provider to run.
    #[arg(long, value_enum, conflicts_with = "grader")]
    pub judge: Option<JudgeProviderName>,
    /// Model name for judge providers.
    #[arg(long, requires = "judge")]
    pub model: Option<String>,
    /// Output JSONL file.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct ValidateArgs {
    /// Input eval cases JSONL file.
    #[arg(long)]
    pub cases: Option<PathBuf>,
    /// Input evaluation results JSONL file.
    #[arg(long)]
    pub results: Option<PathBuf>,
    /// Input case embeddings JSONL file.
    #[arg(long)]
    pub embeddings: Option<PathBuf>,
    /// Input cluster model JSON file.
    #[arg(long = "cluster-model")]
    pub cluster_model: Option<PathBuf>,
    /// Input cluster assignments JSONL file.
    #[arg(long)]
    pub assignments: Option<PathBuf>,
    /// Validation profile to apply.
    #[arg(long, value_enum)]
    pub profile: Option<ValidationProfileName>,
    /// Optional validation report JSON output file.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct CalibrateArgs {
    /// Human ratings JSONL file.
    #[arg(long = "human-ratings")]
    pub human_ratings: PathBuf,
    /// Historical evaluation results JSONL file.
    #[arg(long)]
    pub results: PathBuf,
    /// Human score threshold treated as pass.
    #[arg(long = "pass-threshold", default_value_t = 3)]
    pub pass_threshold: u8,
    /// Output calibration model JSON file.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct ReportArgs {
    /// Evaluation results JSONL file.
    #[arg(long)]
    pub results: PathBuf,
    /// Optional calibration model JSON file.
    #[arg(long)]
    pub calibration: Option<PathBuf>,
    /// Optional cluster definitions JSONL file for cluster weights.
    #[arg(long)]
    pub clusters: Option<PathBuf>,
    /// Output report JSON file.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ExtractFormat {
    Simple,
    #[value(name = "openinference", alias = "open-inference")]
    OpenInference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BehaviorTraceFormat {
    #[value(name = "openinference", alias = "open-inference")]
    OpenInference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FindingSeverityName {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ValidationProfileName {
    DraftCases,
    RunnableCases,
    EvaluationResults,
    CalibrationDataset,
    EmbeddingDataset,
    ClusterModel,
    ClusterAssignments,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DeterministicGraderName {
    NonEmptyOutput,
    ExactMatch,
    Contains,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum JudgeProviderName {
    OpenaiDive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SemanticContentPolicyName {
    StructuredOnly,
    PreRedactedSummaries,
}

#[cfg(any(
    feature = "llm-judge-openai",
    feature = "embeddings-openai",
    feature = "cluster-label-openai"
))]
pub async fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Extract(args) => commands::extract::run(args),
        Command::Detect(args) => {
            #[cfg(feature = "llm-judge-openai")]
            {
                commands::detect::run(args).await
            }
            #[cfg(not(feature = "llm-judge-openai"))]
            {
                commands::detect::run(args)
            }
        }
        Command::VerifyFindings(args) => commands::verify_findings::run(args),
        Command::VerifyRemediation(args) => commands::verify_remediation::run(args),
        Command::CompareRecurrence(args) => commands::compare_recurrence::run(args),
        Command::Grade(args) => {
            #[cfg(feature = "llm-judge-openai")]
            {
                commands::grade::run(args).await
            }
            #[cfg(not(feature = "llm-judge-openai"))]
            {
                commands::grade::run(args)
            }
        }
        Command::Validate(args) => commands::validate::run(args),
        Command::Calibrate(args) => commands::calibrate::run(args),
        Command::Cluster(args) => {
            #[cfg(any(feature = "embeddings-openai", feature = "cluster-label-openai"))]
            {
                commands::cluster::run(args).await
            }
            #[cfg(not(any(feature = "embeddings-openai", feature = "cluster-label-openai")))]
            {
                commands::cluster::run(args)
            }
        }
        Command::Report(args) => commands::report::run(args),
    }
}

#[cfg(not(any(
    feature = "llm-judge-openai",
    feature = "embeddings-openai",
    feature = "cluster-label-openai"
)))]
pub fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Extract(args) => commands::extract::run(args),
        Command::Detect(args) => commands::detect::run(args),
        Command::VerifyFindings(args) => commands::verify_findings::run(args),
        Command::VerifyRemediation(args) => commands::verify_remediation::run(args),
        Command::CompareRecurrence(args) => commands::compare_recurrence::run(args),
        Command::Grade(args) => commands::grade::run(args),
        Command::Validate(args) => commands::validate::run(args),
        Command::Calibrate(args) => commands::calibrate::run(args),
        Command::Cluster(args) => commands::cluster::run(args),
        Command::Report(args) => commands::report::run(args),
    }
}

#[cfg(test)]
#[path = "cli/tests.rs"]
mod tests;
