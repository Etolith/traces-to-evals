use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::commands;

#[derive(Debug, Parser)]
#[command(name = "traceeval")]
#[command(about = "Turn traces into eval cases and score them")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Extract eval cases from trace JSONL.
    Extract(ExtractArgs),
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
pub struct ClusterArgs {
    /// Input eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Cluster definitions JSONL file.
    #[arg(long)]
    pub clusters: PathBuf,
    /// Output cluster assignments JSONL file.
    #[arg(long)]
    pub out: PathBuf,
    /// Optional evaluation results JSONL file to annotate with cluster IDs.
    #[arg(long)]
    pub results: Option<PathBuf>,
    /// Optional output path for annotated evaluation results.
    #[arg(long = "results-out", requires = "results")]
    pub results_out: Option<PathBuf>,
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
pub enum ValidationProfileName {
    DraftCases,
    RunnableCases,
    EvaluationResults,
    CalibrationDataset,
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

#[cfg(feature = "llm-judge-openai")]
pub async fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Extract(args) => commands::extract::run(args),
        Command::Grade(args) => commands::grade::run(args).await,
        Command::Validate(args) => commands::validate::run(args),
        Command::Calibrate(args) => commands::calibrate::run(args),
        Command::Cluster(args) => commands::cluster::run(args),
        Command::Report(args) => commands::report::run(args),
    }
}

#[cfg(not(feature = "llm-judge-openai"))]
pub fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Extract(args) => commands::extract::run(args),
        Command::Grade(args) => commands::grade::run(args),
        Command::Validate(args) => commands::validate::run(args),
        Command::Calibrate(args) => commands::calibrate::run(args),
        Command::Cluster(args) => commands::cluster::run(args),
        Command::Report(args) => commands::report::run(args),
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn parses_judge_grader_args() {
        let cli = Cli::parse_from([
            "traceeval",
            "grade",
            "--cases",
            "cases.jsonl",
            "--judge",
            "openai-dive",
            "--model",
            "gpt-4o",
            "--out",
            "out.jsonl",
        ]);

        let Command::Grade(args) = cli.command else {
            panic!("expected grade command");
        };

        assert_eq!(args.cases, PathBuf::from("cases.jsonl"));
        assert_eq!(args.judge, Some(JudgeProviderName::OpenaiDive));
        assert_eq!(args.model.as_deref(), Some("gpt-4o"));
        assert_eq!(args.out, PathBuf::from("out.jsonl"));
    }

    #[test]
    fn rejects_missing_value_for_cases() {
        let result = Cli::try_parse_from([
            "traceeval",
            "grade",
            "--cases",
            "--judge",
            "openai-dive",
            "--model",
            "gpt-4o",
            "--out",
            "out.jsonl",
        ]);

        assert!(result.is_err());
    }
}
