use std::path::PathBuf;

use anyhow::Result;
use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};

use crate::commands;

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
pub struct ClusterArgs {
    #[command(subcommand)]
    pub command: ClusterCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ClusterCommand {
    /// Generate case embeddings for cluster discovery.
    Embed(ClusterEmbedArgs),
    /// Discover clusters from historical cases and embeddings.
    Discover(ClusterDiscoverArgs),
    /// Label a discovered cluster model.
    Label(ClusterLabelArgs),
    /// Assign cases to known clusters or a discovered cluster model.
    Assign(ClusterAssignArgs),
}

#[derive(Debug, Clone, Args)]
#[command(group(
    ArgGroup::new("cluster_source")
        .required(true)
        .args(["clusters", "model"])
))]
pub struct ClusterAssignArgs {
    /// Input eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Cluster definitions JSONL file.
    #[arg(long, conflicts_with = "model")]
    pub clusters: Option<PathBuf>,
    /// Discovered cluster model JSON file.
    #[arg(long, conflicts_with = "clusters", requires = "embeddings")]
    pub model: Option<PathBuf>,
    /// Case embeddings JSONL file for discovered-model assignment.
    #[arg(long, requires = "model")]
    pub embeddings: Option<PathBuf>,
    /// Distance threshold above which model assignment marks novelty.
    #[arg(long = "novelty-distance-threshold", requires = "model")]
    pub novelty_distance_threshold: Option<f32>,
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
pub struct ClusterEmbedArgs {
    /// Input eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Embedding provider to run.
    #[arg(long, value_enum)]
    pub provider: ClusterEmbeddingProviderName,
    /// Embedding model name.
    #[arg(long)]
    pub model: String,
    /// Optional embedding dimensions override for providers that support it.
    #[arg(long)]
    pub dimensions: Option<u32>,
    /// Project namespace for generated artifact schema versions.
    #[arg(long = "project-name")]
    pub project_name: Option<String>,
    /// Output case embeddings JSONL file.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct ClusterDiscoverArgs {
    /// Input historical eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Input case embeddings JSONL file.
    #[arg(long)]
    pub embeddings: PathBuf,
    /// Discovery algorithm.
    #[arg(long, value_enum)]
    pub algorithm: ClusterAlgorithmName,
    /// Number of clusters for k-means.
    #[arg(long)]
    pub k: Option<usize>,
    /// Representative examples per discovered cluster.
    #[arg(long, default_value_t = 5)]
    pub representatives: usize,
    /// Project namespace for generated artifact schema versions.
    #[arg(long = "project-name")]
    pub project_name: Option<String>,
    /// Output discovered cluster model JSON file.
    #[arg(long = "out-model")]
    pub out_model: PathBuf,
    /// Output cluster assignments JSONL file.
    #[arg(long = "out-assignments")]
    pub out_assignments: PathBuf,
    /// Output report-compatible cluster definitions JSONL file.
    #[arg(long = "out-clusters")]
    pub out_clusters: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct ClusterLabelArgs {
    /// Input discovered cluster model JSON file.
    #[arg(long)]
    pub model: PathBuf,
    /// Input historical eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Label provider to run.
    #[arg(long, value_enum)]
    pub provider: ClusterLabelProviderName,
    /// LLM model name.
    #[arg(long = "llm-model")]
    pub llm_model: String,
    /// Output labeled cluster model JSON file.
    #[arg(long = "out-model")]
    pub out_model: PathBuf,
    /// Output report-compatible labeled cluster definitions JSONL file.
    #[arg(long = "out-clusters")]
    pub out_clusters: PathBuf,
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
pub enum ClusterEmbeddingProviderName {
    Openai,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ClusterLabelProviderName {
    Openai,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ClusterAlgorithmName {
    Kmeans,
    Dbscan,
}

#[cfg(any(
    feature = "llm-judge-openai",
    feature = "embeddings-openai",
    feature = "cluster-label-openai"
))]
pub async fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Extract(args) => commands::extract::run(args),
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

    #[test]
    fn parses_cluster_assign_with_model_source() {
        let cli = Cli::parse_from([
            "traceeval",
            "cluster",
            "assign",
            "--cases",
            "cases.jsonl",
            "--model",
            "cluster_model.json",
            "--embeddings",
            "embeddings.jsonl",
            "--out",
            "assignments.jsonl",
        ]);

        let Command::Cluster(cluster_args) = cli.command else {
            panic!("expected cluster command");
        };
        let ClusterCommand::Assign(args) = cluster_args.command else {
            panic!("expected cluster assign command");
        };

        assert_eq!(args.cases, PathBuf::from("cases.jsonl"));
        assert_eq!(args.model, Some(PathBuf::from("cluster_model.json")));
        assert_eq!(args.embeddings, Some(PathBuf::from("embeddings.jsonl")));
        assert_eq!(args.out, PathBuf::from("assignments.jsonl"));
    }

    #[test]
    fn rejects_cluster_assign_without_source() {
        let result = Cli::try_parse_from([
            "traceeval",
            "cluster",
            "assign",
            "--cases",
            "cases.jsonl",
            "--out",
            "assignments.jsonl",
        ]);

        assert!(result.is_err());
    }

    #[test]
    fn parses_cluster_embed_project_and_dimensions_args() {
        let cli = Cli::parse_from([
            "traceeval",
            "cluster",
            "embed",
            "--cases",
            "cases.jsonl",
            "--provider",
            "openai",
            "--model",
            "text-embedding-3-small",
            "--dimensions",
            "512",
            "--project-name",
            "acme-evals",
            "--out",
            "embeddings.jsonl",
        ]);

        let Command::Cluster(cluster_args) = cli.command else {
            panic!("expected cluster command");
        };
        let ClusterCommand::Embed(args) = cluster_args.command else {
            panic!("expected cluster embed command");
        };

        assert_eq!(args.cases, PathBuf::from("cases.jsonl"));
        assert_eq!(args.provider, ClusterEmbeddingProviderName::Openai);
        assert_eq!(args.model, "text-embedding-3-small");
        assert_eq!(args.dimensions, Some(512));
        assert_eq!(args.project_name.as_deref(), Some("acme-evals"));
        assert_eq!(args.out, PathBuf::from("embeddings.jsonl"));
    }
}
