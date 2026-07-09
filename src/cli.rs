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
    /// Grade eval cases with a deterministic grader or judge provider.
    Grade(GradeArgs),
    /// Compatibility alias for `grade --judge openai-dive`.
    Judge(JudgeArgs),
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
pub struct JudgeArgs {
    /// Input eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Judge provider to run.
    #[arg(long, value_enum)]
    pub provider: JudgeProviderName,
    /// Model name for judge providers.
    #[arg(long)]
    pub model: String,
    /// Output judge results JSONL file.
    #[arg(long)]
    pub out: PathBuf,
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
        Command::Grade(args) => commands::grade::run(args).await,
        Command::Judge(args) => commands::judge::run(args).await,
    }
}

#[cfg(not(feature = "llm-judge-openai"))]
pub fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Grade(args) => commands::grade::run(args),
        Command::Judge(args) => commands::judge::run(args),
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn parses_judge_alias_args() {
        let cli = Cli::parse_from([
            "traceeval",
            "judge",
            "--cases",
            "cases.jsonl",
            "--provider",
            "openai-dive",
            "--model",
            "gpt-4o",
            "--out",
            "out.jsonl",
        ]);

        let Command::Judge(args) = cli.command else {
            panic!("expected judge command");
        };

        assert_eq!(args.cases, PathBuf::from("cases.jsonl"));
        assert_eq!(args.provider, JudgeProviderName::OpenaiDive);
        assert_eq!(args.model, "gpt-4o");
        assert_eq!(args.out, PathBuf::from("out.jsonl"));
    }

    #[test]
    fn rejects_missing_value_for_cases() {
        let result = Cli::try_parse_from([
            "traceeval",
            "judge",
            "--cases",
            "--provider",
            "openai-dive",
            "--model",
            "gpt-4o",
            "--out",
            "out.jsonl",
        ]);

        assert!(result.is_err());
    }
}
