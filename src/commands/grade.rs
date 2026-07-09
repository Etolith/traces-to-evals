use anyhow::{Result, anyhow};

#[cfg(feature = "llm-judge-openai")]
use crate::cli::JudgeProviderName;
use crate::cli::{DeterministicGraderName, GradeArgs};
use crate::evaluation::EvaluationRun;
use crate::graders::{ContainsGrader, ExactMatchGrader, NonEmptyOutputGrader};
use crate::io::jsonl::JsonlFile;
use crate::model::EvalCase;

#[cfg(feature = "llm-judge-openai")]
pub async fn run(args: GradeArgs) -> Result<()> {
    if args.judge.is_some() {
        return run_judge(args).await;
    }

    run_deterministic(args)
}

#[cfg(not(feature = "llm-judge-openai"))]
pub fn run(args: GradeArgs) -> Result<()> {
    if args.judge.is_some() {
        return Err(anyhow!(
            "judge grading requires rebuilding with --features llm-judge-openai"
        ));
    }

    run_deterministic(args)
}

fn run_deterministic(args: GradeArgs) -> Result<()> {
    let cases: Vec<EvalCase> = JsonlFile::new(&args.cases).read_all()?;
    let grader = args
        .grader
        .unwrap_or(DeterministicGraderName::NonEmptyOutput);

    let run = match grader {
        DeterministicGraderName::NonEmptyOutput => {
            EvaluationRun::new(cases).evaluate_with(&NonEmptyOutputGrader)?
        }
        DeterministicGraderName::ExactMatch => {
            EvaluationRun::new(cases).evaluate_with(&ExactMatchGrader)?
        }
        DeterministicGraderName::Contains => {
            let needle = args
                .contains
                .ok_or_else(|| anyhow!("--contains is required for --grader contains"))?;
            EvaluationRun::new(cases).evaluate_with(&ContainsGrader::new(needle))?
        }
    };

    JsonlFile::new(&args.out).write_all(run.results())
}

#[cfg(feature = "llm-judge-openai")]
async fn run_judge(args: GradeArgs) -> Result<()> {
    use crate::judge::openai::OpenAiJudge;

    match args.judge {
        Some(JudgeProviderName::OpenaiDive) => {
            let model = args
                .model
                .ok_or_else(|| anyhow!("--model is required for --judge openai-dive"))?;
            let cases: Vec<EvalCase> = JsonlFile::new(&args.cases).read_all()?;
            let judge = OpenAiJudge::from_env(model);
            let run = EvaluationRun::new(cases)
                .evaluate_with_async(&judge)
                .await?;

            JsonlFile::new(&args.out).write_all(run.results())
        }
        None => Err(anyhow!("missing --judge or --grader")),
    }
}
