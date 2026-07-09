use anyhow::{Result, anyhow};

#[cfg(feature = "llm-judge-openai")]
use crate::cli::JudgeProviderName;
use crate::cli::{DeterministicGraderName, GradeArgs};
use crate::graders::{ContainsGrader, DeterministicGrader, ExactMatchGrader, NonEmptyOutputGrader};
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

    let results = match grader {
        DeterministicGraderName::NonEmptyOutput => grade_with(&NonEmptyOutputGrader, &cases)?,
        DeterministicGraderName::ExactMatch => grade_with(&ExactMatchGrader, &cases)?,
        DeterministicGraderName::Contains => {
            let needle = args
                .contains
                .ok_or_else(|| anyhow!("--contains is required for --grader contains"))?;
            grade_with(&ContainsGrader::new(needle), &cases)?
        }
    };

    JsonlFile::new(&args.out).write_all(&results)
}

fn grade_with<G: DeterministicGrader>(
    grader: &G,
    cases: &[crate::model::EvalCase],
) -> Result<Vec<crate::graders::GradeResult>> {
    cases.iter().map(|case| grader.grade(case)).collect()
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
            let mut results = Vec::with_capacity(cases.len());

            for case in cases {
                results.push(judge.judge_case(&case).await?);
            }

            JsonlFile::new(&args.out).write_all(&results)
        }
        None => Err(anyhow!("missing --judge or --grader")),
    }
}
