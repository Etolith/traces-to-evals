use anyhow::Result;
#[cfg(not(feature = "llm-judge-openai"))]
use anyhow::anyhow;

use crate::cli::JudgeArgs;
#[cfg(feature = "llm-judge-openai")]
use crate::cli::JudgeProviderName;

#[cfg(feature = "llm-judge-openai")]
pub async fn run(args: JudgeArgs) -> Result<()> {
    use crate::io::jsonl::JsonlFile;
    use crate::judge::openai::OpenAiJudge;
    use crate::model::EvalCase;

    match args.provider {
        JudgeProviderName::OpenaiDive => {
            let cases: Vec<EvalCase> = JsonlFile::new(&args.cases).read_all()?;
            let judge = OpenAiJudge::from_env(args.model);
            let mut results = Vec::with_capacity(cases.len());

            for case in cases {
                results.push(judge.judge_case(&case).await?);
            }

            JsonlFile::new(&args.out).write_all(&results)
        }
    }
}

#[cfg(not(feature = "llm-judge-openai"))]
pub fn run(_args: JudgeArgs) -> Result<()> {
    Err(anyhow!(
        "the judge command requires rebuilding with --features llm-judge-openai"
    ))
}
