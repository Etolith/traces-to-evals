use anyhow::{Result, anyhow};

#[cfg(feature = "llm-judge-openai")]
#[tokio::main]
async fn main() -> Result<()> {
    run(std::env::args().skip(1).collect()).await
}

#[cfg(not(feature = "llm-judge-openai"))]
fn main() -> Result<()> {
    run_without_openai(std::env::args().skip(1).collect())
}

#[cfg(feature = "llm-judge-openai")]
async fn run(args: Vec<String>) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("judge") => judge_with_openai_dive(parse_judge_args(&args[1..])?).await,
        Some("help") | Some("--help") | Some("-h") | None => {
            print_help();
            Ok(())
        }
        Some(command) => Err(anyhow!("unknown command {command:?}")),
    }
}

#[cfg(not(feature = "llm-judge-openai"))]
fn run_without_openai(args: Vec<String>) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("judge") => Err(anyhow!(
            "the judge command requires rebuilding with --features llm-judge-openai"
        )),
        Some("help") | Some("--help") | Some("-h") | None => {
            print_help();
            Ok(())
        }
        Some(command) => Err(anyhow!("unknown command {command:?}")),
    }
}

#[cfg(any(feature = "llm-judge-openai", test))]
#[derive(Debug, Clone)]
struct JudgeArgs {
    cases: String,
    provider: String,
    model: String,
    out: String,
}

#[cfg(any(feature = "llm-judge-openai", test))]
fn parse_judge_args(args: &[String]) -> Result<JudgeArgs> {
    let mut cases = None;
    let mut provider = None;
    let mut model = None;
    let mut out = None;
    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--cases" => cases = iter.next().cloned(),
            "--provider" => provider = iter.next().cloned(),
            "--model" => model = iter.next().cloned(),
            "--out" => out = iter.next().cloned(),
            other => return Err(anyhow!("unknown judge argument {other:?}")),
        }
    }

    Ok(JudgeArgs {
        cases: cases.ok_or_else(|| anyhow!("missing --cases"))?,
        provider: provider.ok_or_else(|| anyhow!("missing --provider"))?,
        model: model.ok_or_else(|| anyhow!("missing --model"))?,
        out: out.ok_or_else(|| anyhow!("missing --out"))?,
    })
}

#[cfg(feature = "llm-judge-openai")]
async fn judge_with_openai_dive(args: JudgeArgs) -> Result<()> {
    use traces_to_evals::exporters::{read_eval_cases_jsonl, write_judge_results_jsonl};
    use traces_to_evals::judge::openai_dive_judge::OpenAiDiveJudge;

    if args.provider != "openai-dive" {
        return Err(anyhow!(
            "unsupported provider {:?}; expected \"openai-dive\"",
            args.provider
        ));
    }

    let cases = read_eval_cases_jsonl(&args.cases)?;
    let judge = OpenAiDiveJudge::new_from_env(args.model);
    let mut results = Vec::with_capacity(cases.len());

    for case in cases {
        results.push(judge.judge_case(&case).await?);
    }

    write_judge_results_jsonl(&args.out, &results)?;
    Ok(())
}

fn print_help() {
    eprintln!(
        "usage:\n  traceeval judge --cases eval_cases.jsonl --provider openai-dive --model MODEL --out judge_results.jsonl"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_judge_args() {
        let parsed = parse_judge_args(&[
            "--cases".to_string(),
            "cases.jsonl".to_string(),
            "--provider".to_string(),
            "openai-dive".to_string(),
            "--model".to_string(),
            "gpt-4o".to_string(),
            "--out".to_string(),
            "out.jsonl".to_string(),
        ])
        .unwrap();

        assert_eq!(parsed.cases, "cases.jsonl");
        assert_eq!(parsed.provider, "openai-dive");
        assert_eq!(parsed.model, "gpt-4o");
        assert_eq!(parsed.out, "out.jsonl");
    }
}
