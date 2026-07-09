use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::json;

use crate::calibration::HumanRating;
use crate::graders::GradeResult;
use crate::judge::types::JudgeResult;
use crate::model::{EvalCase, Trace};

pub fn read_traces_jsonl(path: impl AsRef<Path>) -> Result<Vec<Trace>> {
    read_jsonl(path)
}

pub fn write_eval_cases_jsonl(path: impl AsRef<Path>, cases: &[EvalCase]) -> Result<()> {
    write_jsonl(path, cases)
}

pub fn read_eval_cases_jsonl(path: impl AsRef<Path>) -> Result<Vec<EvalCase>> {
    read_jsonl(path)
}

pub fn read_human_ratings_jsonl(path: impl AsRef<Path>) -> Result<Vec<HumanRating>> {
    read_jsonl(path)
}

pub fn write_human_ratings_jsonl(path: impl AsRef<Path>, ratings: &[HumanRating]) -> Result<()> {
    write_jsonl(path, ratings)
}

pub fn write_grade_results_jsonl(path: impl AsRef<Path>, results: &[GradeResult]) -> Result<()> {
    write_jsonl(path, results)
}

pub fn write_judge_results_jsonl(path: impl AsRef<Path>, results: &[JudgeResult]) -> Result<()> {
    write_jsonl(path, results)
}

pub fn write_eval_cases_csv(path: impl AsRef<Path>, cases: &[EvalCase]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path.as_ref())
        .with_context(|| format!("failed to create CSV {}", path.as_ref().display()))?;

    writer.write_record([
        "id",
        "trace_id",
        "input",
        "actual_output",
        "expected_output",
        "rubric",
    ])?;

    for case in cases {
        writer.write_record([
            case.id.as_str(),
            case.trace_id.as_str(),
            case.input.as_str(),
            case.actual_output.as_deref().unwrap_or_default(),
            case.expected_output.as_deref().unwrap_or_default(),
            case.rubric.as_deref().unwrap_or_default(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

pub fn write_openai_eval_jsonl(path: impl AsRef<Path>, cases: &[EvalCase]) -> Result<()> {
    let rows = cases.iter().map(|case| {
        json!({
            "input": case.input,
            "ideal": case.expected_output.as_deref().unwrap_or_default(),
            "metadata": {
                "case_id": case.id,
                "trace_id": case.trace_id,
            }
        })
    });

    write_jsonl_iter(path, rows)
}

pub fn write_promptfoo_json(path: impl AsRef<Path>, cases: &[EvalCase]) -> Result<()> {
    let tests = cases
        .iter()
        .map(|case| {
            json!({
                "vars": {
                    "input": case.input,
                },
                "assert": [
                    {
                        "type": "contains",
                        "value": case.expected_output.as_deref().unwrap_or_default(),
                    }
                ],
                "metadata": {
                    "case_id": case.id,
                    "trace_id": case.trace_id,
                    "rubric": case.rubric,
                }
            })
        })
        .collect::<Vec<_>>();

    let rendered = serde_json::to_string_pretty(&json!({ "tests": tests }))?;
    std::fs::write(path.as_ref(), rendered)
        .with_context(|| format!("failed to write {}", path.as_ref().display()))?;
    Ok(())
}

fn read_jsonl<T>(path: impl AsRef<Path>) -> Result<Vec<T>>
where
    T: serde::de::DeserializeOwned,
{
    let file = File::open(path.as_ref())
        .with_context(|| format!("failed to open {}", path.as_ref().display()))?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from {}",
                index + 1,
                path.as_ref().display()
            )
        })?;

        if line.trim().is_empty() {
            continue;
        }

        let value = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse JSON on line {} from {}",
                index + 1,
                path.as_ref().display()
            )
        })?;
        values.push(value);
    }

    Ok(values)
}

fn write_jsonl<T>(path: impl AsRef<Path>, values: &[T]) -> Result<()>
where
    T: Serialize,
{
    write_jsonl_iter(path, values.iter())
}

fn write_jsonl_iter<T, I>(path: impl AsRef<Path>, values: I) -> Result<()>
where
    T: Serialize,
    I: IntoIterator<Item = T>,
{
    let file = File::create(path.as_ref())
        .with_context(|| format!("failed to create {}", path.as_ref().display()))?;
    let mut writer = BufWriter::new(file);

    for value in values {
        serde_json::to_writer(&mut writer, &value)?;
        writer.write_all(b"\n")?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn writes_and_reads_eval_cases_jsonl() {
        let dir = std::env::temp_dir().join(format!("traces-to-evals-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cases.jsonl");
        let cases = vec![
            EvalCase::new("case-1", "trace-1", "input").with_actual_output("output"),
            EvalCase::new("case-2", "trace-2", "input 2"),
        ];

        write_eval_cases_jsonl(&path, &cases).unwrap();
        let round_tripped = read_eval_cases_jsonl(&path).unwrap();

        assert_eq!(round_tripped, cases);
    }

    #[test]
    fn writes_openai_eval_rows() {
        let dir = std::env::temp_dir().join(format!(
            "traces-to-evals-openai-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("openai.jsonl");
        let cases = vec![EvalCase::new("case-1", "trace-1", "input").with_expected_output("ideal")];

        write_openai_eval_jsonl(&path, &cases).unwrap();
        let rendered = fs::read_to_string(path).unwrap();

        assert!(rendered.contains(r#""input":"input""#));
        assert!(rendered.contains(r#""ideal":"ideal""#));
    }
}
