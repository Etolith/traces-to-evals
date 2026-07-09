use anyhow::{Result, anyhow};

use crate::cli::ValidateArgs;
use crate::evaluation::EvaluationResult;
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlFile;
use crate::model::EvalCase;
use crate::validation::{
    ValidationReport, validate_cases, validate_cases_and_results, validate_results,
};

pub fn run(args: ValidateArgs) -> Result<()> {
    if args.cases.is_none() && args.results.is_none() {
        return Err(anyhow!(
            "validate requires --cases, --results, or both inputs"
        ));
    }

    let cases = args
        .cases
        .as_ref()
        .map(JsonlFile::new)
        .map(|file| file.read_all::<EvalCase>())
        .transpose()?;
    let results = args
        .results
        .as_ref()
        .map(JsonlFile::new)
        .map(|file| file.read_all::<EvaluationResult>())
        .transpose()?;

    let report = match (cases.as_deref(), results.as_deref()) {
        (Some(cases), Some(results)) => validate_cases_and_results(cases, results),
        (Some(cases), None) => validate_cases(cases),
        (None, Some(results)) => validate_results(results),
        (None, None) => unreachable!("validated above"),
    };

    if let Some(path) = args.out {
        JsonFile::new(path).write_pretty(&report)?;
    }

    ensure_valid(&report)
}

fn ensure_valid(report: &ValidationReport) -> Result<()> {
    if report.is_valid() {
        Ok(())
    } else {
        Err(anyhow!(
            "validation failed with {} error(s)",
            report.error_count()
        ))
    }
}
