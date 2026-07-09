use anyhow::{Result, anyhow};

use crate::cli::{ValidateArgs, ValidationProfileName};
use crate::evaluation::EvaluationResult;
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlFile;
use crate::model::EvalCase;
use crate::validation::{
    ValidationProfile, ValidationReport, ValidationReportBuilder,
    validate_cases_and_results_with_profile, validate_cases_with_profile, validate_results,
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

    let profile = validation_profile(args.profile, cases.is_some(), results.is_some())?;

    let report = match (cases.as_deref(), results.as_deref(), profile) {
        (Some(cases), Some(results), ValidationProfile::EvaluationResults) => {
            ValidationReportBuilder::default()
                .check_results(results)
                .check_overlap(cases, results)
                .finish()
        }
        (Some(cases), Some(results), _) => {
            validate_cases_and_results_with_profile(cases, results, case_profile(profile))
        }
        (Some(cases), None, _) => validate_cases_with_profile(cases, case_profile(profile)),
        (None, Some(results), _) => validate_results(results),
        (None, None, _) => unreachable!("validated above"),
    };

    if let Some(path) = args.out {
        JsonFile::new(path).write_pretty(&report)?;
    }

    ensure_valid(&report)
}

fn validation_profile(
    profile: Option<ValidationProfileName>,
    has_cases: bool,
    has_results: bool,
) -> Result<ValidationProfile> {
    let Some(profile) = profile else {
        return Ok(match (has_cases, has_results) {
            (true, _) => ValidationProfile::RunnableCases,
            (false, true) => ValidationProfile::EvaluationResults,
            (false, false) => unreachable!("validated before profile resolution"),
        });
    };

    let profile = match profile {
        ValidationProfileName::DraftCases => ValidationProfile::DraftCases,
        ValidationProfileName::RunnableCases => ValidationProfile::RunnableCases,
        ValidationProfileName::EvaluationResults => ValidationProfile::EvaluationResults,
        ValidationProfileName::CalibrationDataset => ValidationProfile::CalibrationDataset,
    };

    match (profile, has_cases, has_results) {
        (ValidationProfile::DraftCases | ValidationProfile::RunnableCases, true, _) => Ok(profile),
        (ValidationProfile::EvaluationResults, _, true) => Ok(profile),
        (ValidationProfile::CalibrationDataset, _, true) => Ok(profile),
        (ValidationProfile::EvaluationResults, true, false) => Err(anyhow!(
            "--profile evaluation-results requires --results input"
        )),
        (ValidationProfile::DraftCases | ValidationProfile::RunnableCases, false, true) => {
            Err(anyhow!("case validation profiles require --cases input"))
        }
        (ValidationProfile::CalibrationDataset, true, false) => Err(anyhow!(
            "--profile calibration-dataset requires --results input"
        )),
        (_, false, false) => unreachable!("validated before profile resolution"),
    }
}

fn case_profile(profile: ValidationProfile) -> ValidationProfile {
    match profile {
        ValidationProfile::DraftCases => ValidationProfile::DraftCases,
        ValidationProfile::RunnableCases
        | ValidationProfile::EvaluationResults
        | ValidationProfile::CalibrationDataset => ValidationProfile::RunnableCases,
    }
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
