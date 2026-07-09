use anyhow::{Result, anyhow};

use crate::cli::{ValidateArgs, ValidationProfileName};
use crate::clustering::{CaseEmbedding, ClusterAssignment, ClusterModel};
use crate::evaluation::EvaluationResult;
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlFile;
use crate::model::EvalCase;
use crate::validation::{
    ValidationProfile, ValidationReport, ValidationReportBuilder,
    validate_cases_and_results_with_profile, validate_cases_with_profile, validate_results,
};

pub fn run(args: ValidateArgs) -> Result<()> {
    if !has_any_input(&args) {
        return Err(anyhow!(
            "validate requires at least one input: --cases, --results, --embeddings, --cluster-model, or --assignments"
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
    let embeddings = args
        .embeddings
        .as_ref()
        .map(JsonlFile::new)
        .map(|file| file.read_all::<CaseEmbedding>())
        .transpose()?;
    let cluster_model = args
        .cluster_model
        .as_ref()
        .map(JsonFile::new)
        .map(|file| file.read::<ClusterModel>())
        .transpose()?;
    let assignments = args
        .assignments
        .as_ref()
        .map(JsonlFile::new)
        .map(|file| file.read_all::<ClusterAssignment>())
        .transpose()?;

    let profile = validation_profile(
        args.profile,
        cases.is_some(),
        results.is_some(),
        embeddings.is_some(),
        cluster_model.is_some(),
        assignments.is_some(),
    )?;

    let report = build_report(
        profile,
        cases.as_deref(),
        results.as_deref(),
        embeddings.as_deref(),
        cluster_model.as_ref(),
        assignments.as_deref(),
    )?;

    if let Some(path) = args.out {
        JsonFile::new(path).write_pretty(&report)?;
    }

    ensure_valid(&report)
}

fn has_any_input(args: &ValidateArgs) -> bool {
    args.cases.is_some()
        || args.results.is_some()
        || args.embeddings.is_some()
        || args.cluster_model.is_some()
        || args.assignments.is_some()
}

fn validation_profile(
    profile: Option<ValidationProfileName>,
    has_cases: bool,
    has_results: bool,
    has_embeddings: bool,
    has_cluster_model: bool,
    has_assignments: bool,
) -> Result<ValidationProfile> {
    let Some(profile) = profile else {
        if has_cases {
            return Ok(ValidationProfile::RunnableCases);
        }
        if has_results {
            return Ok(ValidationProfile::EvaluationResults);
        }
        if has_embeddings {
            return Ok(ValidationProfile::EmbeddingDataset);
        }
        if has_cluster_model {
            return Ok(ValidationProfile::ClusterModel);
        }
        if has_assignments {
            return Ok(ValidationProfile::ClusterAssignments);
        }
        unreachable!("validated before profile resolution");
    };

    let profile = match profile {
        ValidationProfileName::DraftCases => ValidationProfile::DraftCases,
        ValidationProfileName::RunnableCases => ValidationProfile::RunnableCases,
        ValidationProfileName::EvaluationResults => ValidationProfile::EvaluationResults,
        ValidationProfileName::CalibrationDataset => ValidationProfile::CalibrationDataset,
        ValidationProfileName::EmbeddingDataset => ValidationProfile::EmbeddingDataset,
        ValidationProfileName::ClusterModel => ValidationProfile::ClusterModel,
        ValidationProfileName::ClusterAssignments => ValidationProfile::ClusterAssignments,
    };

    match profile {
        ValidationProfile::DraftCases | ValidationProfile::RunnableCases if has_cases => {
            Ok(profile)
        }
        ValidationProfile::EvaluationResults if has_results => Ok(profile),
        ValidationProfile::CalibrationDataset if has_results => Ok(profile),
        ValidationProfile::EmbeddingDataset if has_embeddings => Ok(profile),
        ValidationProfile::ClusterModel if has_cluster_model => Ok(profile),
        ValidationProfile::ClusterAssignments if has_assignments => Ok(profile),
        ValidationProfile::DraftCases | ValidationProfile::RunnableCases => {
            Err(anyhow!("case validation profiles require --cases input"))
        }
        ValidationProfile::EvaluationResults => Err(anyhow!(
            "--profile evaluation-results requires --results input"
        )),
        ValidationProfile::CalibrationDataset => Err(anyhow!(
            "--profile calibration-dataset requires --results input"
        )),
        ValidationProfile::EmbeddingDataset => Err(anyhow!(
            "--profile embedding-dataset requires --embeddings input"
        )),
        ValidationProfile::ClusterModel => Err(anyhow!(
            "--profile cluster-model requires --cluster-model input"
        )),
        ValidationProfile::ClusterAssignments => Err(anyhow!(
            "--profile cluster-assignments requires --assignments input"
        )),
    }
}

fn build_report(
    profile: ValidationProfile,
    cases: Option<&[EvalCase]>,
    results: Option<&[EvaluationResult]>,
    embeddings: Option<&[CaseEmbedding]>,
    cluster_model: Option<&ClusterModel>,
    assignments: Option<&[ClusterAssignment]>,
) -> Result<ValidationReport> {
    Ok(match profile {
        ValidationProfile::DraftCases | ValidationProfile::RunnableCases => {
            let cases = cases.expect("profile resolution checked cases");
            match results {
                Some(results) => {
                    validate_cases_and_results_with_profile(cases, results, case_profile(profile))
                }
                None => validate_cases_with_profile(cases, case_profile(profile)),
            }
        }
        ValidationProfile::EvaluationResults => {
            let results = results.expect("profile resolution checked results");
            match cases {
                Some(cases) => ValidationReportBuilder::default()
                    .check_results(results)
                    .check_overlap(cases, results)
                    .finish(),
                None => validate_results(results),
            }
        }
        ValidationProfile::CalibrationDataset => {
            let results = results.expect("profile resolution checked results");
            match cases {
                Some(cases) => validate_cases_and_results_with_profile(
                    cases,
                    results,
                    ValidationProfile::CalibrationDataset,
                ),
                None => ValidationReportBuilder::default()
                    .check_results(results)
                    .finish(),
            }
        }
        ValidationProfile::EmbeddingDataset => {
            let embeddings = embeddings.expect("profile resolution checked embeddings");
            let builder = ValidationReportBuilder::default().check_embeddings(embeddings);
            match cases {
                Some(cases) => builder
                    .check_cases_with_profile(cases, ValidationProfile::EmbeddingDataset)
                    .check_embedding_overlap(cases, embeddings, false)
                    .finish(),
                None => builder.finish(),
            }
        }
        ValidationProfile::ClusterModel => ValidationReportBuilder::default()
            .check_cluster_model(cluster_model.expect("profile resolution checked cluster model"))
            .finish(),
        ValidationProfile::ClusterAssignments => {
            let assignments = assignments.expect("profile resolution checked assignments");
            let builder = ValidationReportBuilder::default().check_cluster_assignments(assignments);
            match cases {
                Some(cases) => builder
                    .check_cases_with_profile(cases, ValidationProfile::ClusterAssignments)
                    .check_assignment_overlap(cases, assignments)
                    .finish(),
                None => builder.finish(),
            }
        }
    })
}

fn case_profile(profile: ValidationProfile) -> ValidationProfile {
    match profile {
        ValidationProfile::DraftCases => ValidationProfile::DraftCases,
        ValidationProfile::RunnableCases
        | ValidationProfile::EvaluationResults
        | ValidationProfile::CalibrationDataset => ValidationProfile::RunnableCases,
        ValidationProfile::EmbeddingDataset
        | ValidationProfile::ClusterModel
        | ValidationProfile::ClusterAssignments => ValidationProfile::DraftCases,
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
