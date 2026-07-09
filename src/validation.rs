use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::evaluation::{EvaluationResult, ScoreScale};
use crate::model::EvalCase;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationReport {
    pub checked_cases: usize,
    pub checked_results: usize,
    pub errors: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub case_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
}

#[derive(Debug, Default)]
pub struct ValidationReportBuilder {
    checked_cases: usize,
    checked_results: usize,
    errors: Vec<ValidationIssue>,
}

impl ValidationReportBuilder {
    pub fn check_cases(mut self, cases: &[EvalCase]) -> Self {
        self.checked_cases += cases.len();
        let mut seen_case_ids = BTreeSet::<&str>::new();

        for case in cases {
            if case.id.trim().is_empty() {
                self.push("missing_case_id", "case id is empty", None, None);
            } else if !seen_case_ids.insert(case.id.as_str()) {
                self.push(
                    "duplicate_case_id",
                    "case id appears more than once",
                    Some(case.id.clone()),
                    Some(case.trace_id.clone()),
                );
            }

            if case.trace_id.trim().is_empty() {
                self.push(
                    "missing_trace_id",
                    "trace id is empty",
                    Some(case.id.clone()),
                    None,
                );
            }

            if case.input.trim().is_empty() {
                self.push(
                    "missing_input",
                    "case input is empty",
                    Some(case.id.clone()),
                    Some(case.trace_id.clone()),
                );
            }

            if case
                .actual_output
                .as_deref()
                .is_none_or(|output| output.trim().is_empty())
            {
                self.push(
                    "missing_actual_output",
                    "case actual_output is missing or empty",
                    Some(case.id.clone()),
                    Some(case.trace_id.clone()),
                );
            }
        }

        self
    }

    pub fn check_results(mut self, results: &[EvaluationResult]) -> Self {
        self.checked_results += results.len();
        let mut seen_result_keys = BTreeSet::<(&str, &str)>::new();

        for result in results {
            if result.case_id.trim().is_empty() {
                self.push("missing_case_id", "result case_id is empty", None, None);
            }

            if result.trace_id.trim().is_empty() {
                self.push(
                    "missing_trace_id",
                    "result trace_id is empty",
                    Some(result.case_id.clone()),
                    None,
                );
            }

            if result.evaluator_name.trim().is_empty() {
                self.push(
                    "missing_evaluator_name",
                    "result evaluator_name is empty",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            } else if !seen_result_keys
                .insert((result.case_id.as_str(), result.evaluator_name.as_str()))
            {
                self.push(
                    "duplicate_result",
                    "case has more than one result for the same evaluator",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            }

            if !result.raw_score.is_finite() {
                self.push(
                    "invalid_raw_score",
                    "raw_score must be finite",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            } else if !raw_score_in_scale(result.raw_score, result.score_scale) {
                self.push(
                    "invalid_raw_score",
                    "raw_score is outside its score_scale",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            }

            if !unit_interval(result.normalized_score) {
                self.push(
                    "invalid_normalized_score",
                    "normalized_score must be between 0.0 and 1.0",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            }

            if result
                .calibrated_score
                .is_some_and(|score| !unit_interval(score))
            {
                self.push(
                    "invalid_calibrated_score",
                    "calibrated_score must be between 0.0 and 1.0",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            }
        }

        self
    }

    pub fn check_overlap(mut self, cases: &[EvalCase], results: &[EvaluationResult]) -> Self {
        let case_ids = cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<BTreeSet<_>>();

        for result in results {
            if !case_ids.contains(result.case_id.as_str()) {
                self.push(
                    "unknown_result_case",
                    "result case_id is not present in cases",
                    Some(result.case_id.clone()),
                    Some(result.trace_id.clone()),
                );
            }
        }

        self
    }

    pub fn finish(self) -> ValidationReport {
        ValidationReport {
            checked_cases: self.checked_cases,
            checked_results: self.checked_results,
            errors: self.errors,
        }
    }

    fn push(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
        case_id: Option<String>,
        trace_id: Option<String>,
    ) {
        self.errors.push(ValidationIssue {
            code: code.into(),
            message: message.into(),
            case_id,
            trace_id,
        });
    }
}

pub fn validate_cases(cases: &[EvalCase]) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_cases(cases)
        .finish()
}

pub fn validate_results(results: &[EvaluationResult]) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_results(results)
        .finish()
}

pub fn validate_cases_and_results(
    cases: &[EvalCase],
    results: &[EvaluationResult],
) -> ValidationReport {
    ValidationReportBuilder::default()
        .check_cases(cases)
        .check_results(results)
        .check_overlap(cases, results)
        .finish()
}

fn raw_score_in_scale(raw_score: f32, score_scale: ScoreScale) -> bool {
    match score_scale {
        ScoreScale::Binary | ScoreScale::Unit => unit_interval(raw_score),
        ScoreScale::FourPoint => (1.0..=4.0).contains(&raw_score),
    }
}

fn unit_interval(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catches_missing_case_output_and_duplicate_ids() {
        let cases = vec![
            EvalCase::new("case-1", "trace-1", "input"),
            EvalCase::new("case-1", "trace-2", "input").with_actual_output("ok"),
        ];

        let report = validate_cases(&cases);

        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|issue| issue.code == "missing_actual_output")
        );
        assert!(
            report
                .errors
                .iter()
                .any(|issue| issue.code == "duplicate_case_id")
        );
    }

    #[test]
    fn validates_result_score_ranges() {
        let result = EvaluationResult::from_ids(
            "case-1",
            "trace-1",
            "judge",
            5.0,
            ScoreScale::FourPoint,
            true,
            "bad score",
        );

        let report = validate_results(&[result]);

        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|issue| issue.code == "invalid_raw_score")
        );
    }
}
