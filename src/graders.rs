use serde::{Deserialize, Serialize};

use crate::evaluation::{EvaluationResult, Evaluator, ScoreScale};
use crate::model::EvalCase;
use crate::{Result, TraceEvalError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradeResult {
    pub case_id: String,
    pub trace_id: String,
    pub grader_name: String,
    pub score: u8,
    pub passed: bool,
    pub evaluation: String,
}

impl GradeResult {
    pub fn pass(
        case: &EvalCase,
        grader_name: impl Into<String>,
        evaluation: impl Into<String>,
    ) -> Self {
        Self {
            case_id: case.id.clone(),
            trace_id: case.trace_id.clone(),
            grader_name: grader_name.into(),
            score: 1,
            passed: true,
            evaluation: evaluation.into(),
        }
    }

    pub fn fail(
        case: &EvalCase,
        grader_name: impl Into<String>,
        evaluation: impl Into<String>,
    ) -> Self {
        Self {
            case_id: case.id.clone(),
            trace_id: case.trace_id.clone(),
            grader_name: grader_name.into(),
            score: 0,
            passed: false,
            evaluation: evaluation.into(),
        }
    }
}

impl From<GradeResult> for EvaluationResult {
    fn from(result: GradeResult) -> Self {
        EvaluationResult::from_ids(
            result.case_id,
            result.trace_id,
            result.grader_name,
            f32::from(result.score),
            ScoreScale::Binary,
            result.passed,
            result.evaluation,
        )
        .with_confidence(1.0)
    }
}

pub trait DeterministicGrader {
    fn name(&self) -> &'static str;
    fn grade(&self, case: &EvalCase) -> Result<GradeResult>;
}

impl<T> Evaluator for T
where
    T: DeterministicGrader,
{
    fn evaluator_name(&self) -> String {
        self.name().to_string()
    }

    fn evaluate_case(&self, case: &EvalCase) -> Result<EvaluationResult> {
        Ok(self.grade(case)?.into())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NonEmptyOutputGrader;

impl DeterministicGrader for NonEmptyOutputGrader {
    fn name(&self) -> &'static str {
        "non_empty_output"
    }

    fn grade(&self, case: &EvalCase) -> Result<GradeResult> {
        let output =
            case.actual_output
                .as_deref()
                .ok_or_else(|| TraceEvalError::MissingActualOutput {
                    case_id: case.id.clone(),
                })?;

        if output.trim().is_empty() {
            Ok(GradeResult::fail(
                case,
                self.name(),
                "actual output is empty",
            ))
        } else {
            Ok(GradeResult::pass(
                case,
                self.name(),
                "actual output is present",
            ))
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ExactMatchGrader;

impl DeterministicGrader for ExactMatchGrader {
    fn name(&self) -> &'static str {
        "exact_match"
    }

    fn grade(&self, case: &EvalCase) -> Result<GradeResult> {
        let actual =
            case.actual_output
                .as_deref()
                .ok_or_else(|| TraceEvalError::MissingActualOutput {
                    case_id: case.id.clone(),
                })?;
        let expected = case.expected_output.as_deref().ok_or_else(|| {
            TraceEvalError::MissingExpectedOutput {
                case_id: case.id.clone(),
            }
        })?;

        if actual.trim() == expected.trim() {
            Ok(GradeResult::pass(
                case,
                self.name(),
                "actual output exactly matches expected output",
            ))
        } else {
            Ok(GradeResult::fail(
                case,
                self.name(),
                "actual output does not match expected output",
            ))
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContainsGrader {
    needle: String,
}

impl ContainsGrader {
    pub fn new(needle: impl Into<String>) -> Self {
        Self {
            needle: needle.into(),
        }
    }
}

impl DeterministicGrader for ContainsGrader {
    fn name(&self) -> &'static str {
        "contains"
    }

    fn grade(&self, case: &EvalCase) -> Result<GradeResult> {
        let actual =
            case.actual_output
                .as_deref()
                .ok_or_else(|| TraceEvalError::MissingActualOutput {
                    case_id: case.id.clone(),
                })?;

        if actual.contains(&self.needle) {
            Ok(GradeResult::pass(
                case,
                self.name(),
                format!("actual output contains {:?}", self.needle),
            ))
        } else {
            Ok(GradeResult::fail(
                case,
                self.name(),
                format!("actual output does not contain {:?}", self.needle),
            ))
        }
    }
}

pub fn grade_cases<G: DeterministicGrader>(
    grader: &G,
    cases: &[EvalCase],
) -> Result<Vec<GradeResult>> {
    cases.iter().map(|case| grader.grade(case)).collect()
}

pub fn evaluate_cases<G: Evaluator>(
    evaluator: &G,
    cases: &[EvalCase],
) -> Result<Vec<EvaluationResult>> {
    evaluator.evaluate_cases(cases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_trims_before_comparison() {
        let case = EvalCase::new("case-1", "trace-1", "input")
            .with_actual_output("answer\n")
            .with_expected_output("answer");

        let result = ExactMatchGrader.grade(&case).unwrap();

        assert!(result.passed);
        assert_eq!(result.score, 1);

        let evaluated = ExactMatchGrader.evaluate_case(&case).unwrap();
        assert_eq!(evaluated.evaluator_name, "exact_match");
        assert_eq!(evaluated.normalized_score, 1.0);
    }

    #[test]
    fn non_empty_output_fails_blank_output() {
        let case = EvalCase::new("case-1", "trace-1", "input").with_actual_output("  ");

        let result = NonEmptyOutputGrader.grade(&case).unwrap();

        assert!(!result.passed);
    }
}
