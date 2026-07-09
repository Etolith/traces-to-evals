use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::evaluation::{EvaluationResult, EvaluationRun, ScoreScale, evaluator_names};
use crate::{Result, TraceEvalError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanRating {
    pub case_id: String,
    pub trace_id: String,
    pub score: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationModel {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_name: Option<String>,
    pub human_pass_threshold: u8,
    pub bins: Vec<CalibrationBin>,
    pub global_pass_rate: f32,
    pub mean_absolute_error: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub normalized_score: f32,
    pub count: usize,
    pub human_mean_normalized_score: f32,
    pub human_pass_rate: f32,
}

impl CalibrationModel {
    pub fn fit(
        human_ratings: &[HumanRating],
        results: &[EvaluationResult],
        human_pass_threshold: u8,
    ) -> Result<Self> {
        validate_threshold(human_pass_threshold)?;

        let ratings_by_case = human_ratings
            .iter()
            .map(|rating| {
                validate_score(rating.score, "human rating")?;
                Ok((rating.case_id.as_str(), rating))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let names = evaluator_names(results);
        let evaluator_name = if names.len() == 1 {
            names.into_iter().next()
        } else {
            None
        };

        let mut bins = BTreeMap::<i32, CalibrationBinAccumulator>::new();
        let mut compared = 0usize;
        let mut human_passes = 0usize;
        let mut absolute_error = 0.0f32;

        for result in results {
            let Some(rating) = ratings_by_case.get(result.case_id.as_str()) else {
                continue;
            };

            let human_normalized_score = ScoreScale::FourPoint.normalize(f32::from(rating.score));
            let human_passed = rating
                .passed
                .unwrap_or(rating.score >= human_pass_threshold);
            if human_passed {
                human_passes += 1;
            }

            compared += 1;
            absolute_error += (human_normalized_score - result.normalized_score).abs();

            bins.entry(score_bin_key(result.normalized_score))
                .or_insert_with(|| CalibrationBinAccumulator::new(result.normalized_score))
                .push(human_normalized_score, human_passed);
        }

        if compared == 0 {
            return Err(TraceEvalError::CalibrationOverlap);
        }

        Ok(Self {
            evaluator_name,
            human_pass_threshold,
            bins: bins
                .into_values()
                .map(CalibrationBinAccumulator::into_bin)
                .collect(),
            global_pass_rate: rate(human_passes, compared),
            mean_absolute_error: absolute_error / compared as f32,
        })
    }

    pub fn calibrated_score(&self, result: &EvaluationResult) -> Option<f32> {
        if self
            .evaluator_name
            .as_ref()
            .is_some_and(|name| name != &result.evaluator_name)
        {
            return None;
        }

        self.bins
            .iter()
            .min_by(|left, right| {
                let left_delta = (left.normalized_score - result.normalized_score).abs();
                let right_delta = (right.normalized_score - result.normalized_score).abs();
                left_delta.total_cmp(&right_delta)
            })
            .map(|bin| bin.human_pass_rate)
    }

    pub fn apply(&self, result: EvaluationResult) -> EvaluationResult {
        match self.calibrated_score(&result) {
            Some(score) => result.with_calibrated_score(score),
            None => result,
        }
    }

    pub fn apply_run(&self, mut run: EvaluationRun) -> EvaluationRun {
        run.results = run
            .results
            .into_iter()
            .map(|result| self.apply(result))
            .collect();
        run
    }
}

fn validate_threshold(pass_threshold: u8) -> Result<()> {
    if (1..=4).contains(&pass_threshold) {
        Ok(())
    } else {
        Err(TraceEvalError::InvalidThreshold {
            threshold: pass_threshold,
            scale: "four_point".to_string(),
        })
    }
}

fn validate_score(score: u8, label: &str) -> Result<()> {
    if (1..=4).contains(&score) {
        Ok(())
    } else {
        Err(TraceEvalError::InvalidScore {
            score: f32::from(score),
            scale: label.to_string(),
        })
    }
}

fn rate(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[derive(Debug, Clone)]
struct CalibrationBinAccumulator {
    normalized_score: f32,
    count: usize,
    human_score_sum: f32,
    human_passes: usize,
}

impl CalibrationBinAccumulator {
    fn new(normalized_score: f32) -> Self {
        Self {
            normalized_score,
            count: 0,
            human_score_sum: 0.0,
            human_passes: 0,
        }
    }

    fn push(&mut self, human_normalized_score: f32, human_passed: bool) {
        self.count += 1;
        self.human_score_sum += human_normalized_score;

        if human_passed {
            self.human_passes += 1;
        }
    }

    fn into_bin(self) -> CalibrationBin {
        CalibrationBin {
            normalized_score: self.normalized_score,
            count: self.count,
            human_mean_normalized_score: self.human_score_sum / self.count as f32,
            human_pass_rate: rate(self.human_passes, self.count),
        }
    }
}

fn score_bin_key(normalized_score: f32) -> i32 {
    (normalized_score.clamp(0.0, 1.0) * 1000.0).round() as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::types::{JudgeCriteria, JudgeResult};

    #[test]
    fn requires_overlapping_cases() {
        let human_ratings = vec![HumanRating {
            case_id: "case-1".to_string(),
            trace_id: "trace-1".to_string(),
            score: 4,
            passed: None,
            notes: None,
        }];

        let error = CalibrationModel::fit(&human_ratings, &[], 3)
            .unwrap_err()
            .to_string();

        assert!(error.contains("overlapping case IDs"));
    }

    #[test]
    fn fits_and_applies_calibration_model_to_evaluation_results() {
        let human_ratings = vec![
            HumanRating {
                case_id: "case-1".to_string(),
                trace_id: "trace-1".to_string(),
                score: 4,
                passed: None,
                notes: None,
            },
            HumanRating {
                case_id: "case-2".to_string(),
                trace_id: "trace-2".to_string(),
                score: 2,
                passed: None,
                notes: None,
            },
        ];
        let results = vec![
            EvaluationResult::from(judge_result("case-1", "trace-1", 3)),
            EvaluationResult::from(judge_result("case-2", "trace-2", 2)),
        ];

        let model = CalibrationModel::fit(&human_ratings, &results, 3).unwrap();
        let calibrated = model.apply(results[0].clone());

        assert_eq!(model.evaluator_name.as_deref(), Some("test-judge"));
        assert_eq!(model.bins.len(), 2);
        assert_eq!(calibrated.calibrated_score, Some(1.0));
    }

    fn judge_result(case_id: &str, trace_id: &str, score: u8) -> JudgeResult {
        JudgeResult {
            case_id: case_id.to_string(),
            trace_id: trace_id.to_string(),
            judge_name: "test-judge".to_string(),
            score,
            passed: score >= 3,
            evaluation: "ok".to_string(),
            criteria: JudgeCriteria {
                relevance: true,
                correctness: true,
                completeness: true,
                safety: true,
            },
        }
    }
}
