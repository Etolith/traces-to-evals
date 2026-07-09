use std::collections::HashMap;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::judge::types::JudgeResult;

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
pub struct CalibrationSummary {
    pub compared: usize,
    pub exact_matches: usize,
    pub within_one: usize,
    pub pass_agreements: usize,
    pub mean_absolute_error: f32,
}

impl CalibrationSummary {
    pub fn exact_match_rate(&self) -> f32 {
        rate(self.exact_matches, self.compared)
    }

    pub fn within_one_rate(&self) -> f32 {
        rate(self.within_one, self.compared)
    }

    pub fn pass_agreement_rate(&self) -> f32 {
        rate(self.pass_agreements, self.compared)
    }
}

pub fn calibrate_judge_results(
    human_ratings: &[HumanRating],
    judge_results: &[JudgeResult],
    pass_threshold: u8,
) -> Result<CalibrationSummary> {
    validate_threshold(pass_threshold)?;

    let judge_by_case = judge_results
        .iter()
        .map(|result| (result.case_id.as_str(), result))
        .collect::<HashMap<_, _>>();

    let mut compared = 0usize;
    let mut exact_matches = 0usize;
    let mut within_one = 0usize;
    let mut pass_agreements = 0usize;
    let mut absolute_error = 0u32;

    for rating in human_ratings {
        validate_score(rating.score, "human rating")?;

        let Some(judge) = judge_by_case.get(rating.case_id.as_str()) else {
            continue;
        };

        validate_score(judge.score, "judge result")?;

        compared += 1;

        let delta = u8::abs_diff(rating.score, judge.score);
        absolute_error += u32::from(delta);

        if delta == 0 {
            exact_matches += 1;
        }

        if delta <= 1 {
            within_one += 1;
        }

        let human_passed = rating.passed.unwrap_or(rating.score >= pass_threshold);
        let judge_passed = judge.score >= pass_threshold;
        if human_passed == judge_passed {
            pass_agreements += 1;
        }
    }

    if compared == 0 {
        return Err(anyhow!(
            "cannot calibrate judge results without overlapping case IDs"
        ));
    }

    Ok(CalibrationSummary {
        compared,
        exact_matches,
        within_one,
        pass_agreements,
        mean_absolute_error: absolute_error as f32 / compared as f32,
    })
}

fn validate_threshold(pass_threshold: u8) -> Result<()> {
    if (1..=4).contains(&pass_threshold) {
        Ok(())
    } else {
        Err(anyhow!(
            "pass_threshold must be between 1 and 4, got {}",
            pass_threshold
        ))
    }
}

fn validate_score(score: u8, label: &str) -> Result<()> {
    if (1..=4).contains(&score) {
        Ok(())
    } else {
        Err(anyhow!(
            "{label} score must be between 1 and 4, got {score}"
        ))
    }
}

fn rate(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::types::JudgeCriteria;

    #[test]
    fn calibrates_judge_scores_against_human_ratings() {
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
        let judge_results = vec![
            judge_result("case-1", "trace-1", 3),
            judge_result("case-2", "trace-2", 2),
        ];

        let summary = calibrate_judge_results(&human_ratings, &judge_results, 3).unwrap();

        assert_eq!(summary.compared, 2);
        assert_eq!(summary.exact_matches, 1);
        assert_eq!(summary.within_one, 2);
        assert_eq!(summary.pass_agreements, 2);
        assert_eq!(summary.mean_absolute_error, 0.5);
    }

    #[test]
    fn requires_overlapping_cases() {
        let human_ratings = vec![HumanRating {
            case_id: "case-1".to_string(),
            trace_id: "trace-1".to_string(),
            score: 4,
            passed: None,
            notes: None,
        }];

        let error = calibrate_judge_results(&human_ratings, &[], 3)
            .unwrap_err()
            .to_string();

        assert!(error.contains("overlapping case IDs"));
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
