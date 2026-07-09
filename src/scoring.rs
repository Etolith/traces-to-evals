use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::graders::GradeResult;
use crate::judge::types::{JudgeCriteria, JudgeResult};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredResult {
    pub case_id: String,
    pub trace_id: String,
    pub scorer_name: String,
    pub raw_score: f32,
    pub normalized_score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibrated_score: Option<f32>,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<String>,
    pub evaluation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<JudgeCriteria>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl From<GradeResult> for ScoredResult {
    fn from(result: GradeResult) -> Self {
        Self {
            case_id: result.case_id,
            trace_id: result.trace_id,
            scorer_name: result.grader_name,
            raw_score: f32::from(result.score),
            normalized_score: f32::from(result.score),
            calibrated_score: None,
            passed: result.passed,
            confidence: Some(1.0),
            cluster_id: None,
            evaluation: result.evaluation,
            criteria: None,
            metadata: BTreeMap::new(),
        }
    }
}

impl From<JudgeResult> for ScoredResult {
    fn from(result: JudgeResult) -> Self {
        Self {
            case_id: result.case_id,
            trace_id: result.trace_id,
            scorer_name: result.judge_name,
            raw_score: f32::from(result.score),
            normalized_score: normalize_judge_score(result.score),
            calibrated_score: None,
            passed: result.passed,
            confidence: None,
            cluster_id: None,
            evaluation: result.evaluation,
            criteria: Some(result.criteria),
            metadata: BTreeMap::new(),
        }
    }
}

pub fn normalize_judge_score(score: u8) -> f32 {
    match score {
        0 | 1 => 0.0,
        2 => 1.0 / 3.0,
        3 => 2.0 / 3.0,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::types::JudgeCriteria;

    #[test]
    fn normalizes_judge_score_to_unit_interval() {
        let result = JudgeResult {
            case_id: "case-1".to_string(),
            trace_id: "trace-1".to_string(),
            judge_name: "judge".to_string(),
            score: 3,
            passed: true,
            evaluation: "ok".to_string(),
            criteria: JudgeCriteria {
                relevance: true,
                correctness: true,
                completeness: true,
                safety: true,
            },
        };

        let scored = ScoredResult::from(result);

        assert_eq!(scored.raw_score, 3.0);
        assert_eq!(scored.normalized_score, 2.0 / 3.0);
    }
}
