pub use crate::evaluation::{
    EvaluationResult as ScoredResult, RunScore, ScoreScale, WeightedAggregate,
};

pub fn normalize_judge_score(score: u8) -> f32 {
    ScoreScale::FourPoint.normalize(f32::from(score))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::types::{JudgeCriteria, JudgeResult};

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
