use serde::{Deserialize, Serialize};
#[cfg(feature = "llm-judge-openai")]
use serde_json::Value;

use crate::evaluation::{EvaluationCriteria, EvaluationResult, ScoreScale};
#[cfg(feature = "llm-judge-openai")]
use crate::providers::chat::ResponseSchema;
use crate::{Result, TraceEvalError};

pub type JudgeCriteria = EvaluationCriteria;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeResult {
    pub case_id: String,
    pub trace_id: String,
    pub judge_name: String,
    pub score: u8,
    pub passed: bool,
    pub evaluation: String,
    pub criteria: JudgeCriteria,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "llm-judge-openai", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct JudgePayload {
    /// Concise explanation of the score. This must not contain hidden chain-of-thought.
    pub evaluation: String,
    /// 1=bad, 2=weak, 3=good, 4=excellent.
    pub score: u8,
    pub criteria: JudgeCriteria,
}

impl JudgePayload {
    pub const MIN_SCORE: u8 = 1;
    pub const MAX_SCORE: u8 = 4;

    pub fn into_result(
        self,
        case_id: impl Into<String>,
        trace_id: impl Into<String>,
        judge_name: impl Into<String>,
        pass_threshold: u8,
    ) -> Result<JudgeResult> {
        Self::validate_score(self.score)?;
        Self::validate_score(pass_threshold)?;

        Ok(JudgeResult {
            case_id: case_id.into(),
            trace_id: trace_id.into(),
            judge_name: judge_name.into(),
            score: self.score,
            passed: self.score >= pass_threshold,
            evaluation: self.evaluation,
            criteria: self.criteria,
        })
    }

    pub fn validate_score(score: u8) -> Result<()> {
        if (Self::MIN_SCORE..=Self::MAX_SCORE).contains(&score) {
            Ok(())
        } else {
            Err(TraceEvalError::InvalidScore {
                score: f32::from(score),
                scale: "four_point_judge".to_string(),
            })
        }
    }
}

#[cfg(feature = "llm-judge-openai")]
impl JudgePayload {
    pub fn response_schema() -> anyhow::Result<ResponseSchema> {
        let mut response_schema = ResponseSchema::strict_json::<Self>(
            "trace_eval_judge_result",
            "Judgment result for one trace-derived evaluation case.",
        )?;
        Self::constrain_judge_score(&mut response_schema.schema);
        Ok(response_schema)
    }

    fn constrain_judge_score(schema: &mut Value) {
        if let Some(score) = schema
            .pointer_mut("/properties/score")
            .and_then(Value::as_object_mut)
        {
            score.insert("minimum".to_string(), Value::from(JudgePayload::MIN_SCORE));
            score.insert("maximum".to_string(), Value::from(JudgePayload::MAX_SCORE));
        }
    }
}

impl From<JudgeResult> for EvaluationResult {
    fn from(result: JudgeResult) -> Self {
        EvaluationResult::from_ids(
            result.case_id,
            result.trace_id,
            result.judge_name,
            f32::from(result.score),
            ScoreScale::FourPoint,
            result.passed,
            result.evaluation,
        )
        .with_criteria(result.criteria)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_payload_fields() {
        let json = r#"{
            "evaluation": "fine",
            "score": 3,
            "passed": true,
            "criteria": {
                "relevance": true,
                "correctness": true,
                "completeness": true,
                "safety": true
            }
        }"#;

        assert!(serde_json::from_str::<JudgePayload>(json).is_err());
    }

    #[test]
    fn score_validation_enforces_scale() {
        assert!(JudgePayload::validate_score(1).is_ok());
        assert!(JudgePayload::validate_score(4).is_ok());
        assert!(JudgePayload::validate_score(0).is_err());
        assert!(JudgePayload::validate_score(5).is_err());
    }

    #[cfg(feature = "llm-judge-openai")]
    #[test]
    fn response_schema_is_generated_from_payload_type_and_constrained() {
        let schema = JudgePayload::response_schema().unwrap().schema;

        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["score"]["minimum"], 1);
        assert_eq!(schema["properties"]["score"]["maximum"], 4);
        assert!(schema.get("$schema").is_none());
    }
}
