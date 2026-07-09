pub mod types;

#[cfg(feature = "llm-judge-openai")]
pub mod openai_dive_judge;

#[cfg(feature = "llm-judge-openai")]
use anyhow::Result;
#[cfg(feature = "llm-judge-openai")]
use types::JudgeResult;

#[cfg(feature = "llm-judge-openai")]
use crate::model::EvalCase;

#[cfg(feature = "llm-judge-openai")]
#[async_trait::async_trait]
pub trait LlmJudge {
    async fn judge_case(&self, case: &EvalCase) -> Result<JudgeResult>;
}
