use anyhow::{Result, anyhow};
use openai_dive::v1::api::Client;

use crate::judge::LlmJudge;
use crate::judge::prompt::JudgePrompt;
use crate::judge::types::{JudgePayload, JudgeResult};
use crate::model::EvalCase;
use crate::providers::chat::{ChatClient, ChatRequest};
use crate::providers::openai_dive::chat::OpenAiChatClient;

pub struct OpenAiJudge<C = OpenAiChatClient> {
    chat_client: C,
    model: String,
    pass_threshold: u8,
}

impl<C> OpenAiJudge<C> {
    pub const DEFAULT_PASS_THRESHOLD: u8 = 3;
}

impl OpenAiJudge<OpenAiChatClient> {
    pub fn from_env(model: impl Into<String>) -> Self {
        Self {
            chat_client: OpenAiChatClient::from_env(),
            model: model.into(),
            pass_threshold: Self::DEFAULT_PASS_THRESHOLD,
        }
    }

    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            chat_client: OpenAiChatClient::new(client),
            model: model.into(),
            pass_threshold: Self::DEFAULT_PASS_THRESHOLD,
        }
    }
}

impl<C> OpenAiJudge<C>
where
    C: ChatClient,
{
    pub fn with_pass_threshold(mut self, pass_threshold: u8) -> Self {
        self.pass_threshold = pass_threshold;
        self
    }

    pub async fn judge_case(&self, case: &EvalCase) -> Result<JudgeResult> {
        let actual_output = case
            .actual_output
            .as_deref()
            .ok_or_else(|| anyhow!("case {} has no actual_output", case.id))?;

        let prompt = JudgePrompt::build(case, actual_output);
        let payload: JudgePayload = self
            .chat_client
            .complete_json(ChatRequest {
                model: self.model.clone(),
                system_prompt: prompt.system,
                user_prompt: prompt.user,
                response_schema: JudgePayload::response_schema()?,
                context_id: Some(case.id.clone()),
            })
            .await?;

        payload.into_result(
            case.id.clone(),
            case.trace_id.clone(),
            format!("openai/{}", self.model),
            self.pass_threshold,
        )
    }
}

#[async_trait::async_trait]
impl<C> LlmJudge for OpenAiJudge<C>
where
    C: ChatClient,
{
    async fn judge_case(&self, case: &EvalCase) -> Result<JudgeResult> {
        OpenAiJudge::judge_case(self, case).await
    }
}
