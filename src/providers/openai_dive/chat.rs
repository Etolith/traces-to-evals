use std::time::Instant;

use anyhow::Result;
use openai_dive::v1::api::Client;
use openai_dive::v1::resources::chat::{
    ChatCompletionParametersBuilder, ChatCompletionResponse, ChatCompletionResponseFormat,
    ChatMessage, ChatMessageContent, JsonSchemaBuilder,
};
use openai_dive::v1::resources::shared::Usage;
use serde::{Serialize, de::DeserializeOwned};

use crate::learned::{
    ChatCompletionEnvelopeV1, ProviderExecutionFailureV1, ProviderExecutionStageV1,
    ProviderResponseEnvelopeV1, ProviderTokenUsageV1, canonical_content_id,
};
use crate::providers::chat::{ChatClient, ChatRequest, ResponseSchema};

pub struct OpenAiChatClient {
    client: Client,
}

impl OpenAiChatClient {
    pub fn from_env() -> Self {
        Self {
            client: Client::new_from_env(),
        }
    }

    pub fn new(client: Client) -> Self {
        Self { client }
    }

    async fn execute(&self, request: ChatRequest) -> Result<RawChatCompletion> {
        let requested_model = request.model.clone();
        let context_id = request.context_id.clone();
        let parameters = ChatCompletionParametersBuilder::default()
            .model(request.model)
            .messages(vec![
                ChatMessage::System {
                    content: ChatMessageContent::Text(request.system_prompt),
                    name: None,
                },
                ChatMessage::User {
                    content: ChatMessageContent::Text(request.user_prompt),
                    name: None,
                },
            ])
            .response_format(ChatCompletionResponseFormat::JsonSchema {
                json_schema: build_json_schema(request.response_schema)?,
            })
            .build()?;
        let request_hash = canonical_content_id("traceeval.openai.chat-request.v1", &parameters)?;
        let started = Instant::now();
        let response = match self.client.chat().create(parameters).await {
            Ok(response) => response,
            Err(error) => {
                let latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                return Err(ProviderExecutionFailureV1 {
                    stage: ProviderExecutionStageV1::Transport,
                    message: context_message(
                        format!("failed to call OpenAI through openai_dive: {error}"),
                        &context_id,
                    ),
                    requested_model,
                    request_hash,
                    attempts: 1,
                    latency_ms,
                    provider_response: None,
                }
                .into());
            }
        };
        let latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        response_to_raw(
            response,
            requested_model,
            request_hash,
            latency_ms,
            &context_id,
        )
    }
}

#[async_trait::async_trait]
impl ChatClient for OpenAiChatClient {
    async fn complete_json<T>(&self, request: ChatRequest) -> Result<T>
    where
        T: DeserializeOwned + Send,
    {
        let context_id = request.context_id.clone();
        let raw = self.execute(request).await?;
        serde_json::from_str(&raw.content).map_err(|error| {
            provider_failure(
                ProviderExecutionStageV1::OutputParsing,
                context_message(
                    format!("failed to parse model response as requested JSON: {error}"),
                    &context_id,
                ),
                raw.envelope,
            )
        })
    }

    async fn complete_json_enveloped<T>(
        &self,
        request: ChatRequest,
    ) -> Result<ChatCompletionEnvelopeV1<T>>
    where
        T: DeserializeOwned + Serialize + Send,
    {
        let context_id = request.context_id.clone();
        let raw = self.execute(request).await?;
        let output = serde_json::from_str(&raw.content).map_err(|error| {
            provider_failure(
                ProviderExecutionStageV1::OutputParsing,
                context_message(
                    format!("failed to parse model response as requested JSON: {error}"),
                    &context_id,
                ),
                raw.envelope.clone(),
            )
        })?;
        Ok(ChatCompletionEnvelopeV1::new(output, raw.envelope)?)
    }
}

#[derive(Debug)]
struct RawChatCompletion {
    content: String,
    envelope: ProviderResponseEnvelopeV1,
}

fn response_to_raw(
    response: ChatCompletionResponse,
    requested_model: String,
    request_hash: String,
    latency_ms: u64,
    context_id: &Option<String>,
) -> Result<RawChatCompletion> {
    let response_hash = canonical_content_id("traceeval.openai.chat-response.v1", &response)?;
    let finish_reason = response.choices.first().and_then(|choice| {
        choice.finish_reason.as_ref().and_then(|reason| {
            serde_json::to_value(reason)
                .ok()
                .and_then(|value| value.as_str().map(str::to_string))
        })
    });
    let envelope = ProviderResponseEnvelopeV1 {
        provider: Some("openai".to_string()),
        requested_model,
        returned_model: Some(response.model.clone()),
        response_id: response.id.clone(),
        finish_reason,
        system_fingerprint: response.system_fingerprint.clone(),
        service_tier: response.service_tier.clone(),
        usage: response.usage.as_ref().map(map_usage),
        request_hash,
        response_hash,
        attempts: 1,
        latency_ms,
    };
    let choice = response.choices.first().ok_or_else(|| {
        provider_failure(
            ProviderExecutionStageV1::ResponseValidation,
            context_message("model returned no choices", context_id),
            envelope.clone(),
        )
    })?;
    if let ChatMessage::Assistant {
        refusal: Some(refusal),
        ..
    } = &choice.message
    {
        return Err(provider_failure(
            ProviderExecutionStageV1::ResponseValidation,
            context_message(
                format!("model refused structured response: {refusal}"),
                context_id,
            ),
            envelope,
        ));
    }
    let content = choice.message.text().ok_or_else(|| {
        provider_failure(
            ProviderExecutionStageV1::ResponseValidation,
            context_message("model response was not simple text content", context_id),
            envelope.clone(),
        )
    })?;
    Ok(RawChatCompletion {
        content: content.to_string(),
        envelope,
    })
}

fn map_usage(usage: &Usage) -> ProviderTokenUsageV1 {
    ProviderTokenUsageV1 {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        total_tokens: Some(usage.total_tokens),
        cached_input_tokens: usage
            .prompt_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens),
        reasoning_tokens: usage
            .completion_tokens_details
            .as_ref()
            .map(|details| details.reasoning_tokens),
    }
}

fn build_json_schema(
    schema: ResponseSchema,
) -> Result<openai_dive::v1::resources::chat::JsonSchema> {
    let mut builder = JsonSchemaBuilder::default();
    builder
        .name(schema.name)
        .schema(schema.schema)
        .strict(schema.strict);

    if let Some(description) = schema.description {
        builder.description(description);
    }

    Ok(builder.build()?)
}

fn provider_failure(
    stage: ProviderExecutionStageV1,
    message: String,
    envelope: ProviderResponseEnvelopeV1,
) -> anyhow::Error {
    ProviderExecutionFailureV1 {
        stage,
        message,
        requested_model: envelope.requested_model.clone(),
        request_hash: envelope.request_hash.clone(),
        attempts: envelope.attempts,
        latency_ms: envelope.latency_ms,
        provider_response: Some(envelope),
    }
    .into()
}

fn context_message(message: impl Into<String>, context_id: &Option<String>) -> String {
    let message = message.into();
    match context_id {
        Some(context_id) => format!("{message} for {context_id}"),
        None => message,
    }
}

#[cfg(test)]
mod tests {
    use openai_dive::v1::resources::chat::ChatCompletionChoice;
    use openai_dive::v1::resources::shared::{
        CompletionTokensDetails, FinishReason, PromptTokensDetails,
    };

    use super::*;

    #[test]
    fn response_mapping_preserves_openai_identity_usage_and_finish_reason() {
        let response = ChatCompletionResponse {
            id: Some("chatcmpl-1".to_string()),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage::Assistant {
                    content: Some(ChatMessageContent::Text(
                        r#"{"verdict":"pass"}"#.to_string(),
                    )),
                    reasoning: None,
                    reasoning_content: None,
                    refusal: None,
                    name: None,
                    audio: None,
                    tool_calls: None,
                },
                finish_reason: Some(FinishReason::StopSequenceReached),
                logprobs: None,
            }],
            created: 1,
            model: "gpt-returned".to_string(),
            service_tier: Some("default".to_string()),
            system_fingerprint: Some("fp-1".to_string()),
            object: "chat.completion".to_string(),
            usage: Some(Usage {
                prompt_tokens: Some(11),
                completion_tokens: Some(7),
                total_tokens: 18,
                prompt_tokens_details: Some(PromptTokensDetails {
                    audio_tokens: None,
                    cached_tokens: 3,
                }),
                completion_tokens_details: Some(CompletionTokensDetails {
                    reasoning_tokens: 2,
                    audio_tokens: None,
                    accepted_prediction_tokens: None,
                    rejected_prediction_tokens: None,
                }),
            }),
        };

        let raw = response_to_raw(
            response,
            "gpt-requested".to_string(),
            format!("sha256:{}", "0".repeat(64)),
            42,
            &None,
        )
        .unwrap();

        assert_eq!(raw.content, r#"{"verdict":"pass"}"#);
        assert_eq!(raw.envelope.provider.as_deref(), Some("openai"));
        assert_eq!(raw.envelope.requested_model, "gpt-requested");
        assert_eq!(raw.envelope.returned_model.as_deref(), Some("gpt-returned"));
        assert_eq!(raw.envelope.response_id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(raw.envelope.finish_reason.as_deref(), Some("stop"));
        assert_eq!(raw.envelope.latency_ms, 42);
        raw.envelope.validate().unwrap();
        let usage = raw.envelope.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(11));
        assert_eq!(usage.output_tokens, Some(7));
        assert_eq!(usage.total_tokens, Some(18));
        assert_eq!(usage.cached_input_tokens, Some(3));
        assert_eq!(usage.reasoning_tokens, Some(2));
    }

    #[test]
    fn refusal_failure_preserves_paid_response_metadata() {
        let response = ChatCompletionResponse {
            id: Some("chatcmpl-refusal".to_string()),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage::Assistant {
                    content: None,
                    reasoning: None,
                    reasoning_content: None,
                    refusal: Some("cannot comply".to_string()),
                    name: None,
                    audio: None,
                    tool_calls: None,
                },
                finish_reason: Some(FinishReason::StopSequenceReached),
                logprobs: None,
            }],
            created: 1,
            model: "gpt-returned".to_string(),
            service_tier: Some("default".to_string()),
            system_fingerprint: Some("fp-refusal".to_string()),
            object: "chat.completion".to_string(),
            usage: Some(Usage {
                prompt_tokens: Some(13),
                completion_tokens: Some(2),
                total_tokens: 15,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
        };

        let error = response_to_raw(
            response,
            "gpt-requested".to_string(),
            format!("sha256:{}", "0".repeat(64)),
            51,
            &Some("case-refusal".to_string()),
        )
        .unwrap_err();
        let failure = error.downcast_ref::<ProviderExecutionFailureV1>().unwrap();
        failure.validate().unwrap();
        assert_eq!(failure.stage, ProviderExecutionStageV1::ResponseValidation);
        let envelope = failure.provider_response.as_ref().unwrap();
        assert_eq!(envelope.response_id.as_deref(), Some("chatcmpl-refusal"));
        assert_eq!(envelope.usage.as_ref().unwrap().total_tokens, Some(15));
        assert_eq!(envelope.latency_ms, 51);
    }
}
