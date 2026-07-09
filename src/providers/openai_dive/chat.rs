use anyhow::{Context, Result, anyhow};
use openai_dive::v1::api::Client;
use openai_dive::v1::resources::chat::{
    ChatCompletionParametersBuilder, ChatCompletionResponseFormat, ChatMessage, ChatMessageContent,
    JsonSchemaBuilder,
};
use serde::de::DeserializeOwned;

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
}

#[async_trait::async_trait]
impl ChatClient for OpenAiChatClient {
    async fn complete_json<T>(&self, request: ChatRequest) -> Result<T>
    where
        T: DeserializeOwned + Send,
    {
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

        let response = self
            .client
            .chat()
            .create(parameters)
            .await
            .context("failed to call OpenAI through openai_dive")?;

        let message = &response
            .choices
            .first()
            .ok_or_else(|| context_error("model returned no choices", &request.context_id))?
            .message;

        if let ChatMessage::Assistant {
            refusal: Some(refusal),
            ..
        } = message
        {
            return Err(context_error(
                format!("model refused structured response: {refusal}"),
                &request.context_id,
            ));
        }

        let content = message.text().ok_or_else(|| {
            context_error(
                "model response was not simple text content",
                &request.context_id,
            )
        })?;

        serde_json::from_str(content).with_context(|| {
            context_message(
                "failed to parse model response as requested JSON",
                &request.context_id,
            )
        })
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

fn context_error(message: impl Into<String>, context_id: &Option<String>) -> anyhow::Error {
    anyhow!(context_message(message, context_id))
}

fn context_message(message: impl Into<String>, context_id: &Option<String>) -> String {
    let message = message.into();
    match context_id {
        Some(context_id) => format!("{message} for {context_id}"),
        None => message,
    }
}
