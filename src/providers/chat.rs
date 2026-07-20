use std::time::Instant;

use anyhow::Result;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::learned::{
    ChatCompletionEnvelopeV1, ProviderExecutionFailureV1, ProviderExecutionStageV1,
    ProviderResponseEnvelopeV1, canonical_content_id,
};

#[derive(Debug, Clone, Serialize)]
pub struct ResponseSchema {
    pub name: String,
    pub description: Option<String>,
    pub schema: Value,
    pub strict: bool,
}

#[cfg(any(feature = "llm-judge-openai", feature = "cluster-label-openai"))]
impl ResponseSchema {
    pub fn strict_json<T>(name: impl Into<String>, description: impl Into<String>) -> Result<Self>
    where
        T: schemars::JsonSchema,
    {
        Ok(Self {
            name: name.into(),
            description: Some(description.into()),
            schema: openai_schema_for_type::<T>()?,
            strict: true,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub system_prompt: String,
    pub user_prompt: String,
    pub response_schema: ResponseSchema,
    pub context_id: Option<String>,
}

#[async_trait::async_trait]
pub trait ChatClient: Send + Sync {
    async fn complete_json<T>(&self, request: ChatRequest) -> Result<T>
    where
        T: DeserializeOwned + Send;

    /// Completes a structured request while preserving provider execution metadata.
    ///
    /// Existing clients remain source-compatible: the default delegates to
    /// `complete_json` and records the metadata that can be known locally.
    async fn complete_json_enveloped<T>(
        &self,
        request: ChatRequest,
    ) -> Result<ChatCompletionEnvelopeV1<T>>
    where
        T: DeserializeOwned + Serialize + Send,
    {
        let requested_model = request.model.clone();
        let context_id = request.context_id.clone();
        let request_hash = canonical_content_id("traceeval.chat-request.v1", &request)?;
        let started = Instant::now();
        let output = match self.complete_json(request).await {
            Ok(output) => output,
            Err(error) => {
                let latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                return Err(ProviderExecutionFailureV1 {
                    stage: ProviderExecutionStageV1::Transport,
                    message: match context_id {
                        Some(context_id) => format!("{error} for {context_id}"),
                        None => error.to_string(),
                    },
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
        let response_hash = canonical_content_id("traceeval.chat-response.v1", &output)?;
        Ok(ChatCompletionEnvelopeV1::new(
            output,
            ProviderResponseEnvelopeV1 {
                provider: None,
                requested_model,
                returned_model: None,
                response_id: None,
                finish_reason: None,
                system_fingerprint: None,
                service_tier: None,
                usage: None,
                request_hash,
                response_hash,
                attempts: 1,
                latency_ms,
            },
        )?)
    }
}

#[cfg(any(feature = "llm-judge-openai", feature = "cluster-label-openai"))]
fn openai_schema_for_type<T>() -> Result<Value>
where
    T: schemars::JsonSchema,
{
    let mut schema = serde_json::to_value(schemars::schema_for!(T))?;
    normalize_openai_schema(&mut schema);
    Ok(schema)
}

#[cfg(any(feature = "llm-judge-openai", feature = "cluster-label-openai"))]
fn normalize_openai_schema(schema: &mut Value) {
    match schema {
        Value::Object(object) => {
            object.remove("$schema");
            object.remove("title");
            object.remove("format");

            for value in object.values_mut() {
                normalize_openai_schema(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                normalize_openai_schema(value);
            }
        }
        _ => {}
    }
}
