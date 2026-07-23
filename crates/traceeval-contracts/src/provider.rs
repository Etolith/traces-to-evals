use std::fmt;

use serde::{Deserialize, Serialize};

use super::{ContractError, require_non_empty, require_sha256};

pub const CHAT_COMPLETION_ENVELOPE_SCHEMA_VERSION: &str = "traceeval.chat_completion_envelope.v1";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderTokenUsageV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

impl ProviderTokenUsageV1 {
    fn validate(&self) -> Result<(), ContractError> {
        if let (Some(input), Some(total)) = (self.input_tokens, self.total_tokens)
            && input > total
        {
            return Err(provider_error("input_tokens cannot exceed total_tokens"));
        }
        if let (Some(output), Some(total)) = (self.output_tokens, self.total_tokens)
            && output > total
        {
            return Err(provider_error("output_tokens cannot exceed total_tokens"));
        }
        if let (Some(input), Some(output), Some(total)) =
            (self.input_tokens, self.output_tokens, self.total_tokens)
            && input.saturating_add(output) > total
        {
            return Err(provider_error(
                "input_tokens plus output_tokens cannot exceed total_tokens",
            ));
        }
        if let (Some(cached), Some(input)) = (self.cached_input_tokens, self.input_tokens)
            && cached > input
        {
            return Err(provider_error(
                "cached_input_tokens cannot exceed input_tokens",
            ));
        }
        if let (Some(reasoning), Some(output)) = (self.reasoning_tokens, self.output_tokens)
            && reasoning > output
        {
            return Err(provider_error(
                "reasoning_tokens cannot exceed output_tokens",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderExecutionStageV1 {
    Transport,
    ResponseValidation,
    OutputParsing,
}

/// A provider execution that did not yield a valid typed output.
///
/// `provider_response` remains present for refusals, empty responses, and parsing
/// failures so callers can still account for response identity, usage, and latency.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderExecutionFailureV1 {
    pub stage: ProviderExecutionStageV1,
    pub message: String,
    pub requested_model: String,
    pub request_hash: String,
    pub attempts: u32,
    pub latency_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_response: Option<ProviderResponseEnvelopeV1>,
}

impl ProviderExecutionFailureV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        require_non_empty(&self.message, "failure message", provider_error)?;
        require_non_empty(&self.requested_model, "requested_model", provider_error)?;
        require_sha256(&self.request_hash, "request_hash", provider_error)?;
        if self.attempts == 0 {
            return Err(provider_error("attempts must be greater than zero"));
        }
        if self.stage != ProviderExecutionStageV1::Transport && self.provider_response.is_none() {
            return Err(provider_error(
                "post-transport provider failure requires a response envelope",
            ));
        }
        if let Some(response) = &self.provider_response {
            response.validate()?;
            if response.requested_model != self.requested_model
                || response.request_hash != self.request_hash
                || response.attempts != self.attempts
                || response.latency_ms != self.latency_ms
            {
                return Err(provider_error(
                    "provider failure metadata does not match its response envelope",
                ));
            }
        }
        Ok(())
    }
}

impl fmt::Display for ProviderExecutionFailureV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ProviderExecutionFailureV1 {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderResponseEnvelopeV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    pub requested_model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub returned_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ProviderTokenUsageV1>,
    pub request_hash: String,
    pub response_hash: String,
    pub attempts: u32,
    pub latency_ms: u64,
}

impl ProviderResponseEnvelopeV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        if let Some(provider) = &self.provider {
            require_non_empty(provider, "provider", provider_error)?;
        }
        require_non_empty(&self.requested_model, "requested_model", provider_error)?;
        if let Some(returned_model) = &self.returned_model {
            require_non_empty(returned_model, "returned_model", provider_error)?;
        }
        if let Some(response_id) = &self.response_id {
            require_non_empty(response_id, "response_id", provider_error)?;
        }
        if let Some(finish_reason) = &self.finish_reason {
            require_non_empty(finish_reason, "finish_reason", provider_error)?;
        }
        if let Some(system_fingerprint) = &self.system_fingerprint {
            require_non_empty(system_fingerprint, "system_fingerprint", provider_error)?;
        }
        if let Some(service_tier) = &self.service_tier {
            require_non_empty(service_tier, "service_tier", provider_error)?;
        }
        require_sha256(&self.request_hash, "request_hash", provider_error)?;
        require_sha256(&self.response_hash, "response_hash", provider_error)?;
        if self.attempts == 0 {
            return Err(provider_error("attempts must be greater than zero"));
        }
        if let Some(usage) = &self.usage {
            usage.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionEnvelopeV1<T> {
    pub schema_version: String,
    pub output: T,
    pub provider_response: ProviderResponseEnvelopeV1,
}

impl<T> ChatCompletionEnvelopeV1<T> {
    pub fn new(
        output: T,
        provider_response: ProviderResponseEnvelopeV1,
    ) -> Result<Self, ContractError> {
        let envelope = Self {
            schema_version: CHAT_COMPLETION_ENVELOPE_SCHEMA_VERSION.to_string(),
            output,
            provider_response,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != CHAT_COMPLETION_ENVELOPE_SCHEMA_VERSION {
            return Err(provider_error(
                "unsupported chat completion envelope schema version",
            ));
        }
        self.provider_response.validate()
    }
}

fn provider_error(message: impl Into<String>) -> ContractError {
    ContractError::InvalidProvider(message.into())
}
