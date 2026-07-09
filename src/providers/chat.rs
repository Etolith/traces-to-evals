use anyhow::Result;
use serde::de::DeserializeOwned;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct ResponseSchema {
    pub name: String,
    pub description: Option<String>,
    pub schema: Value,
    pub strict: bool,
}

#[derive(Debug, Clone)]
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
}
