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
