#[cfg(any(feature = "embeddings-openai", feature = "cluster-label-openai"))]
use openai_dive::v1::api::Client;
#[cfg(feature = "embeddings-openai")]
use openai_dive::v1::resources::embedding::{
    EmbeddingEncodingFormat, EmbeddingInput, EmbeddingOutput, EmbeddingParametersBuilder,
};

#[cfg(feature = "embeddings-openai")]
use crate::clustering::EmbeddingProvider;
#[cfg(feature = "cluster-label-openai")]
use crate::clustering::labeling::{ClusterLabelPayload, ClusterLabelPrompt};
#[cfg(feature = "cluster-label-openai")]
use crate::clustering::{ClusterLabel, ClusterLabeler, DiscoveredCluster};
#[cfg(feature = "cluster-label-openai")]
use crate::model::EvalCase;
#[cfg(feature = "cluster-label-openai")]
use crate::providers::chat::{ChatClient, ChatRequest};
#[cfg(feature = "cluster-label-openai")]
use crate::providers::openai_dive::chat::OpenAiChatClient;
use crate::{Result, TraceEvalError};

#[cfg(feature = "embeddings-openai")]
pub const OPENAI_EMBEDDING_PROVIDER_NAME: &str = "openai";
#[cfg(feature = "cluster-label-openai")]
pub const OPENAI_CLUSTER_LABEL_PROVIDER_NAME: &str = "openai";

#[cfg(feature = "embeddings-openai")]
#[async_trait::async_trait]
pub trait TextEmbeddingClient: Send + Sync {
    async fn embed_texts(
        &self,
        model: &str,
        texts: &[String],
        dimensions: Option<u32>,
    ) -> Result<Vec<Vec<f32>>>;
}

#[cfg(feature = "embeddings-openai")]
pub struct OpenAiEmbeddingClient {
    client: Client,
}

#[cfg(feature = "embeddings-openai")]
impl OpenAiEmbeddingClient {
    pub fn from_env() -> Self {
        Self {
            client: Client::new_from_env(),
        }
    }

    pub fn new(client: Client) -> Self {
        Self { client }
    }
}

#[cfg(feature = "embeddings-openai")]
#[async_trait::async_trait]
impl TextEmbeddingClient for OpenAiEmbeddingClient {
    async fn embed_texts(
        &self,
        model: &str,
        texts: &[String],
        dimensions: Option<u32>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut builder = EmbeddingParametersBuilder::default();
        builder
            .model(model.to_string())
            .input(EmbeddingInput::StringArray(texts.to_vec()))
            .encoding_format(EmbeddingEncodingFormat::Float);
        if let Some(dimensions) = dimensions {
            builder.dimensions(dimensions);
        }

        let response = self
            .client
            .embeddings()
            .create(
                builder
                    .build()
                    .map_err(|error| provider_error(error.to_string()))?,
            )
            .await
            .map_err(|error| provider_error(error.to_string()))?;

        let mut data = response.data;
        data.sort_by_key(|embedding| embedding.index);

        if data.len() != texts.len() {
            return Err(provider_error(format!(
                "provider returned {} embeddings for {} inputs",
                data.len(),
                texts.len()
            )));
        }

        data.into_iter()
            .map(|embedding| match embedding.embedding {
                EmbeddingOutput::Float(values) => Ok(values
                    .into_iter()
                    .map(|value| value as f32)
                    .collect::<Vec<_>>()),
                EmbeddingOutput::Base64(_) => Err(provider_error(
                    "provider returned base64 embeddings after float format request",
                )),
            })
            .collect()
    }
}

#[cfg(feature = "embeddings-openai")]
#[derive(Debug, Clone)]
pub struct OpenAiEmbeddingProvider<C = OpenAiEmbeddingClient> {
    client: C,
    model: String,
    dimensions: Option<u32>,
    batch_size: usize,
}

#[cfg(feature = "embeddings-openai")]
impl OpenAiEmbeddingProvider<OpenAiEmbeddingClient> {
    pub fn from_env(model: impl Into<String>) -> Self {
        Self::with_client(OpenAiEmbeddingClient::from_env(), model)
    }

    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self::with_client(OpenAiEmbeddingClient::new(client), model)
    }
}

#[cfg(feature = "embeddings-openai")]
impl<C> OpenAiEmbeddingProvider<C> {
    pub fn with_client(client: C, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            dimensions: None,
            batch_size: 128,
        }
    }

    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }
}

#[cfg(feature = "embeddings-openai")]
#[async_trait::async_trait]
impl<C> EmbeddingProvider for OpenAiEmbeddingProvider<C>
where
    C: TextEmbeddingClient,
{
    fn provider_name(&self) -> String {
        OPENAI_EMBEDDING_PROVIDER_NAME.to_string()
    }

    fn model_name(&self) -> String {
        self.model.clone()
    }

    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_size) {
            let chunk_vectors = self
                .client
                .embed_texts(&self.model, chunk, self.dimensions)
                .await?;
            if chunk_vectors.len() != chunk.len() {
                return Err(provider_error(format!(
                    "provider returned {} embeddings for {} inputs",
                    chunk_vectors.len(),
                    chunk.len()
                )));
            }
            vectors.extend(chunk_vectors);
        }

        Ok(vectors)
    }
}

#[cfg(feature = "cluster-label-openai")]
#[derive(Debug, Clone)]
pub struct OpenAiClusterLabeler<C = OpenAiChatClient> {
    chat_client: C,
    model: String,
}

#[cfg(feature = "cluster-label-openai")]
impl OpenAiClusterLabeler<OpenAiChatClient> {
    pub fn from_env(model: impl Into<String>) -> Self {
        Self::with_client(OpenAiChatClient::from_env(), model)
    }

    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self::with_client(OpenAiChatClient::new(client), model)
    }
}

#[cfg(feature = "cluster-label-openai")]
impl<C> OpenAiClusterLabeler<C> {
    pub fn with_client(chat_client: C, model: impl Into<String>) -> Self {
        Self {
            chat_client,
            model: model.into(),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(feature = "cluster-label-openai")]
#[async_trait::async_trait]
impl<C> ClusterLabeler for OpenAiClusterLabeler<C>
where
    C: ChatClient,
{
    fn labeler_name(&self) -> String {
        format!("{OPENAI_CLUSTER_LABEL_PROVIDER_NAME}/{}", self.model)
    }

    async fn label_cluster(
        &self,
        cluster: &DiscoveredCluster,
        examples: &[EvalCase],
    ) -> Result<ClusterLabel> {
        let prompt = ClusterLabelPrompt::build(cluster, examples);
        let response_schema = ClusterLabelPayload::response_schema()
            .map_err(|error| labeler_error(&cluster.id, error.to_string()))?;
        let payload: ClusterLabelPayload = self
            .chat_client
            .complete_json(ChatRequest {
                model: self.model.clone(),
                system_prompt: prompt.system,
                user_prompt: prompt.user,
                response_schema,
                context_id: Some(cluster.id.clone()),
            })
            .await
            .map_err(|error| labeler_error(&cluster.id, error.to_string()))?;

        payload.into_label(self.labeler_name(), cluster.id.clone())
    }
}

#[cfg(feature = "embeddings-openai")]
fn provider_error(message: impl Into<String>) -> TraceEvalError {
    TraceEvalError::EmbeddingProvider {
        provider: OPENAI_EMBEDDING_PROVIDER_NAME.to_string(),
        message: message.into(),
    }
}

#[cfg(feature = "cluster-label-openai")]
fn labeler_error(cluster_id: impl Into<String>, message: impl Into<String>) -> TraceEvalError {
    TraceEvalError::ClusterLabeling {
        provider: OPENAI_CLUSTER_LABEL_PROVIDER_NAME.to_string(),
        cluster_id: cluster_id.into(),
        message: message.into(),
    }
}

#[cfg(all(test, feature = "embeddings-openai"))]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::clustering::{ClusterTextProjector, DefaultClusterTextProjector};
    use crate::model::EvalCase;

    use super::*;

    type RecordedCalls = Arc<Mutex<Vec<(String, Vec<String>, Option<u32>)>>>;

    #[derive(Clone)]
    struct FakeEmbeddingClient {
        calls: RecordedCalls,
    }

    #[async_trait::async_trait]
    impl TextEmbeddingClient for FakeEmbeddingClient {
        async fn embed_texts(
            &self,
            model: &str,
            texts: &[String],
            dimensions: Option<u32>,
        ) -> Result<Vec<Vec<f32>>> {
            self.calls
                .lock()
                .unwrap()
                .push((model.to_string(), texts.to_vec(), dimensions));
            Ok(texts
                .iter()
                .enumerate()
                .map(|(index, _)| vec![index as f32, 1.0])
                .collect())
        }
    }

    #[tokio::test]
    async fn openai_embedding_provider_batches_and_builds_case_embeddings() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let client = FakeEmbeddingClient {
            calls: calls.clone(),
        };
        let provider = OpenAiEmbeddingProvider::with_client(client, "text-embedding-3-small")
            .with_dimensions(64)
            .with_batch_size(1);
        let cases = vec![
            EvalCase::new("case-1", "trace-1", "one"),
            EvalCase::new("case-2", "trace-2", "two"),
        ];
        let projector = DefaultClusterTextProjector::new();

        let embeddings = provider.embed_cases(&projector, &cases).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].provider, "openai");
        assert_eq!(embeddings[0].model, "text-embedding-3-small");
        assert_eq!(
            embeddings[0].projection_version,
            projector.projection_version()
        );

        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert!(calls.iter().all(|(model, _, dimensions)| {
            model == "text-embedding-3-small" && *dimensions == Some(64)
        }));
    }
}

#[cfg(all(test, feature = "cluster-label-openai"))]
mod cluster_label_tests {
    use anyhow::Result as AnyhowResult;
    use serde::de::DeserializeOwned;

    use crate::clustering::DiscoveredCluster;
    use crate::model::EvalCase;
    use crate::providers::chat::{ChatClient, ChatRequest};

    use super::*;

    #[derive(Clone)]
    struct FakeChatClient;

    #[async_trait::async_trait]
    impl ChatClient for FakeChatClient {
        async fn complete_json<T>(&self, request: ChatRequest) -> AnyhowResult<T>
        where
            T: DeserializeOwned + Send,
        {
            assert_eq!(request.model, "gpt-test");
            assert_eq!(request.response_schema.name, "trace_eval_cluster_label");
            assert!(request.system_prompt.contains("trace-derived eval cases"));
            assert!(request.user_prompt.contains("case-1"));

            let payload = ClusterLabelPayload {
                label: "Retrieval misses".to_string(),
                description: "Cases where retrieval did not provide enough source context."
                    .to_string(),
                suggested_rubric: "Check whether the answer cites relevant retrieved evidence."
                    .to_string(),
                known_failure_modes: vec!["missing evidence".to_string()],
                confidence: 0.87,
                needs_review: false,
            };

            Ok(serde_json::from_value(serde_json::to_value(payload)?)?)
        }
    }

    #[tokio::test]
    async fn openai_cluster_labeler_uses_chat_client_and_payload_schema() {
        let labeler = OpenAiClusterLabeler::with_client(FakeChatClient, "gpt-test");
        let cluster = DiscoveredCluster::new("cluster-1", 1, vec!["case-1".to_string()]);
        let cases = vec![EvalCase::new("case-1", "trace-1", "why?")];

        let label = labeler.label_cluster(&cluster, &cases).await.unwrap();

        assert_eq!(labeler.labeler_name(), "openai/gpt-test");
        assert_eq!(label.label, "Retrieval misses");
        assert_eq!(label.confidence, 0.87);
        assert!(!label.needs_review);
    }
}
