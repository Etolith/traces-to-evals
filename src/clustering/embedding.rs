use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::model::EvalCase;
use crate::project::ProjectName;
use crate::{Result, TraceEvalError};

pub const CASE_EMBEDDING_SCHEMA_KIND: &str = "case_embedding";
pub const CLUSTER_TEXT_PROJECTION_KIND: &str = "cluster_text";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectedField {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterText {
    pub case_id: String,
    pub trace_id: String,
    pub text: String,
    pub fields: Vec<ProjectedField>,
}

pub trait ClusterTextProjector {
    fn projection_version(&self) -> String;
    fn project_case(&self, case: &EvalCase) -> ClusterText;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultClusterTextProjector {
    include_actual_output: bool,
    metadata_keys: Vec<&'static str>,
    project_name: ProjectName,
}

impl Default for DefaultClusterTextProjector {
    fn default() -> Self {
        Self {
            include_actual_output: false,
            project_name: ProjectName::default(),
            metadata_keys: vec![
                "route",
                "task",
                "task_id",
                "scenario",
                "tool",
                "tool_name",
                "product_area",
                "cluster_id",
                "tags",
            ],
        }
    }
}

impl DefaultClusterTextProjector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn include_actual_output(mut self) -> Self {
        self.include_actual_output = true;
        self
    }

    pub fn with_project_name(mut self, project_name: ProjectName) -> Self {
        self.project_name = project_name;
        self
    }

    pub fn try_with_project_name(mut self, project_name: impl Into<String>) -> Result<Self> {
        self.project_name = ProjectName::new(project_name)?;
        Ok(self)
    }

    pub fn with_metadata_key(mut self, key: &'static str) -> Self {
        if !self.metadata_keys.contains(&key) {
            self.metadata_keys.push(key);
        }
        self
    }
}

impl ClusterTextProjector for DefaultClusterTextProjector {
    fn projection_version(&self) -> String {
        self.project_name
            .cluster_text_projection_version(self.include_actual_output)
    }

    fn project_case(&self, case: &EvalCase) -> ClusterText {
        let mut fields = Vec::new();

        push_field(&mut fields, "input", case.input.as_str());
        if let Some(rubric) = case.rubric.as_deref() {
            push_field(&mut fields, "rubric", rubric);
        }
        if let Some(expected_output) = case.expected_output.as_deref() {
            push_field(&mut fields, "expected_output", expected_output);
        }
        if self.include_actual_output
            && let Some(actual_output) = case.actual_output.as_deref()
        {
            push_field(&mut fields, "actual_output", actual_output);
        }

        for key in &self.metadata_keys {
            if let Some(value) = case.metadata.get(*key).and_then(metadata_value_text) {
                push_field(&mut fields, *key, value.as_str());
            }
        }

        ClusterText {
            case_id: case.id.clone(),
            trace_id: case.trace_id.clone(),
            text: render_fields(&fields),
            fields,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseEmbedding {
    pub schema_version: String,
    pub case_id: String,
    pub trace_id: String,
    pub provider: String,
    pub model: String,
    pub dimensions: usize,
    pub vector: Vec<f32>,
    pub projection_version: String,
    pub text_hash: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl CaseEmbedding {
    pub fn new(
        projected: &ClusterText,
        provider: impl Into<String>,
        model: impl Into<String>,
        vector: Vec<f32>,
        projection_version: impl Into<String>,
    ) -> Self {
        Self::new_with_project(
            &ProjectName::default(),
            projected,
            provider,
            model,
            vector,
            projection_version,
        )
    }

    pub fn new_with_project(
        project_name: &ProjectName,
        projected: &ClusterText,
        provider: impl Into<String>,
        model: impl Into<String>,
        vector: Vec<f32>,
        projection_version: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: project_name.case_embedding_schema_version(),
            case_id: projected.case_id.clone(),
            trace_id: projected.trace_id.clone(),
            provider: provider.into(),
            model: model.into(),
            dimensions: vector.len(),
            vector,
            projection_version: projection_version.into(),
            text_hash: hash_projected_text(projected.text.as_str()),
            metadata: BTreeMap::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !ProjectName::matches_schema_version(&self.schema_version, CASE_EMBEDDING_SCHEMA_KIND, 1)
        {
            return Err(TraceEvalError::InvalidEmbedding {
                case_id: self.case_id.clone(),
                message: format!("unsupported schema_version {}", self.schema_version),
            });
        }

        if self.dimensions != self.vector.len() {
            return Err(TraceEvalError::InvalidEmbedding {
                case_id: self.case_id.clone(),
                message: format!(
                    "dimensions {} does not match vector length {}",
                    self.dimensions,
                    self.vector.len()
                ),
            });
        }

        if self.vector.iter().any(|value| !value.is_finite()) {
            return Err(TraceEvalError::InvalidEmbedding {
                case_id: self.case_id.clone(),
                message: "vector contains non-finite value".to_string(),
            });
        }

        Ok(())
    }
}

#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn provider_name(&self) -> String;
    fn model_name(&self) -> String;

    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    async fn embed_cases<P>(&self, projector: &P, cases: &[EvalCase]) -> Result<Vec<CaseEmbedding>>
    where
        P: ClusterTextProjector + Send + Sync,
    {
        self.embed_cases_with_project(&ProjectName::default(), projector, cases)
            .await
    }

    async fn embed_cases_with_project<P>(
        &self,
        project_name: &ProjectName,
        projector: &P,
        cases: &[EvalCase],
    ) -> Result<Vec<CaseEmbedding>>
    where
        P: ClusterTextProjector + Send + Sync,
    {
        let projected = cases
            .iter()
            .map(|case| projector.project_case(case))
            .collect::<Vec<_>>();
        let texts = projected
            .iter()
            .map(|projection| projection.text.clone())
            .collect::<Vec<_>>();
        let vectors = self.embed_texts(&texts).await?;

        if vectors.len() != projected.len() {
            return Err(TraceEvalError::EmbeddingProvider {
                provider: self.provider_name(),
                message: format!(
                    "provider returned {} embeddings for {} cases",
                    vectors.len(),
                    projected.len()
                ),
            });
        }

        projected
            .iter()
            .zip(vectors)
            .map(|(projection, vector)| {
                let embedding = CaseEmbedding::new_with_project(
                    project_name,
                    projection,
                    self.provider_name(),
                    self.model_name(),
                    vector,
                    projector.projection_version(),
                );
                embedding.validate()?;
                Ok(embedding)
            })
            .collect()
    }
}

pub fn hash_projected_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn push_field(fields: &mut Vec<ProjectedField>, name: impl Into<String>, value: &str) {
    let value = value.trim();
    if value.is_empty() {
        return;
    }

    fields.push(ProjectedField {
        name: name.into(),
        value: value.to_string(),
    });
}

fn render_fields(fields: &[ProjectedField]) -> String {
    fields
        .iter()
        .map(|field| format!("{}:\n{}", field.name, field.value))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn metadata_value_text(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        Value::Array(values) => {
            let values = values
                .iter()
                .filter_map(metadata_value_text)
                .collect::<Vec<_>>();
            (!values.is_empty()).then(|| values.join(", "))
        }
        Value::Object(_) => Some(value.to_string()),
        Value::Null => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn default_projection_is_deterministic_and_excludes_actual_output() {
        let mut case = EvalCase::new("case-1", "trace-1", "How do I reset my password?")
            .with_actual_output("bad answer")
            .with_expected_output("Use the password reset link")
            .with_rubric("Must mention the reset link");
        case.metadata.insert("route".to_string(), json!("account"));
        case.metadata
            .insert("ignored".to_string(), json!("not included"));
        case.metadata
            .insert("tags".to_string(), json!(["auth", "password"]));

        let projector = DefaultClusterTextProjector::new();
        let projected = projector.project_case(&case);
        let projected_again = projector.project_case(&case);

        assert_eq!(projector.projection_version(), "traceeval.cluster_text.v1");
        assert_eq!(projected, projected_again);
        assert!(
            projected
                .text
                .contains("input:\nHow do I reset my password?")
        );
        assert!(
            projected
                .text
                .contains("rubric:\nMust mention the reset link")
        );
        assert!(
            projected
                .text
                .contains("expected_output:\nUse the password reset link")
        );
        assert!(projected.text.contains("route:\naccount"));
        assert!(projected.text.contains("tags:\nauth, password"));
        assert!(!projected.text.contains("actual_output"));
        assert!(!projected.text.contains("bad answer"));
        assert!(!projected.text.contains("ignored"));
    }

    #[test]
    fn include_output_projection_changes_version_and_includes_actual_output() {
        let case = EvalCase::new("case-1", "trace-1", "input").with_actual_output("answer");

        let projected = DefaultClusterTextProjector::new()
            .include_actual_output()
            .project_case(&case);

        assert!(projected.text.contains("actual_output:\nanswer"));
    }

    #[test]
    fn embedding_validation_rejects_dimension_mismatch_and_non_finite_values() {
        let projected =
            DefaultClusterTextProjector::new().project_case(&EvalCase::new("case-1", "t", "i"));
        let mut embedding =
            CaseEmbedding::new(&projected, "test", "model", vec![0.1, 0.2], "projection");
        embedding.dimensions = 3;

        assert!(embedding.validate().is_err());

        let embedding = CaseEmbedding::new(&projected, "test", "model", vec![f32::NAN], "p");
        assert!(embedding.validate().is_err());
    }

    #[test]
    fn custom_project_name_scopes_projection_and_embedding_schema() {
        let project = ProjectName::new("acme-evals").unwrap();
        let projector = DefaultClusterTextProjector::new().with_project_name(project.clone());
        let projected = projector.project_case(&EvalCase::new("case-1", "trace-1", "input"));

        assert_eq!(projector.projection_version(), "acme-evals.cluster_text.v1");

        let embedding = CaseEmbedding::new_with_project(
            &project,
            &projected,
            "test",
            "model",
            vec![0.1],
            projector.projection_version(),
        );

        assert_eq!(embedding.schema_version, "acme-evals.case_embedding.v1");
        assert!(embedding.validate().is_ok());
    }
}
