use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
#[cfg(feature = "cluster-label-openai")]
use serde_json::Value as JsonValue;
use serde_json::Value;

use crate::Result;
use crate::TraceEvalError;
use crate::model::EvalCase;
#[cfg(feature = "cluster-label-openai")]
use crate::providers::chat::ResponseSchema;

use super::discovery::{ClusterModel, DiscoveredCluster};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterLabel {
    pub label: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_rubric: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub known_failure_modes: Vec<String>,
    pub confidence: f32,
    pub needs_review: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl ClusterLabel {
    pub fn new(label: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: description.into(),
            suggested_rubric: None,
            known_failure_modes: Vec::new(),
            confidence: 0.0,
            needs_review: true,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterLabelPrompt {
    pub system: String,
    pub user: String,
}

impl ClusterLabelPrompt {
    pub fn build(cluster: &DiscoveredCluster, examples: &[EvalCase]) -> Self {
        let request = ClusterLabelPromptRequest::from_cluster(cluster, examples);
        let request_json = serde_json::to_string_pretty(&request)
            .expect("cluster label prompt request should serialize");

        Self {
            system: concat!(
                "You label clusters of trace-derived eval cases. ",
                "Return concise JSON that names the shared behavior, explains why the examples belong together, ",
                "and flags whether a human should review the label. ",
                "Do not include hidden reasoning or unsupported claims."
            )
            .to_string(),
            user: format!(
                "Label this discovered evaluation-case cluster.\n\n\
                 Requirements:\n\
                 - label: short noun phrase, stable enough for reports\n\
                 - description: one or two sentences grounded in the examples\n\
                 - suggested_rubric: empty string if there is no useful rubric suggestion\n\
                 - known_failure_modes: concrete failure patterns seen or likely for this cluster\n\
                 - confidence: number from 0.0 to 1.0\n\
                 - needs_review: true when the examples are sparse, mixed, or ambiguous\n\n\
                 Cluster input:\n{request_json}"
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ClusterLabelPromptRequest {
    cluster: ClusterLabelPromptCluster,
    representative_cases: Vec<ClusterLabelPromptCase>,
}

impl ClusterLabelPromptRequest {
    fn from_cluster(cluster: &DiscoveredCluster, examples: &[EvalCase]) -> Self {
        Self {
            cluster: ClusterLabelPromptCluster::from_cluster(cluster),
            representative_cases: examples
                .iter()
                .map(ClusterLabelPromptCase::from_case)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ClusterLabelPromptCluster {
    id: String,
    size: usize,
    representative_case_ids: Vec<String>,
    radius: Option<f32>,
    mean_distance: Option<f32>,
    silhouette_score: Option<f32>,
}

impl ClusterLabelPromptCluster {
    fn from_cluster(cluster: &DiscoveredCluster) -> Self {
        Self {
            id: cluster.id.clone(),
            size: cluster.size,
            representative_case_ids: cluster.representative_case_ids.clone(),
            radius: cluster.radius,
            mean_distance: cluster.mean_distance,
            silhouette_score: cluster.quality.silhouette_score,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ClusterLabelPromptCase {
    id: String,
    trace_id: String,
    input: String,
    actual_output: Option<String>,
    expected_output: Option<String>,
    rubric: Option<String>,
    metadata: BTreeMap<String, Value>,
}

impl ClusterLabelPromptCase {
    fn from_case(case: &EvalCase) -> Self {
        Self {
            id: case.id.clone(),
            trace_id: case.trace_id.clone(),
            input: truncate_prompt_text(&case.input),
            actual_output: case.actual_output.as_deref().map(truncate_prompt_text),
            expected_output: case.expected_output.as_deref().map(truncate_prompt_text),
            rubric: case.rubric.as_deref().map(truncate_prompt_text),
            metadata: case.metadata.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "cluster-label-openai", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ClusterLabelPayload {
    pub label: String,
    pub description: String,
    pub suggested_rubric: String,
    pub known_failure_modes: Vec<String>,
    pub confidence: f32,
    pub needs_review: bool,
}

impl ClusterLabelPayload {
    pub fn into_label(
        self,
        provider: impl Into<String>,
        cluster_id: impl Into<String>,
    ) -> Result<ClusterLabel> {
        let provider = provider.into();
        let cluster_id = cluster_id.into();
        let label = self.label.trim();
        let description = self.description.trim();

        if label.is_empty() {
            return Err(labeling_error(
                provider,
                cluster_id,
                "label cannot be empty",
            ));
        }
        if label.chars().count() > 80 {
            return Err(labeling_error(
                provider,
                cluster_id,
                "label cannot exceed 80 characters",
            ));
        }
        if description.is_empty() {
            return Err(labeling_error(
                provider,
                cluster_id,
                "description cannot be empty",
            ));
        }
        if description.chars().count() > 600 {
            return Err(labeling_error(
                provider,
                cluster_id,
                "description cannot exceed 600 characters",
            ));
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(labeling_error(
                provider,
                cluster_id,
                format!(
                    "confidence must be between 0.0 and 1.0, got {}",
                    self.confidence
                ),
            ));
        }

        let suggested_rubric = empty_string_as_none(self.suggested_rubric);
        let mut known_failure_modes = Vec::new();
        for mode in self.known_failure_modes {
            let Some(mode) = empty_string_as_none(mode) else {
                return Err(labeling_error(
                    provider,
                    cluster_id,
                    "known_failure_modes entries cannot be empty",
                ));
            };
            known_failure_modes.push(mode);
        }

        Ok(ClusterLabel {
            label: label.to_string(),
            description: description.to_string(),
            suggested_rubric,
            known_failure_modes,
            confidence: self.confidence,
            needs_review: self.needs_review,
            metadata: BTreeMap::new(),
        })
    }
}

#[cfg(feature = "cluster-label-openai")]
impl ClusterLabelPayload {
    pub fn response_schema() -> anyhow::Result<ResponseSchema> {
        let mut response_schema = ResponseSchema::strict_json::<Self>(
            "trace_eval_cluster_label",
            "Label and review metadata for one discovered evaluation-case cluster.",
        )?;
        Self::constrain_confidence(&mut response_schema.schema);
        Ok(response_schema)
    }

    fn constrain_confidence(schema: &mut JsonValue) {
        if let Some(confidence) = schema
            .pointer_mut("/properties/confidence")
            .and_then(JsonValue::as_object_mut)
        {
            confidence.insert("minimum".to_string(), JsonValue::from(0.0));
            confidence.insert("maximum".to_string(), JsonValue::from(1.0));
        }
    }
}

#[async_trait::async_trait]
pub trait ClusterLabeler: Send + Sync {
    fn labeler_name(&self) -> String;

    async fn label_cluster(
        &self,
        cluster: &DiscoveredCluster,
        examples: &[EvalCase],
    ) -> Result<ClusterLabel>;

    async fn label_model(
        &self,
        mut model: ClusterModel,
        cases: &[EvalCase],
    ) -> Result<ClusterModel> {
        for cluster in &mut model.clusters {
            let representative_ids = cluster
                .representative_case_ids
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            let examples = cases
                .iter()
                .filter(|case| representative_ids.contains(case.id.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            cluster.label = Some(self.label_cluster(cluster, &examples).await?);
        }

        Ok(model)
    }
}

fn empty_string_as_none(value: String) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn truncate_prompt_text(text: &str) -> String {
    const MAX_CHARS: usize = 2_000;

    let mut output = String::new();
    let mut chars = text.chars();

    for _ in 0..MAX_CHARS {
        let Some(ch) = chars.next() else {
            return output;
        };
        output.push(ch);
    }

    if chars.next().is_some() {
        output.push_str("...");
    }

    output
}

fn labeling_error(
    provider: impl Into<String>,
    cluster_id: impl Into<String>,
    message: impl Into<String>,
) -> TraceEvalError {
    TraceEvalError::ClusterLabeling {
        provider: provider.into(),
        cluster_id: cluster_id.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_label_payload_trims_and_validates_output() {
        let payload = ClusterLabelPayload {
            label: " Retrieval misses ".to_string(),
            description: " Cases where retrieval did not surface source material. ".to_string(),
            suggested_rubric: " ".to_string(),
            known_failure_modes: vec![" missing citations ".to_string()],
            confidence: 0.8,
            needs_review: false,
        };

        let label = payload.into_label("test", "cluster-1").unwrap();

        assert_eq!(label.label, "Retrieval misses");
        assert_eq!(
            label.description,
            "Cases where retrieval did not surface source material."
        );
        assert_eq!(label.suggested_rubric, None);
        assert_eq!(label.known_failure_modes, vec!["missing citations"]);
        assert_eq!(label.confidence, 0.8);
        assert!(!label.needs_review);
    }

    #[test]
    fn cluster_label_payload_rejects_invalid_confidence() {
        let payload = ClusterLabelPayload {
            label: "label".to_string(),
            description: "description".to_string(),
            suggested_rubric: String::new(),
            known_failure_modes: Vec::new(),
            confidence: f32::NAN,
            needs_review: true,
        };

        assert!(payload.into_label("test", "cluster-1").is_err());
    }

    #[test]
    fn cluster_label_payload_rejects_empty_failure_modes() {
        let payload = ClusterLabelPayload {
            label: "label".to_string(),
            description: "description".to_string(),
            suggested_rubric: String::new(),
            known_failure_modes: vec![" ".to_string()],
            confidence: 0.5,
            needs_review: true,
        };

        assert!(payload.into_label("test", "cluster-1").is_err());
    }

    #[test]
    fn cluster_label_prompt_includes_representative_case_content() {
        let cluster = DiscoveredCluster::new("cluster-1", 1, vec!["case-1".to_string()]);
        let cases =
            vec![EvalCase::new("case-1", "trace-1", "question").with_actual_output("answer")];

        let prompt = ClusterLabelPrompt::build(&cluster, &cases);

        assert!(prompt.system.contains("trace-derived eval cases"));
        assert!(prompt.user.contains("cluster-1"));
        assert!(prompt.user.contains("question"));
        assert!(prompt.user.contains("answer"));
    }

    #[cfg(feature = "cluster-label-openai")]
    #[test]
    fn cluster_label_response_schema_is_strict_and_constrained() {
        let schema = ClusterLabelPayload::response_schema().unwrap().schema;

        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["confidence"]["minimum"], 0.0);
        assert_eq!(schema["properties"]["confidence"]["maximum"], 1.0);
        assert!(schema.get("$schema").is_none());
    }
}
