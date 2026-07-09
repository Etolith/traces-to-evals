use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;
use crate::model::EvalCase;

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
