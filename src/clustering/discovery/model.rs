use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::clustering::assignment::{ClusterAssignment, EvalCluster, UNCLUSTERED};
use crate::clustering::labeling::ClusterLabel;
use crate::clustering::quality::{ClusterQuality, ClusterQualityReport};
use crate::project::ProjectName;
use crate::{Result, TraceEvalError};

pub const CLUSTER_MODEL_SCHEMA_KIND: &str = "cluster_model";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterModel {
    pub schema_version: String,
    pub model_id: String,
    pub created_at: String,
    pub source: ClusterModelSource,
    pub clusters: Vec<DiscoveredCluster>,
    pub assignments: Vec<ClusterAssignment>,
    pub quality: ClusterQualityReport,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl ClusterModel {
    pub fn new(
        model_id: impl Into<String>,
        created_at: impl Into<String>,
        source: ClusterModelSource,
        clusters: Vec<DiscoveredCluster>,
        assignments: Vec<ClusterAssignment>,
        quality: ClusterQualityReport,
    ) -> Self {
        Self::new_with_project(
            &ProjectName::default(),
            model_id,
            created_at,
            source,
            clusters,
            assignments,
            quality,
        )
    }

    pub fn new_with_project(
        project_name: &ProjectName,
        model_id: impl Into<String>,
        created_at: impl Into<String>,
        source: ClusterModelSource,
        clusters: Vec<DiscoveredCluster>,
        assignments: Vec<ClusterAssignment>,
        quality: ClusterQualityReport,
    ) -> Self {
        Self {
            schema_version: project_name.cluster_model_schema_version(),
            model_id: model_id.into(),
            created_at: created_at.into(),
            source,
            clusters,
            assignments,
            quality,
            metadata: BTreeMap::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !ProjectName::matches_schema_version(&self.schema_version, CLUSTER_MODEL_SCHEMA_KIND, 1)
        {
            return Err(self.validation_error(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }

        if self.clusters.is_empty() {
            return Err(self.validation_error("model has no clusters"));
        }

        let cluster_ids = self
            .clusters
            .iter()
            .map(|cluster| cluster.id.as_str())
            .collect::<BTreeSet<_>>();

        for cluster in &self.clusters {
            if cluster.size == 0 {
                return Err(self.validation_error(format!("cluster {} has size 0", cluster.id)));
            }
            if cluster.representative_case_ids.is_empty() {
                return Err(self.validation_error(format!(
                    "cluster {} has no representative cases",
                    cluster.id
                )));
            }
        }

        for assignment in &self.assignments {
            if assignment.cluster_id != UNCLUSTERED
                && !cluster_ids.contains(assignment.cluster_id.as_str())
            {
                return Err(self.validation_error(format!(
                    "assignment for case {} references unknown cluster {}",
                    assignment.case_id, assignment.cluster_id
                )));
            }
        }

        Ok(())
    }

    pub fn to_eval_clusters(&self) -> Vec<EvalCluster> {
        self.clusters
            .iter()
            .map(|cluster| {
                let mut metadata = cluster.metadata.clone();
                metadata.insert(
                    "source_model_id".to_string(),
                    Value::String(self.model_id.clone()),
                );
                metadata.insert("cluster_size".to_string(), Value::from(cluster.size));
                if let Some(radius) = cluster.radius {
                    metadata.insert("cluster_radius".to_string(), Value::from(radius));
                }

                EvalCluster {
                    id: cluster.id.clone(),
                    label: cluster
                        .label
                        .as_ref()
                        .map(|label| label.label.clone())
                        .unwrap_or_else(|| cluster.id.clone()),
                    description: cluster
                        .label
                        .as_ref()
                        .map(|label| label.description.clone()),
                    weight: 1.0,
                    metadata,
                }
            })
            .collect()
    }

    fn validation_error(&self, message: impl Into<String>) -> TraceEvalError {
        TraceEvalError::ClusterModelValidation {
            model_id: self.model_id.clone(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterModelSource {
    pub case_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_dimensions: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_version: Option<String>,
    pub algorithm: String,
    pub distance_metric: String,
    pub random_seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredCluster {
    pub id: String,
    pub size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub centroid: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub representative_case_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_distance: Option<f32>,
    pub quality: ClusterQuality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<ClusterLabel>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl DiscoveredCluster {
    pub fn new(id: impl Into<String>, size: usize, representative_case_ids: Vec<String>) -> Self {
        let id = id.into();
        Self {
            quality: ClusterQuality {
                representative_case_ids: representative_case_ids.clone(),
                ..ClusterQuality::new(id.clone(), size)
            },
            id,
            size,
            centroid: None,
            representative_case_ids,
            radius: None,
            mean_distance: None,
            label: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_centroid(mut self, centroid: Vec<f32>) -> Self {
        self.centroid = Some(centroid);
        self
    }
}
