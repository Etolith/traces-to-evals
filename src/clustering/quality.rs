use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterQualityReport {
    pub cluster_count: usize,
    pub assigned_case_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_distance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub silhouette_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clusters: Vec<ClusterQuality>,
}

impl ClusterQualityReport {
    pub fn empty() -> Self {
        Self {
            cluster_count: 0,
            assigned_case_count: 0,
            mean_distance: None,
            silhouette_score: None,
            clusters: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterQuality {
    pub cluster_id: String,
    pub size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_distance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_distance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub silhouette_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub representative_case_ids: Vec<String>,
}

impl ClusterQuality {
    pub fn new(cluster_id: impl Into<String>, size: usize) -> Self {
        Self {
            cluster_id: cluster_id.into(),
            size,
            mean_distance: None,
            max_distance: None,
            silhouette_score: None,
            representative_case_ids: Vec::new(),
        }
    }
}
