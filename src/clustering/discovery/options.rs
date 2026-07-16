use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;
use crate::calibration::HumanRating;
use crate::clustering::embedding::CaseEmbedding;
use crate::evaluation::EvaluationResult;
use crate::model::EvalCase;
use crate::project::ProjectName;

use super::{ClusterModel, DistanceMetric};

#[derive(Debug, Clone, Copy)]
pub struct ClusterDiscoveryInput<'a> {
    pub cases: &'a [EvalCase],
    pub embeddings: Option<&'a [CaseEmbedding]>,
    pub human_ratings: Option<&'a [HumanRating]>,
    pub previous_results: Option<&'a [EvaluationResult]>,
    pub options: &'a ClusterDiscoveryOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterDiscoveryOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default)]
    pub project_name: ProjectName,
    pub algorithm: ClusterAlgorithm,
    pub distance_metric: DistanceMetric,
    pub representative_count: usize,
    pub random_seed: u64,
    #[serde(default)]
    pub quality_evaluation: ClusterQualityEvaluation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub novelty_distance_threshold: Option<f32>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl Default for ClusterDiscoveryOptions {
    fn default() -> Self {
        Self {
            model_id: None,
            project_name: ProjectName::default(),
            algorithm: ClusterAlgorithm::KMeans {
                k: 2,
                max_iterations: 100,
                tolerance: 0.0001,
            },
            distance_metric: DistanceMetric::Cosine,
            representative_count: 5,
            random_seed: 42,
            quality_evaluation: ClusterQualityEvaluation::default(),
            novelty_distance_threshold: None,
            metadata: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterQualityEvaluation {
    Disabled,
    Sampled { maximum_cases: usize },
    Exact,
}

impl Default for ClusterQualityEvaluation {
    fn default() -> Self {
        Self::Sampled {
            maximum_cases: 1_000,
        }
    }
}

impl ClusterQualityEvaluation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Sampled { .. } => "sampled",
            Self::Exact => "exact",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterAlgorithm {
    KMeans {
        k: usize,
        max_iterations: usize,
        tolerance: f32,
    },
    Dbscan {
        min_points: usize,
        epsilon: f32,
    },
}

impl ClusterAlgorithm {
    pub fn name(self) -> &'static str {
        match self {
            Self::KMeans { .. } => "kmeans",
            Self::Dbscan { .. } => "dbscan",
        }
    }
}

pub trait ClusterDiscovery {
    fn algorithm_name(&self) -> &'static str;
    fn fit(&self, input: ClusterDiscoveryInput<'_>) -> Result<ClusterModel>;
}
