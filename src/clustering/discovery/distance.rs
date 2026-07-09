use serde::{Deserialize, Serialize};

use crate::{Result, TraceEvalError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
}

impl DistanceMetric {
    pub fn distance(self, left: &[f32], right: &[f32]) -> Result<f32> {
        if left.len() != right.len() {
            return Err(TraceEvalError::ClusterAssignment {
                case_id: "<embedding>".to_string(),
                message: format!(
                    "embedding dimensions differ: {} vs {}",
                    left.len(),
                    right.len()
                ),
            });
        }

        match self {
            Self::Cosine => cosine_distance(left, right),
            Self::Euclidean => euclidean_distance(left, right),
        }
    }

    pub(crate) fn confidence(self, distance: f32) -> f32 {
        match self {
            Self::Cosine => (1.0 - (distance / 2.0)).clamp(0.0, 1.0),
            Self::Euclidean => (1.0 - (distance / (1.0 + distance))).clamp(0.0, 1.0),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::Euclidean => "euclidean",
        }
    }
}

fn cosine_distance(left: &[f32], right: &[f32]) -> Result<f32> {
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;

    for (left, right) in left.iter().zip(right) {
        if !left.is_finite() || !right.is_finite() {
            return Err(TraceEvalError::ClusterAssignment {
                case_id: "<embedding>".to_string(),
                message: "embedding contains non-finite value".to_string(),
            });
        }
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        return Ok(1.0);
    }

    Ok(1.0 - (dot / (left_norm.sqrt() * right_norm.sqrt())))
}

fn euclidean_distance(left: &[f32], right: &[f32]) -> Result<f32> {
    let mut sum = 0.0f32;

    for (left, right) in left.iter().zip(right) {
        if !left.is_finite() || !right.is_finite() {
            return Err(TraceEvalError::ClusterAssignment {
                case_id: "<embedding>".to_string(),
                message: "embedding contains non-finite value".to_string(),
            });
        }
        let delta = left - right;
        sum += delta * delta;
    }

    Ok(sum.sqrt())
}
