use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::evaluation::EvaluationResult;
use crate::model::EvalCase;

pub const UNCLUSTERED: &str = "unclustered";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalCluster {
    pub id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "default_weight")]
    pub weight: f32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterAssignment {
    pub case_id: String,
    pub trace_id: String,
    pub cluster_id: String,
    pub confidence: f32,
    pub method: String,
}

pub trait Clusterer {
    fn assign_case(&self, case: &EvalCase) -> Result<ClusterAssignment>;

    fn assign_cases(&self, cases: &[EvalCase]) -> Result<Vec<ClusterAssignment>> {
        cases.iter().map(|case| self.assign_case(case)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct MetadataClusterer {
    clusters: Vec<EvalCluster>,
}

impl MetadataClusterer {
    pub fn new(clusters: Vec<EvalCluster>) -> Self {
        Self { clusters }
    }

    pub fn clusters(&self) -> &[EvalCluster] {
        &self.clusters
    }
}

impl Clusterer for MetadataClusterer {
    fn assign_case(&self, case: &EvalCase) -> Result<ClusterAssignment> {
        if let Some(cluster_id) = self.metadata_cluster_id(case) {
            return Ok(ClusterAssignment {
                case_id: case.id.clone(),
                trace_id: case.trace_id.clone(),
                cluster_id,
                confidence: 1.0,
                method: "metadata".to_string(),
            });
        }

        if let Some((cluster_id, confidence)) = self.lexical_cluster_id(case) {
            return Ok(ClusterAssignment {
                case_id: case.id.clone(),
                trace_id: case.trace_id.clone(),
                cluster_id,
                confidence,
                method: "lexical".to_string(),
            });
        }

        Ok(ClusterAssignment {
            case_id: case.id.clone(),
            trace_id: case.trace_id.clone(),
            cluster_id: UNCLUSTERED.to_string(),
            confidence: 0.0,
            method: "fallback".to_string(),
        })
    }
}

impl MetadataClusterer {
    fn metadata_cluster_id(&self, case: &EvalCase) -> Option<String> {
        for candidate in metadata_candidates(case) {
            for cluster in &self.clusters {
                if cluster_matches_metadata(cluster, &candidate) {
                    return Some(cluster.id.clone());
                }
            }
        }

        None
    }

    fn lexical_cluster_id(&self, case: &EvalCase) -> Option<(String, f32)> {
        let haystack = case_text(case);
        let mut best: Option<(&EvalCluster, usize)> = None;

        for cluster in &self.clusters {
            let matches = cluster
                .keywords()
                .iter()
                .filter(|keyword| haystack.contains(&keyword.to_ascii_lowercase()))
                .count();

            if matches == 0 {
                continue;
            }

            if best.is_none_or(|(_, best_matches)| matches > best_matches) {
                best = Some((cluster, matches));
            }
        }

        best.map(|(cluster, matches)| {
            let confidence = (0.6 + ((matches.saturating_sub(1)) as f32 * 0.1)).min(0.9);
            (cluster.id.clone(), confidence)
        })
    }
}

impl EvalCluster {
    fn keywords(&self) -> Vec<String> {
        let mut keywords = Vec::new();

        if !self.label.trim().is_empty() {
            keywords.push(self.label.to_ascii_lowercase());
        }

        if let Some(Value::String(tag)) = self.metadata.get("tag") {
            keywords.push(tag.to_ascii_lowercase());
        }

        if let Some(Value::Array(values)) = self.metadata.get("keywords") {
            keywords.extend(values.iter().filter_map(Value::as_str).map(str::to_string));
        }

        keywords
            .into_iter()
            .map(|keyword| keyword.trim().to_ascii_lowercase())
            .filter(|keyword| !keyword.is_empty())
            .collect()
    }
}

pub fn apply_assignments_to_results(
    mut results: Vec<EvaluationResult>,
    assignments: &[ClusterAssignment],
) -> Vec<EvaluationResult> {
    let assignment_by_case = assignments
        .iter()
        .map(|assignment| (assignment.case_id.as_str(), assignment))
        .collect::<BTreeMap<_, _>>();

    for result in &mut results {
        if let Some(assignment) = assignment_by_case.get(result.case_id.as_str()) {
            result.cluster_id = Some(assignment.cluster_id.clone());
            result.metadata.insert(
                "cluster_method".to_string(),
                Value::String(assignment.method.clone()),
            );
            result.metadata.insert(
                "cluster_confidence".to_string(),
                Value::from(assignment.confidence),
            );
        }
    }

    results
}

fn metadata_candidates(case: &EvalCase) -> Vec<String> {
    let mut candidates = Vec::new();

    for key in ["cluster_id", "cluster", "task_cluster"] {
        if let Some(Value::String(value)) = case.metadata.get(key) {
            candidates.push(value.clone());
        }
    }

    if let Some(Value::Array(values)) = case.metadata.get("tags") {
        candidates.extend(values.iter().filter_map(Value::as_str).map(str::to_string));
    }

    candidates
}

fn cluster_matches_metadata(cluster: &EvalCluster, candidate: &str) -> bool {
    let candidate = candidate.trim();

    cluster.id == candidate
        || cluster.label == candidate
        || cluster
            .metadata
            .get("tag")
            .and_then(Value::as_str)
            .is_some_and(|tag| tag == candidate)
}

fn case_text(case: &EvalCase) -> String {
    [
        Some(case.input.as_str()),
        case.actual_output.as_deref(),
        case.expected_output.as_deref(),
        case.rubric.as_deref(),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("\n")
    .to_ascii_lowercase()
}

fn default_weight() -> f32 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::EvalCase;

    #[test]
    fn assigns_cluster_from_metadata() {
        let mut case = EvalCase::new("case-1", "trace-1", "input");
        case.metadata.insert(
            "cluster_id".to_string(),
            Value::String("arithmetic".to_string()),
        );

        let assignment = MetadataClusterer::new(vec![cluster("arithmetic", "Arithmetic")])
            .assign_case(&case)
            .unwrap();

        assert_eq!(assignment.cluster_id, "arithmetic");
        assert_eq!(assignment.method, "metadata");
    }

    #[test]
    fn applies_assignments_to_results() {
        let case = EvalCase::new("case-1", "trace-1", "input");
        let result = EvaluationResult::binary(&case, "non_empty", true, "ok");
        let assignment = ClusterAssignment {
            case_id: "case-1".to_string(),
            trace_id: "trace-1".to_string(),
            cluster_id: "arithmetic".to_string(),
            confidence: 1.0,
            method: "metadata".to_string(),
        };

        let results = apply_assignments_to_results(vec![result], &[assignment]);

        assert_eq!(results[0].cluster_id.as_deref(), Some("arithmetic"));
    }

    fn cluster(id: &str, label: &str) -> EvalCluster {
        EvalCluster {
            id: id.to_string(),
            label: label.to_string(),
            description: None,
            weight: 1.0,
            metadata: BTreeMap::new(),
        }
    }
}
