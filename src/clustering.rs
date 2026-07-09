use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;
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

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterRuleMatch {
    pub cluster_id: String,
    pub confidence: f32,
}

impl ClusterRuleMatch {
    pub fn new(cluster_id: impl Into<String>, confidence: f32) -> Self {
        Self {
            cluster_id: cluster_id.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

pub trait ClusterAssignmentRule: Send + Sync {
    fn method(&self) -> &str;
    fn assign(&self, case: &EvalCase, clusters: &[EvalCluster]) -> Option<ClusterRuleMatch>;
}

pub struct FnClusterAssignmentRule<F> {
    method: String,
    assign: F,
}

impl<F> FnClusterAssignmentRule<F>
where
    F: Fn(&EvalCase, &[EvalCluster]) -> Option<ClusterRuleMatch> + Send + Sync,
{
    pub fn new(method: impl Into<String>, assign: F) -> Self {
        Self {
            method: method.into(),
            assign,
        }
    }
}

impl<F> ClusterAssignmentRule for FnClusterAssignmentRule<F>
where
    F: Fn(&EvalCase, &[EvalCluster]) -> Option<ClusterRuleMatch> + Send + Sync,
{
    fn method(&self) -> &str {
        &self.method
    }

    fn assign(&self, case: &EvalCase, clusters: &[EvalCluster]) -> Option<ClusterRuleMatch> {
        (self.assign)(case, clusters)
    }
}

pub trait ClusterAssigner {
    fn assign_case(&self, case: &EvalCase) -> Result<ClusterAssignment>;

    fn assign_cases(&self, cases: &[EvalCase]) -> Result<Vec<ClusterAssignment>> {
        cases.iter().map(|case| self.assign_case(case)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct MetadataAssignmentRule {
    metadata_keys: Vec<String>,
    tag_array_keys: Vec<String>,
}

impl Default for MetadataAssignmentRule {
    fn default() -> Self {
        Self {
            metadata_keys: vec![
                "cluster_id".to_string(),
                "cluster".to_string(),
                "task_cluster".to_string(),
            ],
            tag_array_keys: vec!["tags".to_string()],
        }
    }
}

impl MetadataAssignmentRule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.metadata_keys.push(key.into());
        self
    }

    pub fn with_tag_array_key(mut self, key: impl Into<String>) -> Self {
        self.tag_array_keys.push(key.into());
        self
    }
}

impl ClusterAssignmentRule for MetadataAssignmentRule {
    fn method(&self) -> &str {
        "metadata"
    }

    fn assign(&self, case: &EvalCase, clusters: &[EvalCluster]) -> Option<ClusterRuleMatch> {
        for candidate in metadata_candidates(case, &self.metadata_keys, &self.tag_array_keys) {
            for cluster in clusters {
                if cluster_matches_metadata(cluster, &candidate) {
                    return Some(ClusterRuleMatch::new(cluster.id.clone(), 1.0));
                }
            }
        }

        None
    }
}

#[derive(Debug, Clone)]
pub struct KeywordAssignmentRule {
    base_confidence: f32,
    confidence_per_extra_match: f32,
    max_confidence: f32,
}

impl Default for KeywordAssignmentRule {
    fn default() -> Self {
        Self {
            base_confidence: 0.6,
            confidence_per_extra_match: 0.1,
            max_confidence: 0.9,
        }
    }
}

impl KeywordAssignmentRule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_confidence_bounds(mut self, base: f32, per_extra_match: f32, max: f32) -> Self {
        self.base_confidence = base.clamp(0.0, 1.0);
        self.confidence_per_extra_match = per_extra_match.max(0.0);
        self.max_confidence = max.clamp(0.0, 1.0);
        self
    }
}

impl ClusterAssignmentRule for KeywordAssignmentRule {
    fn method(&self) -> &str {
        "keyword"
    }

    fn assign(&self, case: &EvalCase, clusters: &[EvalCluster]) -> Option<ClusterRuleMatch> {
        let haystack = case_text(case);
        let mut best: Option<(&EvalCluster, usize)> = None;

        for cluster in clusters {
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
            let confidence = (self.base_confidence
                + ((matches.saturating_sub(1)) as f32 * self.confidence_per_extra_match))
                .min(self.max_confidence);
            ClusterRuleMatch::new(cluster.id.clone(), confidence)
        })
    }
}

#[derive(Clone)]
pub struct RuleBasedClusterAssigner {
    clusters: Vec<EvalCluster>,
    rules: Vec<Arc<dyn ClusterAssignmentRule>>,
    fallback_cluster_id: String,
}

impl RuleBasedClusterAssigner {
    pub fn new(clusters: Vec<EvalCluster>) -> Self {
        Self::empty(clusters)
            .with_rule(MetadataAssignmentRule::default())
            .with_rule(KeywordAssignmentRule::default())
    }

    pub fn empty(clusters: Vec<EvalCluster>) -> Self {
        Self {
            clusters,
            rules: Vec::new(),
            fallback_cluster_id: UNCLUSTERED.to_string(),
        }
    }

    pub fn clusters(&self) -> &[EvalCluster] {
        &self.clusters
    }

    pub fn with_rule<R>(mut self, rule: R) -> Self
    where
        R: ClusterAssignmentRule + 'static,
    {
        self.rules.push(Arc::new(rule));
        self
    }

    pub fn with_shared_rule(mut self, rule: Arc<dyn ClusterAssignmentRule>) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_fallback_cluster_id(mut self, cluster_id: impl Into<String>) -> Self {
        self.fallback_cluster_id = cluster_id.into();
        self
    }

    pub fn rules_len(&self) -> usize {
        self.rules.len()
    }
}

impl fmt::Debug for RuleBasedClusterAssigner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuleBasedClusterAssigner")
            .field("clusters", &self.clusters)
            .field("rules_len", &self.rules.len())
            .field("fallback_cluster_id", &self.fallback_cluster_id)
            .finish()
    }
}

impl ClusterAssigner for RuleBasedClusterAssigner {
    fn assign_case(&self, case: &EvalCase) -> Result<ClusterAssignment> {
        for rule in &self.rules {
            if let Some(rule_match) = rule.assign(case, &self.clusters) {
                return Ok(ClusterAssignment {
                    case_id: case.id.clone(),
                    trace_id: case.trace_id.clone(),
                    cluster_id: rule_match.cluster_id,
                    confidence: rule_match.confidence,
                    method: rule.method().to_string(),
                });
            }
        }

        Ok(ClusterAssignment {
            case_id: case.id.clone(),
            trace_id: case.trace_id.clone(),
            cluster_id: self.fallback_cluster_id.clone(),
            confidence: 0.0,
            method: "fallback".to_string(),
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

fn metadata_candidates(
    case: &EvalCase,
    metadata_keys: &[String],
    tag_array_keys: &[String],
) -> Vec<String> {
    let mut candidates = Vec::new();

    for key in metadata_keys {
        if let Some(Value::String(value)) = case.metadata.get(key) {
            candidates.push(value.clone());
        }
    }

    for key in tag_array_keys {
        if let Some(Value::Array(values)) = case.metadata.get(key) {
            candidates.extend(values.iter().filter_map(Value::as_str).map(str::to_string));
        }
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

        let assignment = RuleBasedClusterAssigner::new(vec![cluster("arithmetic", "Arithmetic")])
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

    #[test]
    fn supports_custom_assignment_rules() {
        let case = EvalCase::new("case-1", "trace-1", "refund this order");
        let assigner =
            RuleBasedClusterAssigner::empty(vec![cluster("support", "Customer Support")])
                .with_rule(FnClusterAssignmentRule::new(
                    "custom_refund_rule",
                    |case, clusters| {
                        if case.input.contains("refund")
                            && clusters.iter().any(|cluster| cluster.id == "support")
                        {
                            Some(ClusterRuleMatch::new("support", 0.95))
                        } else {
                            None
                        }
                    },
                ));

        let assignment = assigner.assign_case(&case).unwrap();

        assert_eq!(assignment.cluster_id, "support");
        assert_eq!(assignment.confidence, 0.95);
        assert_eq!(assignment.method, "custom_refund_rule");
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
