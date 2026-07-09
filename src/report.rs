use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::clustering::{EvalCluster, UNCLUSTERED};
use crate::evaluation::{EvaluationResult, RunScore, WeightedAggregate};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub run_score: RunScore,
    pub evaluator_scores: Vec<EvaluatorScore>,
    pub cluster_scores: Vec<ClusterScore>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed_cases: Vec<FailedCase>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub worst_clusters: Vec<ClusterIssue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_impact: Option<CalibrationImpact>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub novelty_count: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub unclustered_count: usize,
    pub total_cases: usize,
    pub total_results: usize,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluatorScore {
    pub evaluator_name: String,
    pub score: RunScore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterScore {
    pub cluster_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub score: RunScore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailedCase {
    pub case_id: String,
    pub trace_id: String,
    pub evaluator_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_confidence: Option<f32>,
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibrated_score: Option<f32>,
    pub evaluation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationImpact {
    pub uncalibrated_score: f32,
    pub calibrated_score: f32,
    pub delta: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterIssue {
    pub cluster_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub score: RunScore,
    pub failed_cases: Vec<FailedCase>,
}

impl EvaluationReport {
    pub fn from_results(results: &[EvaluationResult]) -> Self {
        Self::from_results_with_aggregate(results, &WeightedAggregate::default())
    }

    pub fn from_results_with_aggregate(
        results: &[EvaluationResult],
        aggregate: &WeightedAggregate,
    ) -> Self {
        Self::from_results_with_aggregate_and_clusters(results, aggregate, &[])
    }

    pub fn from_results_with_clusters(
        results: &[EvaluationResult],
        clusters: &[EvalCluster],
    ) -> Self {
        let aggregate = clusters
            .iter()
            .fold(WeightedAggregate::new(), |aggregate, cluster| {
                aggregate.with_cluster_weight(cluster.id.clone(), cluster.weight)
            });

        Self::from_results_with_aggregate_and_clusters(results, &aggregate, clusters)
    }

    pub fn from_results_with_aggregate_and_clusters(
        results: &[EvaluationResult],
        aggregate: &WeightedAggregate,
        clusters: &[EvalCluster],
    ) -> Self {
        let case_ids = results
            .iter()
            .map(|result| result.case_id.as_str())
            .collect::<BTreeSet<_>>();
        let cluster_lookup = ClusterLookup::new(clusters);

        Self {
            run_score: aggregate.score(results),
            evaluator_scores: evaluator_scores(results, aggregate),
            cluster_scores: cluster_scores(results, aggregate, &cluster_lookup),
            failed_cases: failed_cases(results, &cluster_lookup),
            worst_clusters: worst_clusters(results, aggregate, &cluster_lookup),
            calibration_impact: calibration_impact(results, aggregate),
            novelty_count: novelty_count(results),
            unclustered_count: unclustered_count(results),
            total_cases: case_ids.len(),
            total_results: results.len(),
            metadata: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Default)]
struct ClusterLookup {
    clusters: BTreeMap<String, ClusterInfo>,
}

impl ClusterLookup {
    fn new(clusters: &[EvalCluster]) -> Self {
        Self {
            clusters: clusters
                .iter()
                .map(|cluster| {
                    (
                        cluster.id.clone(),
                        ClusterInfo {
                            label: Some(cluster.label.clone()),
                            description: cluster.description.clone(),
                        },
                    )
                })
                .collect(),
        }
    }

    fn label(&self, cluster_id: &str) -> Option<String> {
        self.clusters
            .get(cluster_id)
            .and_then(|cluster| cluster.label.clone())
    }

    fn description(&self, cluster_id: &str) -> Option<String> {
        self.clusters
            .get(cluster_id)
            .and_then(|cluster| cluster.description.clone())
    }
}

#[derive(Debug, Clone)]
struct ClusterInfo {
    label: Option<String>,
    description: Option<String>,
}

fn failed_cases(results: &[EvaluationResult], cluster_lookup: &ClusterLookup) -> Vec<FailedCase> {
    let mut failed = results
        .iter()
        .filter(|result| !result.passed)
        .map(|result| failed_case(result, cluster_lookup))
        .collect::<Vec<_>>();

    sort_failed_cases(&mut failed);
    failed
}

fn failed_case(result: &EvaluationResult, cluster_lookup: &ClusterLookup) -> FailedCase {
    FailedCase {
        case_id: result.case_id.clone(),
        trace_id: result.trace_id.clone(),
        evaluator_name: result.evaluator_name.clone(),
        cluster_id: result.cluster_id.clone(),
        cluster_label: result
            .cluster_id
            .as_deref()
            .and_then(|cluster_id| cluster_lookup.label(cluster_id)),
        cluster_confidence: cluster_confidence(result),
        score: result.score_for_aggregation(),
        calibrated_score: result.calibrated_score,
        evaluation: result.evaluation.clone(),
    }
}

fn cluster_confidence(result: &EvaluationResult) -> Option<f32> {
    result
        .metadata
        .get("cluster_confidence")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
}

fn novelty_count(results: &[EvaluationResult]) -> usize {
    results
        .iter()
        .filter(|result| {
            result
                .metadata
                .get("cluster_novelty")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .count()
}

fn unclustered_count(results: &[EvaluationResult]) -> usize {
    results
        .iter()
        .filter(|result| {
            result
                .cluster_id
                .as_deref()
                .is_none_or(|cluster_id| cluster_id == UNCLUSTERED)
        })
        .count()
}

fn evaluator_scores(
    results: &[EvaluationResult],
    aggregate: &WeightedAggregate,
) -> Vec<EvaluatorScore> {
    group_by(results, |result| result.evaluator_name.clone())
        .into_iter()
        .map(|(evaluator_name, group)| EvaluatorScore {
            evaluator_name,
            score: aggregate.score(&group),
        })
        .collect()
}

fn cluster_scores(
    results: &[EvaluationResult],
    aggregate: &WeightedAggregate,
    cluster_lookup: &ClusterLookup,
) -> Vec<ClusterScore> {
    group_by(results, |result| {
        result
            .cluster_id
            .clone()
            .unwrap_or_else(|| UNCLUSTERED.to_string())
    })
    .into_iter()
    .map(|(cluster_id, group)| ClusterScore {
        label: cluster_lookup.label(&cluster_id),
        description: cluster_lookup.description(&cluster_id),
        cluster_id,
        score: aggregate.score(&group),
    })
    .collect()
}

fn worst_clusters(
    results: &[EvaluationResult],
    aggregate: &WeightedAggregate,
    cluster_lookup: &ClusterLookup,
) -> Vec<ClusterIssue> {
    let mut issues = group_by(results, |result| {
        result
            .cluster_id
            .clone()
            .unwrap_or_else(|| UNCLUSTERED.to_string())
    })
    .into_iter()
    .filter_map(|(cluster_id, group)| {
        let failed = failed_cases(&group, cluster_lookup);

        if failed.is_empty() {
            return None;
        }

        Some(ClusterIssue {
            label: cluster_lookup.label(&cluster_id),
            description: cluster_lookup.description(&cluster_id),
            cluster_id,
            score: aggregate.score(&group),
            failed_cases: failed,
        })
    })
    .collect::<Vec<_>>();

    issues.sort_by(|left, right| {
        left.score
            .weighted_score
            .total_cmp(&right.score.weighted_score)
            .then_with(|| left.cluster_id.cmp(&right.cluster_id))
    });
    issues
}

fn calibration_impact(
    results: &[EvaluationResult],
    aggregate: &WeightedAggregate,
) -> Option<CalibrationImpact> {
    if !results
        .iter()
        .any(|result| result.calibrated_score.is_some())
    {
        return None;
    }

    let mut uncalibrated_results = results.to_vec();
    for result in &mut uncalibrated_results {
        result.calibrated_score = None;
    }

    let uncalibrated_score = aggregate.score(&uncalibrated_results).weighted_score;
    let calibrated_score = aggregate.score(results).weighted_score;

    Some(CalibrationImpact {
        uncalibrated_score,
        calibrated_score,
        delta: calibrated_score - uncalibrated_score,
    })
}

fn group_by<F>(results: &[EvaluationResult], key_for: F) -> BTreeMap<String, Vec<EvaluationResult>>
where
    F: Fn(&EvaluationResult) -> String,
{
    let mut groups = BTreeMap::<String, Vec<EvaluationResult>>::new();

    for result in results {
        groups
            .entry(key_for(result))
            .or_default()
            .push(result.clone());
    }

    groups
}

fn sort_failed_cases(failed_cases: &mut [FailedCase]) {
    failed_cases.sort_by(|left, right| {
        left.cluster_id
            .cmp(&right.cluster_id)
            .then_with(|| left.evaluator_name.cmp(&right.evaluator_name))
            .then_with(|| left.score.total_cmp(&right.score))
            .then_with(|| left.case_id.cmp(&right.case_id))
    });
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::EvalCase;

    #[test]
    fn reports_overall_evaluator_and_cluster_scores() {
        let case = EvalCase::new("case-1", "trace-1", "input");
        let results = vec![
            EvaluationResult::binary(&case, "fast", true, "ok").with_cluster_id("a"),
            EvaluationResult::binary(&case, "slow", false, "bad"),
        ];

        let report = EvaluationReport::from_results(&results);

        assert_eq!(report.total_cases, 1);
        assert_eq!(report.total_results, 2);
        assert_eq!(report.run_score.result_count, 2);
        assert_eq!(report.evaluator_scores.len(), 2);
        assert_eq!(report.cluster_scores.len(), 2);
        assert_eq!(report.cluster_scores[1].cluster_id, UNCLUSTERED);
        assert_eq!(report.failed_cases.len(), 1);
        assert_eq!(report.failed_cases[0].case_id, "case-1");
        assert_eq!(report.worst_clusters.len(), 1);
        assert_eq!(report.worst_clusters[0].cluster_id, UNCLUSTERED);
    }

    #[test]
    fn reports_calibration_impact_when_calibrated_scores_exist() {
        let case = EvalCase::new("case-1", "trace-1", "input");
        let results =
            vec![EvaluationResult::binary(&case, "fast", true, "ok").with_calibrated_score(0.25)];

        let report = EvaluationReport::from_results(&results);
        let impact = report.calibration_impact.unwrap();

        assert_eq!(impact.uncalibrated_score, 1.0);
        assert_eq!(impact.calibrated_score, 0.25);
        assert_eq!(impact.delta, -0.75);
    }

    #[test]
    fn report_uses_cluster_labels_and_assignment_confidence() {
        let case = EvalCase::new("case-1", "trace-1", "input");
        let mut failed =
            EvaluationResult::binary(&case, "fast", false, "bad").with_cluster_id("retrieval");
        failed
            .metadata
            .insert("cluster_confidence".to_string(), Value::from(0.82));
        failed
            .metadata
            .insert("cluster_novelty".to_string(), Value::from(true));
        let missing_cluster = EvaluationResult::binary(&case, "slow", true, "ok");
        let clusters = vec![EvalCluster {
            id: "retrieval".to_string(),
            label: "Retrieval".to_string(),
            description: Some("Retrieval failures".to_string()),
            weight: 2.0,
            metadata: BTreeMap::new(),
        }];

        let report =
            EvaluationReport::from_results_with_clusters(&[failed, missing_cluster], &clusters);

        let retrieval_score = report
            .cluster_scores
            .iter()
            .find(|score| score.cluster_id == "retrieval")
            .unwrap();
        assert_eq!(retrieval_score.label.as_deref(), Some("Retrieval"));
        assert_eq!(
            retrieval_score.description.as_deref(),
            Some("Retrieval failures")
        );
        assert_eq!(
            report.failed_cases[0].cluster_label.as_deref(),
            Some("Retrieval")
        );
        assert_eq!(report.failed_cases[0].cluster_confidence, Some(0.82));
        assert_eq!(report.worst_clusters[0].label.as_deref(), Some("Retrieval"));
        assert_eq!(report.novelty_count, 1);
        assert_eq!(report.unclustered_count, 1);
    }
}
