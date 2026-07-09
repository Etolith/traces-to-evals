use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::clustering::UNCLUSTERED;
use crate::evaluation::{EvaluationResult, RunScore, WeightedAggregate};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub run_score: RunScore,
    pub evaluator_scores: Vec<EvaluatorScore>,
    pub cluster_scores: Vec<ClusterScore>,
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
    pub score: RunScore,
}

impl EvaluationReport {
    pub fn from_results(results: &[EvaluationResult]) -> Self {
        Self::from_results_with_aggregate(results, &WeightedAggregate::default())
    }

    pub fn from_results_with_aggregate(
        results: &[EvaluationResult],
        aggregate: &WeightedAggregate,
    ) -> Self {
        let case_ids = results
            .iter()
            .map(|result| result.case_id.as_str())
            .collect::<BTreeSet<_>>();

        Self {
            run_score: aggregate.score(results),
            evaluator_scores: evaluator_scores(results, aggregate),
            cluster_scores: cluster_scores(results, aggregate),
            total_cases: case_ids.len(),
            total_results: results.len(),
            metadata: BTreeMap::new(),
        }
    }
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
) -> Vec<ClusterScore> {
    group_by(results, |result| {
        result
            .cluster_id
            .clone()
            .unwrap_or_else(|| UNCLUSTERED.to_string())
    })
    .into_iter()
    .map(|(cluster_id, group)| ClusterScore {
        cluster_id,
        score: aggregate.score(&group),
    })
    .collect()
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
    }
}
