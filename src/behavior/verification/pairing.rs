use std::collections::{BTreeMap, BTreeSet};

use crate::evaluation::{EvaluationResult, ScoreScale};

use super::model::{
    IncidentRegressionGate, PairedEvaluationComparison, PairedEvaluationKey,
    RemediationVerificationPolicy, SuiteRegressionGate,
};

pub(super) struct EvaluationResultIndex<'a> {
    results: BTreeMap<PairedEvaluationKey, &'a EvaluationResult>,
    duplicates: Vec<PairedEvaluationKey>,
    invalid_identities: Vec<PairedEvaluationKey>,
    invalid_scores: Vec<PairedEvaluationKey>,
}

impl<'a> EvaluationResultIndex<'a> {
    pub(super) fn new(results: &'a [EvaluationResult]) -> Self {
        let mut indexed = BTreeMap::new();
        let mut duplicates = BTreeSet::new();
        let mut invalid_identities = BTreeSet::new();
        let mut invalid_scores = BTreeSet::new();
        for result in results {
            let key = evaluation_key(result);
            if result.case_id.trim().is_empty()
                || result.trace_id.trim().is_empty()
                || result.evaluator_name.trim().is_empty()
            {
                invalid_identities.insert(key.clone());
            }
            if !scores_are_valid(result) {
                invalid_scores.insert(key.clone());
            }
            if indexed.insert(key.clone(), result).is_some() {
                duplicates.insert(key);
            }
        }
        Self {
            results: indexed,
            duplicates: duplicates.into_iter().collect(),
            invalid_identities: invalid_identities.into_iter().collect(),
            invalid_scores: invalid_scores.into_iter().collect(),
        }
    }
}

fn evaluation_key(result: &EvaluationResult) -> PairedEvaluationKey {
    let evaluator_version = ["evaluator_spec_hash", "evaluator_version"]
        .iter()
        .find_map(|key| result.metadata.get(*key).and_then(|value| value.as_str()))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    PairedEvaluationKey {
        case_id: result.case_id.clone(),
        evaluator_name: result.evaluator_name.clone(),
        evaluator_version,
    }
}

fn comparable_score(result: &EvaluationResult) -> Option<f32> {
    let score = result.score_for_aggregation();
    (score.is_finite() && (0.0..=1.0).contains(&score)).then_some(score)
}

fn scores_are_valid(result: &EvaluationResult) -> bool {
    let raw_valid = result.raw_score.is_finite()
        && match result.score_scale {
            ScoreScale::Binary | ScoreScale::Unit => (0.0..=1.0).contains(&result.raw_score),
            ScoreScale::FourPoint => (1.0..=4.0).contains(&result.raw_score),
        };
    raw_valid
        && result.normalized_score.is_finite()
        && (0.0..=1.0).contains(&result.normalized_score)
        && result
            .calibrated_score
            .is_none_or(|score| score.is_finite() && (0.0..=1.0).contains(&score))
}

fn comparison(
    key: &PairedEvaluationKey,
    baseline: &EvaluationResult,
    candidate: &EvaluationResult,
) -> PairedEvaluationComparison {
    let baseline_score = comparable_score(baseline);
    let candidate_score = comparable_score(candidate);
    let score_drop = baseline_score
        .zip(candidate_score)
        .map(|(baseline, candidate)| (baseline - candidate).max(0.0));
    PairedEvaluationComparison {
        key: key.clone(),
        baseline_passed: baseline.passed,
        candidate_passed: candidate.passed,
        baseline_score,
        candidate_score,
        score_drop,
    }
}

fn scoped_union(
    left: impl IntoIterator<Item = PairedEvaluationKey>,
    right: impl IntoIterator<Item = PairedEvaluationKey>,
) -> Vec<PairedEvaluationKey> {
    left.into_iter()
        .chain(right)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(super) fn incident_gate(
    case_id: &str,
    baseline: &EvaluationResultIndex<'_>,
    candidate: &EvaluationResultIndex<'_>,
) -> IncidentRegressionGate {
    let baseline_keys = baseline
        .results
        .keys()
        .filter(|key| key.case_id == case_id)
        .cloned()
        .collect::<Vec<_>>();
    let candidate_keys = candidate
        .results
        .keys()
        .filter(|key| key.case_id == case_id)
        .cloned()
        .collect::<Vec<_>>();
    let missing_candidate_results = baseline_keys
        .iter()
        .filter(|key| !candidate.results.contains_key(*key))
        .cloned()
        .collect::<Vec<_>>();
    let unexpected_candidate_results = candidate_keys
        .iter()
        .filter(|key| !baseline.results.contains_key(*key))
        .cloned()
        .collect::<Vec<_>>();
    let paired_results = baseline_keys
        .iter()
        .filter_map(|key| {
            candidate
                .results
                .get(key)
                .map(|candidate_result| comparison(key, baseline.results[key], candidate_result))
        })
        .collect::<Vec<_>>();
    let failed_candidate_results = paired_results
        .iter()
        .filter(|result| !result.candidate_passed)
        .map(|result| result.key.clone())
        .collect::<Vec<_>>();
    let unversioned_results = scoped_union(
        baseline_keys
            .iter()
            .filter(|key| key.evaluator_version.is_none())
            .cloned(),
        candidate_keys
            .iter()
            .filter(|key| key.evaluator_version.is_none())
            .cloned(),
    );
    let invalid_score_results = scoped_union(
        baseline
            .invalid_scores
            .iter()
            .filter(|key| key.case_id == case_id)
            .cloned(),
        candidate
            .invalid_scores
            .iter()
            .filter(|key| key.case_id == case_id)
            .cloned(),
    );
    let invalid_identity_results = scoped_union(
        baseline
            .invalid_identities
            .iter()
            .filter(|key| key.case_id == case_id)
            .cloned(),
        candidate
            .invalid_identities
            .iter()
            .filter(|key| key.case_id == case_id)
            .cloned(),
    );
    let duplicate_baseline_results = baseline
        .duplicates
        .iter()
        .filter(|key| key.case_id == case_id)
        .cloned()
        .collect::<Vec<_>>();
    let duplicate_candidate_results = candidate
        .duplicates
        .iter()
        .filter(|key| key.case_id == case_id)
        .cloned()
        .collect::<Vec<_>>();
    let paired_result_count = paired_results.len();
    let passed = !baseline_keys.is_empty()
        && missing_candidate_results.is_empty()
        && unexpected_candidate_results.is_empty()
        && failed_candidate_results.is_empty()
        && unversioned_results.is_empty()
        && invalid_identity_results.is_empty()
        && invalid_score_results.is_empty()
        && duplicate_baseline_results.is_empty()
        && duplicate_candidate_results.is_empty();
    IncidentRegressionGate {
        case_id: case_id.to_string(),
        passed,
        paired_result_count,
        paired_results,
        missing_candidate_results,
        unexpected_candidate_results,
        failed_candidate_results,
        unversioned_results,
        invalid_identity_results,
        invalid_score_results,
        duplicate_baseline_results,
        duplicate_candidate_results,
    }
}

pub(super) fn suite_gate(
    suite_case_ids: &BTreeSet<String>,
    baseline: &EvaluationResultIndex<'_>,
    candidate: &EvaluationResultIndex<'_>,
    policy: RemediationVerificationPolicy,
) -> SuiteRegressionGate {
    let baseline_keys = baseline
        .results
        .keys()
        .filter(|key| suite_case_ids.contains(&key.case_id))
        .cloned()
        .collect::<Vec<_>>();
    let candidate_keys = candidate
        .results
        .keys()
        .filter(|key| suite_case_ids.contains(&key.case_id))
        .cloned()
        .collect::<Vec<_>>();
    let missing_baseline_cases = suite_case_ids
        .iter()
        .filter(|case_id| !baseline_keys.iter().any(|key| &key.case_id == *case_id))
        .cloned()
        .collect::<Vec<_>>();
    let missing_candidate_results = baseline_keys
        .iter()
        .filter(|key| !candidate.results.contains_key(*key))
        .cloned()
        .collect::<Vec<_>>();
    let unexpected_candidate_results = candidate_keys
        .iter()
        .filter(|key| !baseline.results.contains_key(*key))
        .cloned()
        .collect::<Vec<_>>();
    let paired_results = baseline_keys
        .iter()
        .filter_map(|key| {
            candidate
                .results
                .get(key)
                .map(|candidate_result| comparison(key, baseline.results[key], candidate_result))
        })
        .collect::<Vec<_>>();
    let new_failure_results = paired_results
        .iter()
        .filter(|result| result.baseline_passed && !result.candidate_passed)
        .map(|result| result.key.clone())
        .collect::<Vec<_>>();
    let score_drop_results = paired_results
        .iter()
        .filter(|result| {
            result
                .score_drop
                .is_some_and(|drop| drop > policy.max_suite_score_drop)
        })
        .map(|result| result.key.clone())
        .collect::<Vec<_>>();
    let maximum_score_drop = paired_results
        .iter()
        .filter_map(|result| result.score_drop)
        .fold(0.0_f32, f32::max);
    let unversioned_results = scoped_union(
        baseline_keys
            .iter()
            .filter(|key| key.evaluator_version.is_none())
            .cloned(),
        candidate_keys
            .iter()
            .filter(|key| key.evaluator_version.is_none())
            .cloned(),
    );
    let invalid_score_results = scoped_union(
        baseline
            .invalid_scores
            .iter()
            .filter(|key| suite_case_ids.contains(&key.case_id))
            .cloned(),
        candidate
            .invalid_scores
            .iter()
            .filter(|key| suite_case_ids.contains(&key.case_id))
            .cloned(),
    );
    let invalid_identity_results = scoped_union(
        baseline
            .invalid_identities
            .iter()
            .filter(|key| suite_case_ids.contains(&key.case_id))
            .cloned(),
        candidate
            .invalid_identities
            .iter()
            .filter(|key| suite_case_ids.contains(&key.case_id))
            .cloned(),
    );
    let duplicate_baseline_results = baseline
        .duplicates
        .iter()
        .filter(|key| suite_case_ids.contains(&key.case_id))
        .cloned()
        .collect::<Vec<_>>();
    let duplicate_candidate_results = candidate
        .duplicates
        .iter()
        .filter(|key| suite_case_ids.contains(&key.case_id))
        .cloned()
        .collect::<Vec<_>>();
    let passed = !suite_case_ids.is_empty()
        && missing_baseline_cases.is_empty()
        && missing_candidate_results.is_empty()
        && unexpected_candidate_results.is_empty()
        && new_failure_results.len() <= policy.max_new_suite_failures
        && score_drop_results.is_empty()
        && unversioned_results.is_empty()
        && invalid_identity_results.is_empty()
        && invalid_score_results.is_empty()
        && duplicate_baseline_results.is_empty()
        && duplicate_candidate_results.is_empty();
    SuiteRegressionGate {
        passed,
        suite_case_count: suite_case_ids.len(),
        suite_case_ids: suite_case_ids.iter().cloned().collect(),
        paired_result_count: paired_results.len(),
        paired_results,
        missing_baseline_cases,
        missing_candidate_results,
        unexpected_candidate_results,
        new_failure_results,
        score_drop_results,
        unversioned_results,
        invalid_identity_results,
        invalid_score_results,
        duplicate_baseline_results,
        duplicate_candidate_results,
        maximum_score_drop,
    }
}
