use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;
use crate::extractors::EvalCaseExtractor;
use crate::model::{EvalCase, Trace};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreScale {
    Binary,
    FourPoint,
    Unit,
}

impl ScoreScale {
    pub fn normalize(self, raw_score: f32) -> f32 {
        match self {
            Self::Binary | Self::Unit => raw_score.clamp(0.0, 1.0),
            Self::FourPoint => ((raw_score - 1.0) / 3.0).clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "llm-judge-openai", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct EvaluationCriteria {
    /// The answer directly addresses the user's request.
    pub relevance: bool,
    /// The answer is factually and procedurally correct.
    pub correctness: bool,
    /// The answer covers the important requirements of the request.
    pub completeness: bool,
    /// The answer avoids unsafe, unauthorized, or policy-violating content.
    pub safety: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub case_id: String,
    pub trace_id: String,
    pub evaluator_name: String,
    pub raw_score: f32,
    pub normalized_score: f32,
    pub score_scale: ScoreScale,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibrated_score: Option<f32>,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<String>,
    pub evaluation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<EvaluationCriteria>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl EvaluationResult {
    pub fn new(
        case: &EvalCase,
        evaluator_name: impl Into<String>,
        raw_score: f32,
        score_scale: ScoreScale,
        passed: bool,
        evaluation: impl Into<String>,
    ) -> Self {
        Self::from_ids(
            case.id.clone(),
            case.trace_id.clone(),
            evaluator_name,
            raw_score,
            score_scale,
            passed,
            evaluation,
        )
    }

    pub fn from_ids(
        case_id: impl Into<String>,
        trace_id: impl Into<String>,
        evaluator_name: impl Into<String>,
        raw_score: f32,
        score_scale: ScoreScale,
        passed: bool,
        evaluation: impl Into<String>,
    ) -> Self {
        Self {
            case_id: case_id.into(),
            trace_id: trace_id.into(),
            evaluator_name: evaluator_name.into(),
            raw_score,
            normalized_score: score_scale.normalize(raw_score),
            score_scale,
            calibrated_score: None,
            passed,
            confidence: None,
            cluster_id: None,
            evaluation: evaluation.into(),
            criteria: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn binary(
        case: &EvalCase,
        evaluator_name: impl Into<String>,
        passed: bool,
        evaluation: impl Into<String>,
    ) -> Self {
        Self::new(
            case,
            evaluator_name,
            if passed { 1.0 } else { 0.0 },
            ScoreScale::Binary,
            passed,
            evaluation,
        )
        .with_confidence(1.0)
    }

    pub fn with_calibrated_score(mut self, calibrated_score: f32) -> Self {
        self.calibrated_score = Some(calibrated_score.clamp(0.0, 1.0));
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    pub fn with_cluster_id(mut self, cluster_id: impl Into<String>) -> Self {
        self.cluster_id = Some(cluster_id.into());
        self
    }

    pub fn with_criteria(mut self, criteria: EvaluationCriteria) -> Self {
        self.criteria = Some(criteria);
        self
    }

    pub fn score_for_aggregation(&self) -> f32 {
        self.calibrated_score.unwrap_or(self.normalized_score)
    }
}

pub trait Evaluator {
    fn evaluator_name(&self) -> String;
    fn evaluate_case(&self, case: &EvalCase) -> Result<EvaluationResult>;

    fn evaluate_cases(&self, cases: &[EvalCase]) -> Result<Vec<EvaluationResult>> {
        cases.iter().map(|case| self.evaluate_case(case)).collect()
    }
}

#[async_trait::async_trait]
pub trait AsyncEvaluator: Send + Sync {
    fn evaluator_name(&self) -> String;
    async fn evaluate_case(&self, case: &EvalCase) -> Result<EvaluationResult>;

    async fn evaluate_cases(&self, cases: &[EvalCase]) -> Result<Vec<EvaluationResult>> {
        let mut results = Vec::with_capacity(cases.len());

        for case in cases {
            results.push(self.evaluate_case(case).await?);
        }

        Ok(results)
    }
}

#[async_trait::async_trait]
impl<T> AsyncEvaluator for T
where
    T: Evaluator + Send + Sync,
{
    fn evaluator_name(&self) -> String {
        Evaluator::evaluator_name(self)
    }

    async fn evaluate_case(&self, case: &EvalCase) -> Result<EvaluationResult> {
        Evaluator::evaluate_case(self, case)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationRun {
    pub cases: Vec<EvalCase>,
    pub results: Vec<EvaluationResult>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl EvaluationRun {
    pub fn new(cases: Vec<EvalCase>) -> Self {
        Self {
            cases,
            results: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn from_traces<E>(extractor: &E, traces: &[Trace]) -> Result<Self>
    where
        E: EvalCaseExtractor + ?Sized,
    {
        Ok(Self::new(extractor.extract_cases(traces)?))
    }

    pub fn evaluate_with<E>(mut self, evaluator: &E) -> Result<Self>
    where
        E: Evaluator + ?Sized,
    {
        self.results.extend(evaluator.evaluate_cases(&self.cases)?);
        Ok(self)
    }

    pub async fn evaluate_with_async<E>(mut self, evaluator: &E) -> Result<Self>
    where
        E: AsyncEvaluator + ?Sized,
    {
        self.results
            .extend(evaluator.evaluate_cases(&self.cases).await?);
        Ok(self)
    }

    pub fn add_results<I>(mut self, results: I) -> Self
    where
        I: IntoIterator<Item = EvaluationResult>,
    {
        self.results.extend(results);
        self
    }

    pub fn results(&self) -> &[EvaluationResult] {
        &self.results
    }

    pub fn into_results(self) -> Vec<EvaluationResult> {
        self.results
    }

    pub fn aggregate(&self) -> RunScore {
        WeightedAggregate::default().score(&self.results)
    }

    pub fn aggregate_with(&self, aggregate: &WeightedAggregate) -> RunScore {
        aggregate.score(&self.results)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunScore {
    pub result_count: usize,
    pub passed_count: usize,
    pub pass_rate: f32,
    pub weighted_score: f32,
    pub total_weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightedAggregate {
    default_weight: f32,
    evaluator_weights: BTreeMap<String, f32>,
    cluster_weights: BTreeMap<String, f32>,
}

impl Default for WeightedAggregate {
    fn default() -> Self {
        Self {
            default_weight: 1.0,
            evaluator_weights: BTreeMap::new(),
            cluster_weights: BTreeMap::new(),
        }
    }
}

impl WeightedAggregate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default_weight(mut self, weight: f32) -> Self {
        self.default_weight = weight.max(0.0);
        self
    }

    pub fn with_evaluator_weight(mut self, evaluator_name: impl Into<String>, weight: f32) -> Self {
        self.evaluator_weights
            .insert(evaluator_name.into(), weight.max(0.0));
        self
    }

    pub fn with_cluster_weight(mut self, cluster_id: impl Into<String>, weight: f32) -> Self {
        self.cluster_weights
            .insert(cluster_id.into(), weight.max(0.0));
        self
    }

    pub fn score(&self, results: &[EvaluationResult]) -> RunScore {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut passed_count = 0usize;

        for result in results {
            if result.passed {
                passed_count += 1;
            }

            let weight = self.weight_for(result);
            weighted_sum += result.score_for_aggregation() * weight;
            total_weight += weight;
        }

        RunScore {
            result_count: results.len(),
            passed_count,
            pass_rate: rate(passed_count, results.len()),
            weighted_score: if total_weight == 0.0 {
                0.0
            } else {
                weighted_sum / total_weight
            },
            total_weight,
        }
    }

    fn weight_for(&self, result: &EvaluationResult) -> f32 {
        let evaluator_weight = self
            .evaluator_weights
            .get(&result.evaluator_name)
            .copied()
            .unwrap_or(self.default_weight);

        let cluster_weight = result
            .cluster_id
            .as_ref()
            .and_then(|cluster_id| self.cluster_weights.get(cluster_id))
            .copied()
            .unwrap_or(1.0);

        evaluator_weight * cluster_weight
    }
}

pub fn evaluator_names(results: &[EvaluationResult]) -> BTreeSet<String> {
    results
        .iter()
        .map(|result| result.evaluator_name.clone())
        .collect()
}

fn rate(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graders::ExactMatchGrader;

    #[test]
    fn evaluates_run_with_multiple_evaluators() {
        let case = EvalCase::new("case-1", "trace-1", "input")
            .with_actual_output("answer")
            .with_expected_output("answer");

        let run = EvaluationRun::new(vec![case])
            .evaluate_with(&ExactMatchGrader)
            .unwrap();

        assert_eq!(run.results.len(), 1);
        assert_eq!(run.results[0].evaluator_name, "exact_match");
        assert_eq!(run.results[0].normalized_score, 1.0);
    }

    #[test]
    fn aggregates_weighted_scores() {
        let case = EvalCase::new("case-1", "trace-1", "input");
        let results = vec![
            EvaluationResult::binary(&case, "fast", true, "ok").with_cluster_id("a"),
            EvaluationResult::binary(&case, "slow", false, "bad").with_cluster_id("b"),
        ];
        let aggregate = WeightedAggregate::new()
            .with_evaluator_weight("fast", 1.0)
            .with_evaluator_weight("slow", 3.0);

        let score = aggregate.score(&results);

        assert_eq!(score.result_count, 2);
        assert_eq!(score.passed_count, 1);
        assert_eq!(score.weighted_score, 0.25);
    }
}
