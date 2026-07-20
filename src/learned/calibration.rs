use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{ContractError, canonical_content_id, require_non_empty, require_sha256};

pub const BINARY_CALIBRATION_MODEL_SCHEMA_VERSION: &str = "traceeval.binary_calibration_model.v1";
const BINARY_CALIBRATION_MODEL_HASH_DOMAIN: &str = "traceeval.binary-calibration-model.v1";
const FEATURE_NAMES: [&str; 10] = [
    "normalized_failure_score",
    "model_reported_confidence",
    "model_reported_confidence_missing",
    "evidence_coverage",
    "projection_truncated",
    "evaluator_disagreement",
    "evaluator_disagreement_missing",
    "missing_telemetry",
    "out_of_distribution_score",
    "out_of_distribution_score_missing",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationDataSplitV1 {
    Train,
    Calibration,
    Test,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedCalibrationFeaturesV1 {
    /// Raw evaluator signal with failure-oriented directionality: 1 is most
    /// likely to be a failure and 0 is least likely.
    pub normalized_failure_score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_reported_confidence: Option<f64>,
    pub evidence_coverage: f64,
    pub projection_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_disagreement: Option<f64>,
    pub missing_telemetry: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_of_distribution_score: Option<f64>,
}

impl LearnedCalibrationFeaturesV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        validate_probability(self.normalized_failure_score, "normalized_failure_score")?;
        validate_optional_probability(self.model_reported_confidence, "model_reported_confidence")?;
        validate_probability(self.evidence_coverage, "evidence_coverage")?;
        validate_optional_probability(self.evaluator_disagreement, "evaluator_disagreement")?;
        validate_probability(self.missing_telemetry, "missing_telemetry")?;
        validate_optional_probability(self.out_of_distribution_score, "out_of_distribution_score")
    }

    fn vector(&self) -> [f64; FEATURE_NAMES.len()] {
        [
            self.normalized_failure_score,
            self.model_reported_confidence.unwrap_or(0.0),
            f64::from(self.model_reported_confidence.is_none()),
            self.evidence_coverage,
            f64::from(self.projection_truncated),
            self.evaluator_disagreement.unwrap_or(0.0),
            f64::from(self.evaluator_disagreement.is_none()),
            self.missing_telemetry,
            self.out_of_distribution_score.unwrap_or(0.0),
            f64::from(self.out_of_distribution_score.is_none()),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryCalibrationExampleV1 {
    pub observation_id: String,
    pub group_id: String,
    pub evaluator_release_id: String,
    pub split: CalibrationDataSplitV1,
    pub features: LearnedCalibrationFeaturesV1,
    /// Positive-class convention is fixed: `true` means observed failure.
    pub label_failure: bool,
}

impl BinaryCalibrationExampleV1 {
    fn validate(&self) -> Result<(), ContractError> {
        require_non_empty(&self.observation_id, "observation_id", calibration_error)?;
        require_non_empty(&self.group_id, "group_id", calibration_error)?;
        require_sha256(
            &self.evaluator_release_id,
            "evaluator_release_id",
            calibration_error,
        )?;
        self.features.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryCalibrationFitOptionsV1 {
    pub l2_lambda: f64,
    pub max_iterations: u32,
    pub gradient_tolerance: f64,
    pub initial_step_size: f64,
}

impl Default for BinaryCalibrationFitOptionsV1 {
    fn default() -> Self {
        Self {
            l2_lambda: 0.1,
            max_iterations: 10_000,
            gradient_tolerance: 1e-8,
            initial_step_size: 1.0,
        }
    }
}

impl BinaryCalibrationFitOptionsV1 {
    fn validate(&self) -> Result<(), ContractError> {
        if !self.l2_lambda.is_finite() || self.l2_lambda <= 0.0 {
            return Err(calibration_error(
                "l2_lambda must be finite and greater than zero",
            ));
        }
        if self.max_iterations == 0 {
            return Err(calibration_error(
                "max_iterations must be greater than zero",
            ));
        }
        if !self.gradient_tolerance.is_finite() || self.gradient_tolerance <= 0.0 {
            return Err(calibration_error(
                "gradient_tolerance must be finite and greater than zero",
            ));
        }
        if !self.initial_step_size.is_finite() || self.initial_step_size <= 0.0 {
            return Err(calibration_error(
                "initial_step_size must be finite and greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryCalibrationModelV1 {
    pub schema_version: String,
    pub evaluator_release_id: String,
    pub positive_class: String,
    pub fit_options: BinaryCalibrationFitOptionsV1,
    pub feature_names: Vec<String>,
    pub feature_means: Vec<f64>,
    pub feature_scales: Vec<f64>,
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub calibration_observations: u64,
    pub positive_observations: u64,
    pub group_count: u64,
    pub converged: bool,
    pub iterations: u32,
}

impl BinaryCalibrationModelV1 {
    /// Fits L2-regularized logistic calibration on a caller-frozen calibration
    /// split. Group IDs are retained for leakage audits; this routine never
    /// creates a random split internally.
    pub fn fit(
        examples: &[BinaryCalibrationExampleV1],
        options: BinaryCalibrationFitOptionsV1,
    ) -> Result<Self, ContractError> {
        options.validate()?;
        if examples.len() < 2 {
            return Err(calibration_error(
                "at least two calibration observations are required",
            ));
        }
        let mut observation_ids = BTreeSet::new();
        let mut evaluator_release_id = None::<&str>;
        let mut positives = 0_u64;
        let mut groups = BTreeSet::new();
        let mut ordered = examples.iter().collect::<Vec<_>>();
        ordered.sort_by(|left, right| left.observation_id.cmp(&right.observation_id));
        let mut raw = Vec::with_capacity(examples.len());
        for example in ordered {
            example.validate()?;
            if example.split != CalibrationDataSplitV1::Calibration {
                return Err(calibration_error(
                    "fit accepts only the frozen calibration split",
                ));
            }
            if !observation_ids.insert(example.observation_id.as_str()) {
                return Err(calibration_error(format!(
                    "duplicate observation_id {}",
                    example.observation_id
                )));
            }
            match evaluator_release_id {
                Some(expected) if expected != example.evaluator_release_id => {
                    return Err(calibration_error(
                        "one calibration model cannot mix evaluator releases",
                    ));
                }
                None => evaluator_release_id = Some(&example.evaluator_release_id),
                _ => {}
            }
            positives += u64::from(example.label_failure);
            groups.insert(example.group_id.as_str());
            raw.push((example.features.vector(), example.label_failure));
        }
        if positives == 0 || positives == examples.len() as u64 {
            return Err(calibration_error(
                "calibration requires both failure and non-failure observations",
            ));
        }

        let (means, scales) = feature_standardization(&raw);
        let rows = raw
            .iter()
            .map(|(features, label)| {
                let standardized =
                    std::array::from_fn(|index| (features[index] - means[index]) / scales[index]);
                (standardized, *label)
            })
            .collect::<Vec<_>>();
        let (intercept, coefficients, iterations) = fit_logistic(&rows, &options)?;
        let model = Self {
            schema_version: BINARY_CALIBRATION_MODEL_SCHEMA_VERSION.into(),
            evaluator_release_id: evaluator_release_id.unwrap_or_default().into(),
            positive_class: "failure".into(),
            fit_options: options,
            feature_names: FEATURE_NAMES.into_iter().map(str::to_string).collect(),
            feature_means: means.into(),
            feature_scales: scales.into(),
            intercept,
            coefficients: coefficients.into(),
            calibration_observations: examples.len() as u64,
            positive_observations: positives,
            group_count: groups.len() as u64,
            converged: true,
            iterations,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != BINARY_CALIBRATION_MODEL_SCHEMA_VERSION {
            return Err(calibration_error(
                "unsupported binary calibration schema version",
            ));
        }
        require_sha256(
            &self.evaluator_release_id,
            "evaluator_release_id",
            calibration_error,
        )?;
        if self.positive_class != "failure" {
            return Err(calibration_error("positive_class must be failure"));
        }
        self.fit_options.validate()?;
        if self.feature_names != FEATURE_NAMES
            || self.feature_means.len() != FEATURE_NAMES.len()
            || self.feature_scales.len() != FEATURE_NAMES.len()
            || self.coefficients.len() != FEATURE_NAMES.len()
        {
            return Err(calibration_error(
                "feature names, means, scales, and coefficients must match the v1 schema",
            ));
        }
        if !self.intercept.is_finite()
            || self
                .feature_means
                .iter()
                .chain(self.feature_scales.iter())
                .chain(self.coefficients.iter())
                .any(|value| !value.is_finite())
            || self.feature_scales.iter().any(|scale| *scale <= 0.0)
        {
            return Err(calibration_error(
                "calibration parameters must be finite and scales positive",
            ));
        }
        if self.calibration_observations == 0
            || self.positive_observations == 0
            || self.positive_observations >= self.calibration_observations
            || self.group_count == 0
            || !self.converged
        {
            return Err(calibration_error(
                "calibration provenance counts are inconsistent",
            ));
        }
        Ok(())
    }

    pub fn model_id(&self) -> Result<String, ContractError> {
        self.validate()?;
        canonical_content_id(BINARY_CALIBRATION_MODEL_HASH_DOMAIN, self)
    }

    pub fn predict_failure_probability(
        &self,
        features: &LearnedCalibrationFeaturesV1,
    ) -> Result<f64, ContractError> {
        self.validate()?;
        features.validate()?;
        let features = features.vector();
        let logit = self.coefficients.iter().enumerate().fold(
            self.intercept,
            |value, (index, coefficient)| {
                value
                    + coefficient
                        * ((features[index] - self.feature_means[index])
                            / self.feature_scales[index])
            },
        );
        Ok(sigmoid(logit))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryPredictionV1 {
    pub observation_id: String,
    pub group_id: String,
    pub evaluator_release_id: String,
    pub calibration_model_id: String,
    pub split: CalibrationDataSplitV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probability_failure: Option<f64>,
    pub label_failure: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfusionMatrixV1 {
    pub positive_class: String,
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBinV1 {
    pub lower_bound_inclusive: f64,
    pub upper_bound: f64,
    pub upper_bound_inclusive: bool,
    pub count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_predicted_failure: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub empirical_failure_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absolute_gap: Option<f64>,
    pub weighted_contribution: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectiveRiskPointV1 {
    pub decided_count: u64,
    pub attempted_count: u64,
    pub coverage: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classification_risk: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_confidence: Option<f64>,
}

/// A deterministic 95% Wilson score interval for a binomial rate. The exact
/// numerator and denominator travel with the interval so a UI or canonical
/// scorer can reconcile every displayed bound.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinomialRateIntervalV1 {
    pub successes: u64,
    pub trials: u64,
    pub estimate: f64,
    pub lower_95: f64,
    pub upper_95: f64,
    pub method: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryCalibrationReportV1 {
    pub evaluator_release_id: String,
    pub calibration_model_id: String,
    pub split: CalibrationDataSplitV1,
    pub positive_class: String,
    pub decision_threshold: f64,
    pub attempted_count: u64,
    pub decided_count: u64,
    pub abstained_count: u64,
    pub confusion: ConfusionMatrixV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision_interval: Option<BinomialRateIntervalV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_interval: Option<BinomialRateIntervalV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub specificity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub specificity_interval: Option<BinomialRateIntervalV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f1: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub macro_f1: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matthews_correlation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_precision: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auprc_trapezoid: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brier_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_calibration_error: Option<f64>,
    pub calibration_bins: Vec<CalibrationBinV1>,
    pub selective_risk: Vec<SelectiveRiskPointV1>,
}

impl BinaryCalibrationReportV1 {
    pub fn from_predictions(
        predictions: &[BinaryPredictionV1],
        decision_threshold: f64,
        calibration_bin_count: u32,
    ) -> Result<Self, ContractError> {
        validate_probability(decision_threshold, "decision_threshold")?;
        if calibration_bin_count == 0 || calibration_bin_count > 1_000 {
            return Err(calibration_error(
                "calibration_bin_count must be between 1 and 1000",
            ));
        }
        if predictions.is_empty() {
            return Err(calibration_error(
                "at least one held-out prediction is required",
            ));
        }
        let mut observation_ids = BTreeSet::new();
        let mut evaluator_release_id = None::<&str>;
        let mut calibration_model_id = None::<&str>;
        let mut split = None::<CalibrationDataSplitV1>;
        let mut decided = Vec::new();
        for prediction in predictions {
            require_non_empty(
                &prediction.observation_id,
                "observation_id",
                calibration_error,
            )?;
            require_non_empty(&prediction.group_id, "group_id", calibration_error)?;
            require_sha256(
                &prediction.evaluator_release_id,
                "evaluator_release_id",
                calibration_error,
            )?;
            require_sha256(
                &prediction.calibration_model_id,
                "calibration_model_id",
                calibration_error,
            )?;
            if !observation_ids.insert(prediction.observation_id.as_str()) {
                return Err(calibration_error(format!(
                    "duplicate observation_id {}",
                    prediction.observation_id
                )));
            }
            enforce_same(
                &mut evaluator_release_id,
                &prediction.evaluator_release_id,
                "evaluator_release_id",
            )?;
            enforce_same(
                &mut calibration_model_id,
                &prediction.calibration_model_id,
                "calibration_model_id",
            )?;
            match split {
                Some(expected) if expected != prediction.split => {
                    return Err(calibration_error(
                        "one report cannot mix train, calibration, and test splits",
                    ));
                }
                None => split = Some(prediction.split),
                _ => {}
            }
            if let Some(probability) = prediction.probability_failure {
                validate_probability(probability, "probability_failure")?;
                decided.push((probability, prediction.label_failure));
            }
        }

        let confusion = confusion_matrix(&decided, decision_threshold);
        let precision = ratio(
            confusion.true_positive,
            confusion.true_positive + confusion.false_positive,
        );
        let recall = ratio(
            confusion.true_positive,
            confusion.true_positive + confusion.false_negative,
        );
        let specificity = ratio(
            confusion.true_negative,
            confusion.true_negative + confusion.false_positive,
        );
        let precision_interval = wilson_95(
            confusion.true_positive,
            confusion.true_positive + confusion.false_positive,
        );
        let recall_interval = wilson_95(
            confusion.true_positive,
            confusion.true_positive + confusion.false_negative,
        );
        let specificity_interval = wilson_95(
            confusion.true_negative,
            confusion.true_negative + confusion.false_positive,
        );
        let f1 = harmonic_mean(precision, recall);
        let negative_precision = ratio(
            confusion.true_negative,
            confusion.true_negative + confusion.false_negative,
        );
        let negative_recall = specificity;
        let negative_f1 = harmonic_mean(negative_precision, negative_recall);
        let macro_f1 = match (f1, negative_f1) {
            (Some(positive), Some(negative)) => Some((positive + negative) / 2.0),
            _ => None,
        };
        let matthews_correlation = matthews(&confusion);
        let (average_precision, auprc_trapezoid) = precision_recall_areas(&decided);
        let brier_score = (!decided.is_empty()).then(|| {
            decided
                .iter()
                .map(|(probability, label)| {
                    let label = f64::from(*label);
                    (probability - label).powi(2)
                })
                .sum::<f64>()
                / decided.len() as f64
        });
        let calibration_bins = calibration_bins(&decided, calibration_bin_count);
        let expected_calibration_error = (!decided.is_empty()).then(|| {
            calibration_bins
                .iter()
                .map(|bin| bin.weighted_contribution)
                .sum()
        });
        let attempted_count = predictions.len() as u64;
        let decided_count = decided.len() as u64;
        Ok(Self {
            evaluator_release_id: evaluator_release_id.unwrap_or_default().into(),
            calibration_model_id: calibration_model_id.unwrap_or_default().into(),
            split: split.unwrap_or(CalibrationDataSplitV1::Test),
            positive_class: "failure".into(),
            decision_threshold,
            attempted_count,
            decided_count,
            abstained_count: attempted_count - decided_count,
            confusion,
            precision,
            precision_interval,
            recall,
            recall_interval,
            specificity,
            specificity_interval,
            f1,
            macro_f1,
            matthews_correlation,
            average_precision,
            auprc_trapezoid,
            brier_score,
            expected_calibration_error,
            calibration_bins,
            selective_risk: selective_risk(&decided, attempted_count, decision_threshold),
        })
    }
}

fn wilson_95(successes: u64, trials: u64) -> Option<BinomialRateIntervalV1> {
    if trials == 0 || successes > trials {
        return None;
    }
    // 1.959963984540054 is the 97.5th percentile of the standard normal.
    let z = 1.959_963_984_540_054_f64;
    let n = trials as f64;
    let estimate = successes as f64 / n;
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / n;
    let center = (estimate + z_squared / (2.0 * n)) / denominator;
    let half_width =
        z * ((estimate * (1.0 - estimate) / n + z_squared / (4.0 * n * n)).sqrt()) / denominator;
    let lower_95 = if successes == 0 {
        0.0
    } else {
        (center - half_width).max(0.0)
    };
    let upper_95 = if successes == trials {
        1.0
    } else {
        (center + half_width).min(1.0)
    };
    Some(BinomialRateIntervalV1 {
        successes,
        trials,
        estimate,
        lower_95,
        upper_95,
        method: "wilson_score_95".into(),
    })
}

fn feature_standardization(
    rows: &[([f64; FEATURE_NAMES.len()], bool)],
) -> ([f64; FEATURE_NAMES.len()], [f64; FEATURE_NAMES.len()]) {
    let mut means = [0.0; FEATURE_NAMES.len()];
    for (features, _) in rows {
        for (index, value) in features.iter().enumerate() {
            means[index] += value;
        }
    }
    for mean in &mut means {
        *mean /= rows.len() as f64;
    }
    let mut scales = [0.0; FEATURE_NAMES.len()];
    for (features, _) in rows {
        for (index, value) in features.iter().enumerate() {
            scales[index] += (value - means[index]).powi(2);
        }
    }
    for scale in &mut scales {
        *scale = (*scale / rows.len() as f64).sqrt();
        if *scale < 1e-12 {
            *scale = 1.0;
        }
    }
    (means, scales)
}

fn fit_logistic(
    rows: &[([f64; FEATURE_NAMES.len()], bool)],
    options: &BinaryCalibrationFitOptionsV1,
) -> Result<(f64, [f64; FEATURE_NAMES.len()], u32), ContractError> {
    let mut intercept = 0.0;
    let mut coefficients = [0.0; FEATURE_NAMES.len()];
    let mut current_loss = logistic_loss(rows, intercept, &coefficients, options.l2_lambda);
    for iteration in 1..=options.max_iterations {
        let (intercept_gradient, coefficient_gradient) =
            logistic_gradient(rows, intercept, &coefficients, options.l2_lambda);
        let gradient_norm = (intercept_gradient.powi(2)
            + coefficient_gradient
                .iter()
                .map(|value| value.powi(2))
                .sum::<f64>())
        .sqrt();
        if gradient_norm <= options.gradient_tolerance {
            return Ok((intercept, coefficients, iteration - 1));
        }
        let mut step = options.initial_step_size;
        let mut accepted = None;
        while step >= 1e-12 {
            let candidate_intercept = intercept - step * intercept_gradient;
            let candidate_coefficients = std::array::from_fn(|index| {
                coefficients[index] - step * coefficient_gradient[index]
            });
            let candidate_loss = logistic_loss(
                rows,
                candidate_intercept,
                &candidate_coefficients,
                options.l2_lambda,
            );
            if candidate_loss.is_finite() && candidate_loss < current_loss {
                accepted = Some((candidate_intercept, candidate_coefficients, candidate_loss));
                break;
            }
            step *= 0.5;
        }
        let Some((next_intercept, next_coefficients, next_loss)) = accepted else {
            return Err(calibration_error(format!(
                "logistic calibration line search failed at iteration {iteration}"
            )));
        };
        intercept = next_intercept;
        coefficients = next_coefficients;
        current_loss = next_loss;
    }
    Err(calibration_error(format!(
        "logistic calibration did not converge in {} iterations",
        options.max_iterations
    )))
}

fn logistic_loss(
    rows: &[([f64; FEATURE_NAMES.len()], bool)],
    intercept: f64,
    coefficients: &[f64; FEATURE_NAMES.len()],
    l2_lambda: f64,
) -> f64 {
    let data_loss = rows
        .iter()
        .map(|(features, label)| {
            let logit = coefficients
                .iter()
                .zip(features.iter())
                .fold(intercept, |value, (coefficient, feature)| {
                    value + coefficient * feature
                });
            let probability = sigmoid(logit).clamp(1e-15, 1.0 - 1e-15);
            if *label {
                -probability.ln()
            } else {
                -(1.0 - probability).ln()
            }
        })
        .sum::<f64>()
        / rows.len() as f64;
    data_loss + 0.5 * l2_lambda * coefficients.iter().map(|value| value.powi(2)).sum::<f64>()
}

fn logistic_gradient(
    rows: &[([f64; FEATURE_NAMES.len()], bool)],
    intercept: f64,
    coefficients: &[f64; FEATURE_NAMES.len()],
    l2_lambda: f64,
) -> (f64, [f64; FEATURE_NAMES.len()]) {
    let mut intercept_gradient = 0.0;
    let mut coefficient_gradient = [0.0; FEATURE_NAMES.len()];
    for (features, label) in rows {
        let logit = coefficients
            .iter()
            .zip(features.iter())
            .fold(intercept, |value, (coefficient, feature)| {
                value + coefficient * feature
            });
        let residual = sigmoid(logit) - f64::from(*label);
        intercept_gradient += residual;
        for (index, feature) in features.iter().enumerate() {
            coefficient_gradient[index] += residual * feature;
        }
    }
    intercept_gradient /= rows.len() as f64;
    for (gradient, coefficient) in coefficient_gradient.iter_mut().zip(coefficients.iter()) {
        *gradient = *gradient / rows.len() as f64 + l2_lambda * coefficient;
    }
    (intercept_gradient, coefficient_gradient)
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn confusion_matrix(decided: &[(f64, bool)], threshold: f64) -> ConfusionMatrixV1 {
    let mut confusion = ConfusionMatrixV1 {
        positive_class: "failure".into(),
        true_positive: 0,
        false_positive: 0,
        true_negative: 0,
        false_negative: 0,
    };
    for (probability, label) in decided {
        match (*probability >= threshold, *label) {
            (true, true) => confusion.true_positive += 1,
            (true, false) => confusion.false_positive += 1,
            (false, false) => confusion.true_negative += 1,
            (false, true) => confusion.false_negative += 1,
        }
    }
    confusion
}

fn calibration_bins(decided: &[(f64, bool)], bin_count: u32) -> Vec<CalibrationBinV1> {
    let mut bins = BTreeMap::<u32, (u64, f64, u64)>::new();
    for (probability, label) in decided {
        let index = ((*probability * bin_count as f64).floor() as u32).min(bin_count - 1);
        let entry = bins.entry(index).or_default();
        entry.0 += 1;
        entry.1 += probability;
        entry.2 += u64::from(*label);
    }
    (0..bin_count)
        .map(|index| {
            let (count, probability_sum, positive_count) =
                bins.get(&index).copied().unwrap_or_default();
            let mean_predicted_failure = (count > 0).then(|| probability_sum / count as f64);
            let empirical_failure_rate = (count > 0).then(|| positive_count as f64 / count as f64);
            let absolute_gap = mean_predicted_failure
                .zip(empirical_failure_rate)
                .map(|(predicted, observed)| (predicted - observed).abs());
            CalibrationBinV1 {
                lower_bound_inclusive: index as f64 / bin_count as f64,
                upper_bound: (index + 1) as f64 / bin_count as f64,
                upper_bound_inclusive: index + 1 == bin_count,
                count,
                mean_predicted_failure,
                empirical_failure_rate,
                absolute_gap,
                weighted_contribution: absolute_gap.unwrap_or_default() * count as f64
                    / decided.len().max(1) as f64,
            }
        })
        .collect()
}

fn precision_recall_areas(decided: &[(f64, bool)]) -> (Option<f64>, Option<f64>) {
    let positive_count = decided.iter().filter(|(_, label)| *label).count();
    if decided.is_empty() || positive_count == 0 {
        return (None, None);
    }
    let mut ranked = decided.to_vec();
    ranked.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut true_positive = 0_u64;
    let mut false_positive = 0_u64;
    let mut average_precision = 0.0;
    let mut previous_recall = 0.0;
    let mut previous_precision = 1.0;
    let mut trapezoid = 0.0;
    let mut index = 0;
    while index < ranked.len() {
        let threshold = ranked[index].0;
        let mut group_positives = 0_u64;
        let mut group_negatives = 0_u64;
        while index < ranked.len() && ranked[index].0 == threshold {
            if ranked[index].1 {
                group_positives += 1;
            } else {
                group_negatives += 1;
            }
            index += 1;
        }
        true_positive += group_positives;
        false_positive += group_negatives;
        let recall = true_positive as f64 / positive_count as f64;
        let precision = true_positive as f64 / (true_positive + false_positive) as f64;
        average_precision += (recall - previous_recall) * precision;
        trapezoid += (recall - previous_recall) * (precision + previous_precision) / 2.0;
        previous_recall = recall;
        previous_precision = precision;
    }
    (Some(average_precision), Some(trapezoid))
}

fn selective_risk(
    decided: &[(f64, bool)],
    attempted_count: u64,
    threshold: f64,
) -> Vec<SelectiveRiskPointV1> {
    let mut ranked = decided
        .iter()
        .map(|(probability, label)| {
            (
                if *probability >= threshold {
                    *probability
                } else {
                    1.0 - *probability
                },
                (*probability >= threshold) != *label,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut curve = vec![SelectiveRiskPointV1 {
        decided_count: 0,
        attempted_count,
        coverage: 0.0,
        classification_risk: None,
        minimum_confidence: None,
    }];
    let mut errors = 0_u64;
    let mut count = 0_u64;
    let mut index = 0;
    while index < ranked.len() {
        let confidence = ranked[index].0;
        while index < ranked.len() && ranked[index].0 == confidence {
            errors += u64::from(ranked[index].1);
            count += 1;
            index += 1;
        }
        curve.push(SelectiveRiskPointV1 {
            decided_count: count,
            attempted_count,
            coverage: count as f64 / attempted_count as f64,
            classification_risk: Some(errors as f64 / count as f64),
            minimum_confidence: Some(confidence),
        });
    }
    curve
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn harmonic_mean(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) if left + right > 0.0 => {
            Some(2.0 * left * right / (left + right))
        }
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    }
}

fn matthews(confusion: &ConfusionMatrixV1) -> Option<f64> {
    let tp = confusion.true_positive as f64;
    let fp = confusion.false_positive as f64;
    let tn = confusion.true_negative as f64;
    let fn_ = confusion.false_negative as f64;
    let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    (denominator > 0.0).then(|| (tp * tn - fp * fn_) / denominator)
}

fn enforce_same<'a>(
    current: &mut Option<&'a str>,
    candidate: &'a str,
    field: &str,
) -> Result<(), ContractError> {
    match current {
        Some(expected) if *expected != candidate => Err(calibration_error(format!(
            "one report cannot mix {field} values"
        ))),
        None => {
            *current = Some(candidate);
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_probability(value: f64, field: &str) -> Result<(), ContractError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(calibration_error(format!(
            "{field} must be finite and between 0 and 1"
        )));
    }
    Ok(())
}

fn validate_optional_probability(value: Option<f64>, field: &str) -> Result<(), ContractError> {
    if let Some(value) = value {
        validate_probability(value, field)?;
    }
    Ok(())
}

fn calibration_error(message: impl Into<String>) -> ContractError {
    ContractError::InvalidCalibration(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn features(score: f64) -> LearnedCalibrationFeaturesV1 {
        LearnedCalibrationFeaturesV1 {
            normalized_failure_score: score,
            model_reported_confidence: Some(0.9),
            evidence_coverage: 1.0,
            projection_truncated: false,
            evaluator_disagreement: None,
            missing_telemetry: 0.0,
            out_of_distribution_score: None,
        }
    }

    #[test]
    fn logistic_calibration_is_deterministic_and_failure_oriented() {
        let examples = (0..40)
            .map(|index| {
                let label_failure = index >= 20;
                BinaryCalibrationExampleV1 {
                    observation_id: format!("cal-{index}"),
                    group_id: format!("session-{}", index / 2),
                    evaluator_release_id: digest('a'),
                    split: CalibrationDataSplitV1::Calibration,
                    features: features(if label_failure { 0.8 } else { 0.2 }),
                    label_failure,
                }
            })
            .collect::<Vec<_>>();
        let first =
            BinaryCalibrationModelV1::fit(&examples, BinaryCalibrationFitOptionsV1::default())
                .unwrap();
        let mut reordered = examples.clone();
        reordered.reverse();
        let second =
            BinaryCalibrationModelV1::fit(&reordered, BinaryCalibrationFitOptionsV1::default())
                .unwrap();

        assert!(first.converged);
        assert_eq!(first, second);
        assert_eq!(first.model_id().unwrap(), second.model_id().unwrap());
        assert!(
            first.predict_failure_probability(&features(0.9)).unwrap()
                > first.predict_failure_probability(&features(0.1)).unwrap()
        );
    }

    #[test]
    fn fit_rejects_split_leakage_duplicates_and_single_class_data() {
        let example = BinaryCalibrationExampleV1 {
            observation_id: "one".into(),
            group_id: "session-one".into(),
            evaluator_release_id: digest('a'),
            split: CalibrationDataSplitV1::Train,
            features: features(0.5),
            label_failure: true,
        };
        assert!(
            BinaryCalibrationModelV1::fit(
                &[example.clone(), example],
                BinaryCalibrationFitOptionsV1::default()
            )
            .unwrap_err()
            .to_string()
            .contains("calibration split")
        );
    }

    #[test]
    fn report_preserves_abstentions_and_matches_known_metrics() {
        let predictions = vec![
            ("a", Some(0.9), true),
            ("b", Some(0.8), false),
            ("c", Some(0.2), false),
            ("d", Some(0.1), true),
            ("e", None, true),
        ]
        .into_iter()
        .map(
            |(observation_id, probability_failure, label_failure)| BinaryPredictionV1 {
                observation_id: observation_id.into(),
                group_id: format!("group-{observation_id}"),
                evaluator_release_id: digest('a'),
                calibration_model_id: digest('b'),
                split: CalibrationDataSplitV1::Test,
                probability_failure,
                label_failure,
            },
        )
        .collect::<Vec<_>>();
        let report = BinaryCalibrationReportV1::from_predictions(&predictions, 0.5, 5).unwrap();

        assert_eq!(report.attempted_count, 5);
        assert_eq!(report.decided_count, 4);
        assert_eq!(report.abstained_count, 1);
        assert_eq!(report.confusion.true_positive, 1);
        assert_eq!(report.confusion.false_positive, 1);
        assert_eq!(report.confusion.true_negative, 1);
        assert_eq!(report.confusion.false_negative, 1);
        assert_eq!(report.precision, Some(0.5));
        assert_eq!(report.recall, Some(0.5));
        let precision_interval = report.precision_interval.as_ref().unwrap();
        assert_eq!(
            (precision_interval.successes, precision_interval.trials),
            (1, 2)
        );
        assert!((precision_interval.lower_95 - 0.094_531_205_734_230_74).abs() < 1e-12);
        assert!((precision_interval.upper_95 - 0.905_468_794_265_769_3).abs() < 1e-12);
        assert_eq!(precision_interval.method, "wilson_score_95");
        assert_eq!(report.f1, Some(0.5));
        assert_eq!(report.macro_f1, Some(0.5));
        assert_eq!(report.matthews_correlation, Some(0.0));
        assert!(report.brier_score.unwrap() > 0.3);
        assert_eq!(report.selective_risk.last().unwrap().coverage, 0.8);
    }

    #[test]
    fn wilson_interval_handles_boundaries_and_missing_denominators() {
        assert_eq!(wilson_95(0, 0), None);
        assert_eq!(wilson_95(2, 1), None);
        let none_succeeded = wilson_95(0, 10).unwrap();
        assert_eq!(none_succeeded.lower_95, 0.0);
        assert!(none_succeeded.upper_95 > 0.27 && none_succeeded.upper_95 < 0.28);
        let all_succeeded = wilson_95(10, 10).unwrap();
        assert!(all_succeeded.lower_95 > 0.72 && all_succeeded.lower_95 < 0.73);
        assert_eq!(all_succeeded.upper_95, 1.0);
    }

    #[test]
    fn report_rejects_mixed_releases_and_invalid_probabilities() {
        let mut predictions = vec![BinaryPredictionV1 {
            observation_id: "a".into(),
            group_id: "g-a".into(),
            evaluator_release_id: digest('a'),
            calibration_model_id: digest('b'),
            split: CalibrationDataSplitV1::Test,
            probability_failure: Some(0.5),
            label_failure: true,
        }];
        let mut second = predictions[0].clone();
        second.observation_id = "b".into();
        second.evaluator_release_id = digest('c');
        predictions.push(second);
        assert!(
            BinaryCalibrationReportV1::from_predictions(&predictions, 0.5, 10)
                .unwrap_err()
                .to_string()
                .contains("evaluator_release_id")
        );
    }

    #[test]
    fn report_matches_exact_calibration_metrics_and_preserves_empty_bins() {
        let predictions = [
            ("a", 0.1, false),
            ("b", 0.2, false),
            ("c", 0.8, true),
            ("d", 0.9, true),
        ]
        .into_iter()
        .map(
            |(observation_id, probability_failure, label_failure)| BinaryPredictionV1 {
                observation_id: observation_id.into(),
                group_id: format!("group-{observation_id}"),
                evaluator_release_id: digest('a'),
                calibration_model_id: digest('b'),
                split: CalibrationDataSplitV1::Test,
                probability_failure: Some(probability_failure),
                label_failure,
            },
        )
        .collect::<Vec<_>>();
        let report = BinaryCalibrationReportV1::from_predictions(&predictions, 0.5, 4).unwrap();

        assert!((report.brier_score.unwrap() - 0.025).abs() < 1e-12);
        assert!((report.expected_calibration_error.unwrap() - 0.15).abs() < 1e-12);
        assert_eq!(report.average_precision, Some(1.0));
        assert_eq!(report.matthews_correlation, Some(1.0));
        assert_eq!(report.calibration_bins.len(), 4);
        assert_eq!(report.calibration_bins[1].count, 0);
        assert_eq!(report.calibration_bins[1].absolute_gap, None);
        assert!(!report.calibration_bins[0].upper_bound_inclusive);
        assert!(report.calibration_bins[3].upper_bound_inclusive);
    }

    #[test]
    fn ranking_and_selective_risk_are_tie_invariant() {
        let rows = [
            ("a", 0.9, true),
            ("b", 0.9, false),
            ("c", 0.8, true),
            ("d", 0.8, false),
        ];
        let predictions = rows
            .into_iter()
            .map(
                |(observation_id, probability_failure, label_failure)| BinaryPredictionV1 {
                    observation_id: observation_id.into(),
                    group_id: format!("group-{observation_id}"),
                    evaluator_release_id: digest('a'),
                    calibration_model_id: digest('b'),
                    split: CalibrationDataSplitV1::Test,
                    probability_failure: Some(probability_failure),
                    label_failure,
                },
            )
            .collect::<Vec<_>>();
        let mut reordered = predictions.clone();
        reordered.reverse();
        let first = BinaryCalibrationReportV1::from_predictions(&predictions, 0.5, 2).unwrap();
        let second = BinaryCalibrationReportV1::from_predictions(&reordered, 0.5, 2).unwrap();

        assert_eq!(first.average_precision, Some(0.5));
        assert_eq!(first.average_precision, second.average_precision);

        let selective_predictions = [
            ("a", 0.1, false),
            ("b", 0.4, true),
            ("c", 0.6, false),
            ("d", 0.9, true),
        ]
        .into_iter()
        .map(
            |(observation_id, probability_failure, label_failure)| BinaryPredictionV1 {
                observation_id: observation_id.into(),
                group_id: format!("group-{observation_id}"),
                evaluator_release_id: digest('a'),
                calibration_model_id: digest('b'),
                split: CalibrationDataSplitV1::Test,
                probability_failure: Some(probability_failure),
                label_failure,
            },
        )
        .collect::<Vec<_>>();
        let selective =
            BinaryCalibrationReportV1::from_predictions(&selective_predictions, 0.5, 2).unwrap();
        assert_eq!(selective.selective_risk.len(), 3);
        assert_eq!(selective.selective_risk[1].coverage, 0.5);
        assert_eq!(selective.selective_risk[1].classification_risk, Some(0.0));
        assert_eq!(selective.selective_risk[2].coverage, 1.0);
        assert_eq!(selective.selective_risk[2].classification_risk, Some(0.5));
    }
}
