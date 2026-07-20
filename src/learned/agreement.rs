use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{ContractError, require_non_empty};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgreementRatingV1 {
    pub item_id: String,
    pub rater_id: String,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgreementLabelScaleV1 {
    /// Stable label order. For nominal reports this still defines the complete
    /// allowlist; for weighted kappa it also defines ordinal distance.
    pub labels: Vec<String>,
    pub ordinal: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanAgreementReportV1 {
    pub item_count: u64,
    pub pairable_item_count: u64,
    pub rating_count: u64,
    pub rater_count: u64,
    pub disagreement_item_count: u64,
    pub label_counts: BTreeMap<String, u64>,
    /// Nominal Krippendorff alpha over all items with at least two ratings.
    pub krippendorff_alpha: Option<f64>,
    /// Available only for an exact two-rater panel with both raters present on
    /// every pairable item.
    pub cohen_kappa: Option<f64>,
    /// Quadratic-weighted kappa for an ordinal two-rater panel.
    pub quadratic_weighted_kappa: Option<f64>,
}

impl HumanAgreementReportV1 {
    pub fn from_ratings(
        ratings: &[AgreementRatingV1],
        scale: &AgreementLabelScaleV1,
    ) -> Result<Self, ContractError> {
        validate_scale(scale)?;
        if ratings.is_empty() {
            return Err(agreement_error("at least one human rating is required"));
        }
        let allowed = scale
            .labels
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        let mut by_item = BTreeMap::<&str, Vec<&AgreementRatingV1>>::new();
        let mut raters = BTreeSet::new();
        let mut label_counts = scale
            .labels
            .iter()
            .cloned()
            .map(|label| (label, 0_u64))
            .collect::<BTreeMap<_, _>>();
        for rating in ratings {
            require_non_empty(&rating.item_id, "item_id", agreement_error)?;
            require_non_empty(&rating.rater_id, "rater_id", agreement_error)?;
            require_non_empty(&rating.label, "label", agreement_error)?;
            if !allowed.contains(rating.label.as_str()) {
                return Err(agreement_error(format!(
                    "rating label {} is absent from the frozen scale",
                    rating.label
                )));
            }
            if !seen.insert((rating.item_id.as_str(), rating.rater_id.as_str())) {
                return Err(agreement_error(format!(
                    "rater {} submitted more than one current rating for item {}",
                    rating.rater_id, rating.item_id
                )));
            }
            by_item.entry(&rating.item_id).or_default().push(rating);
            raters.insert(rating.rater_id.as_str());
            *label_counts
                .get_mut(&rating.label)
                .expect("validated label allowlist") += 1;
        }
        for item_ratings in by_item.values_mut() {
            item_ratings.sort_by(|left, right| left.rater_id.cmp(&right.rater_id));
        }
        let pairable = by_item
            .values()
            .filter(|item_ratings| item_ratings.len() >= 2)
            .collect::<Vec<_>>();
        let disagreement_item_count = pairable
            .iter()
            .filter(|item_ratings| {
                item_ratings
                    .iter()
                    .skip(1)
                    .any(|rating| rating.label != item_ratings[0].label)
            })
            .count() as u64;
        let krippendorff_alpha = nominal_krippendorff_alpha(&pairable);
        let panel = (pairable.len() == by_item.len())
            .then(|| exact_two_rater_panel(&pairable, &raters))
            .flatten();
        let cohen_kappa = panel
            .as_ref()
            .and_then(|panel| cohen_kappa(panel, &scale.labels));
        let quadratic_weighted_kappa = (scale.ordinal)
            .then_some(panel.as_ref())
            .flatten()
            .and_then(|panel| quadratic_weighted_kappa(panel, &scale.labels));
        Ok(Self {
            item_count: by_item.len() as u64,
            pairable_item_count: pairable.len() as u64,
            rating_count: ratings.len() as u64,
            rater_count: raters.len() as u64,
            disagreement_item_count,
            label_counts,
            krippendorff_alpha,
            cohen_kappa,
            quadratic_weighted_kappa,
        })
    }
}

fn validate_scale(scale: &AgreementLabelScaleV1) -> Result<(), ContractError> {
    if scale.labels.len() < 2 {
        return Err(agreement_error(
            "agreement scale requires at least two labels",
        ));
    }
    let mut unique = BTreeSet::new();
    for label in &scale.labels {
        require_non_empty(label, "agreement label", agreement_error)?;
        if !unique.insert(label.as_str()) {
            return Err(agreement_error(format!(
                "duplicate agreement label {label}"
            )));
        }
    }
    Ok(())
}

fn nominal_krippendorff_alpha(pairable: &[&Vec<&AgreementRatingV1>]) -> Option<f64> {
    let observed_pairs = pairable
        .iter()
        .map(|ratings| ratings.len() * (ratings.len() - 1))
        .sum::<usize>();
    if observed_pairs == 0 {
        return None;
    }
    let observed_disagreements = pairable
        .iter()
        .map(|ratings| {
            ratings
                .iter()
                .enumerate()
                .flat_map(|(left_index, left)| {
                    ratings
                        .iter()
                        .enumerate()
                        .filter_map(move |(right_index, right)| {
                            (left_index != right_index && left.label != right.label)
                                .then_some(1_u64)
                        })
                })
                .sum::<u64>()
        })
        .sum::<u64>();
    let observed = observed_disagreements as f64 / observed_pairs as f64;
    let mut label_counts = BTreeMap::<&str, u64>::new();
    for rating in pairable.iter().flat_map(|ratings| ratings.iter()) {
        *label_counts.entry(rating.label.as_str()).or_default() += 1;
    }
    let total = label_counts.values().sum::<u64>();
    if total < 2 {
        return None;
    }
    let expected_disagreements = label_counts
        .values()
        .map(|count| count * (total - count))
        .sum::<u64>();
    let expected = expected_disagreements as f64 / (total * (total - 1)) as f64;
    (expected > 0.0).then(|| 1.0 - observed / expected)
}

fn exact_two_rater_panel<'a>(
    pairable: &[&Vec<&'a AgreementRatingV1>],
    raters: &BTreeSet<&str>,
) -> Option<Vec<(&'a str, &'a str)>> {
    if raters.len() != 2 || pairable.is_empty() {
        return None;
    }
    let ordered_raters = raters.iter().copied().collect::<Vec<_>>();
    let mut panel = Vec::with_capacity(pairable.len());
    for ratings in pairable {
        if ratings.len() != 2
            || ratings[0].rater_id != ordered_raters[0]
            || ratings[1].rater_id != ordered_raters[1]
        {
            return None;
        }
        panel.push((ratings[0].label.as_str(), ratings[1].label.as_str()));
    }
    Some(panel)
}

fn cohen_kappa(panel: &[(&str, &str)], labels: &[String]) -> Option<f64> {
    let total = panel.len() as f64;
    let observed = panel.iter().filter(|(left, right)| left == right).count() as f64 / total;
    let mut left_counts = BTreeMap::<&str, u64>::new();
    let mut right_counts = BTreeMap::<&str, u64>::new();
    for (left, right) in panel {
        *left_counts.entry(left).or_default() += 1;
        *right_counts.entry(right).or_default() += 1;
    }
    let expected = labels
        .iter()
        .map(|label| {
            left_counts.get(label.as_str()).copied().unwrap_or(0) as f64
                * right_counts.get(label.as_str()).copied().unwrap_or(0) as f64
                / total.powi(2)
        })
        .sum::<f64>();
    ((1.0 - expected).abs() > f64::EPSILON).then(|| (observed - expected) / (1.0 - expected))
}

fn quadratic_weighted_kappa(panel: &[(&str, &str)], labels: &[String]) -> Option<f64> {
    let label_index = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let maximum_distance = (labels.len() - 1).pow(2) as f64;
    let total = panel.len() as f64;
    let observed_disagreement = panel
        .iter()
        .map(|(left, right)| {
            let left = *label_index.get(left).expect("validated panel label");
            let right = *label_index.get(right).expect("validated panel label");
            left.abs_diff(right).pow(2) as f64 / maximum_distance
        })
        .sum::<f64>()
        / total;
    let mut left_counts = vec![0_u64; labels.len()];
    let mut right_counts = vec![0_u64; labels.len()];
    for (left, right) in panel {
        left_counts[*label_index.get(left).expect("validated panel label")] += 1;
        right_counts[*label_index.get(right).expect("validated panel label")] += 1;
    }
    let mut expected_disagreement = 0.0;
    for (left, left_count) in left_counts.iter().enumerate() {
        for (right, right_count) in right_counts.iter().enumerate() {
            let weight = left.abs_diff(right).pow(2) as f64 / maximum_distance;
            expected_disagreement +=
                weight * (*left_count as f64 / total) * (*right_count as f64 / total);
        }
    }
    (expected_disagreement > 0.0).then(|| 1.0 - observed_disagreement / expected_disagreement)
}

fn agreement_error(message: impl Into<String>) -> ContractError {
    ContractError::InvalidCalibration(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rating(item: &str, rater: &str, label: &str) -> AgreementRatingV1 {
        AgreementRatingV1 {
            item_id: item.into(),
            rater_id: rater.into(),
            label: label.into(),
        }
    }

    fn scale(ordinal: bool) -> AgreementLabelScaleV1 {
        AgreementLabelScaleV1 {
            labels: vec!["completed".into(), "partial".into(), "failed".into()],
            ordinal,
        }
    }

    #[test]
    fn perfect_two_rater_panel_has_unit_agreement() {
        let report = HumanAgreementReportV1::from_ratings(
            &[
                rating("a", "r1", "completed"),
                rating("a", "r2", "completed"),
                rating("b", "r1", "failed"),
                rating("b", "r2", "failed"),
            ],
            &scale(true),
        )
        .unwrap();
        assert_eq!(report.krippendorff_alpha, Some(1.0));
        assert_eq!(report.cohen_kappa, Some(1.0));
        assert_eq!(report.quadratic_weighted_kappa, Some(1.0));
        assert_eq!(report.disagreement_item_count, 0);
    }

    #[test]
    fn disagreements_and_missing_ratings_are_counted_without_fabrication() {
        let report = HumanAgreementReportV1::from_ratings(
            &[
                rating("a", "r1", "completed"),
                rating("a", "r2", "failed"),
                rating("b", "r1", "partial"),
            ],
            &scale(true),
        )
        .unwrap();
        assert_eq!(report.item_count, 2);
        assert_eq!(report.pairable_item_count, 1);
        assert_eq!(report.disagreement_item_count, 1);
        assert!(report.krippendorff_alpha.unwrap() <= 0.0);
        assert_eq!(report.cohen_kappa, None);
        assert_eq!(report.quadratic_weighted_kappa, None);
    }

    #[test]
    fn duplicate_current_rating_and_unknown_label_fail_closed() {
        let duplicate = HumanAgreementReportV1::from_ratings(
            &[rating("a", "r1", "completed"), rating("a", "r1", "failed")],
            &scale(false),
        )
        .unwrap_err();
        assert!(
            duplicate
                .to_string()
                .contains("more than one current rating")
        );
        let unknown =
            HumanAgreementReportV1::from_ratings(&[rating("a", "r1", "abstain")], &scale(false))
                .unwrap_err();
        assert!(unknown.to_string().contains("frozen scale"));
    }

    #[test]
    fn report_is_input_order_invariant() {
        let ratings = vec![
            rating("a", "r1", "completed"),
            rating("a", "r2", "partial"),
            rating("b", "r1", "failed"),
            rating("b", "r2", "failed"),
        ];
        let mut reversed = ratings.clone();
        reversed.reverse();
        assert_eq!(
            HumanAgreementReportV1::from_ratings(&ratings, &scale(true)).unwrap(),
            HumanAgreementReportV1::from_ratings(&reversed, &scale(true)).unwrap()
        );
    }
}
