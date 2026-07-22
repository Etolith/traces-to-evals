use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{ContractError, require_non_empty, require_sha256};

pub const LEARNED_EVALUATION_SCHEMA_VERSION: &str = "traceeval.learned_evaluation.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnedVerdictV1 {
    Pass,
    Fail,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnedAbstentionReasonV1 {
    ContextUnresolved,
    ContextInsufficient,
    ContentUnavailable,
    ContentTruncated,
    PrivacyBlocked,
    EvidenceInsufficient,
    OutOfDistribution,
    ProviderUnavailable,
    InvalidProviderOutput,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationCriterionV1 {
    pub criterion_id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_keys: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationEvidenceKindV1 {
    Span,
    Event,
    TraceAttribute,
    SpanAttribute,
    InputSegment,
    OutputSegment,
    StateObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "location_kind", rename_all = "snake_case")]
pub enum EvaluationEvidenceLocationV1 {
    Span {
        span_id: String,
    },
    Event {
        span_id: String,
        event_id: String,
    },
    TraceAttribute {
        attribute_path: String,
    },
    SpanAttribute {
        span_id: String,
        attribute_path: String,
    },
    Segment {
        span_id: String,
        start_byte: u32,
        end_byte: u32,
    },
}

impl EvaluationEvidenceLocationV1 {
    fn validate(&self) -> Result<(), ContractError> {
        match self {
            Self::Span { span_id } => require_non_empty(span_id, "span_id", evaluation_error),
            Self::Event { span_id, event_id } => {
                require_non_empty(span_id, "span_id", evaluation_error)?;
                require_non_empty(event_id, "event_id", evaluation_error)
            }
            Self::TraceAttribute { attribute_path } => {
                require_non_empty(attribute_path, "attribute_path", evaluation_error)
            }
            Self::SpanAttribute {
                span_id,
                attribute_path,
            } => {
                require_non_empty(span_id, "span_id", evaluation_error)?;
                require_non_empty(attribute_path, "attribute_path", evaluation_error)
            }
            Self::Segment {
                span_id,
                start_byte,
                end_byte,
            } => {
                require_non_empty(span_id, "span_id", evaluation_error)?;
                if start_byte >= end_byte {
                    return Err(evaluation_error(
                        "evidence segment start_byte must be before end_byte",
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationEvidenceRecordV1 {
    pub target_key: String,
    pub target_revision: String,
    pub projection_hash: String,
    pub evidence_kind: EvaluationEvidenceKindV1,
    pub location: EvaluationEvidenceLocationV1,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub applicable_criterion_ids: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationEvidenceCitationV1 {
    pub evidence_key: String,
    pub evidence_kind: EvaluationEvidenceKindV1,
    pub location: EvaluationEvidenceLocationV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criterion_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationEvidenceCatalogV1 {
    pub target_key: String,
    pub target_revision: String,
    pub projection_hash: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub entries: BTreeMap<String, EvaluationEvidenceRecordV1>,
}

impl EvaluationEvidenceCatalogV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        require_non_empty(&self.target_key, "catalog target_key", evaluation_error)?;
        require_non_empty(
            &self.target_revision,
            "catalog target_revision",
            evaluation_error,
        )?;
        require_sha256(
            &self.projection_hash,
            "catalog projection_hash",
            evaluation_error,
        )?;
        for (key, record) in &self.entries {
            require_non_empty(key, "catalog evidence_key", evaluation_error)?;
            if record.target_key != self.target_key
                || record.target_revision != self.target_revision
                || record.projection_hash != self.projection_hash
            {
                return Err(evaluation_error(format!(
                    "evidence record {key} crosses the catalog target revision or projection"
                )));
            }
            record.location.validate()?;
            validate_evidence_kind_location(record.evidence_kind, &record.location)?;
            for criterion_id in &record.applicable_criterion_ids {
                require_non_empty(criterion_id, "applicable criterion_id", evaluation_error)?;
            }
        }
        Ok(())
    }
}

fn validate_evidence_kind_location(
    kind: EvaluationEvidenceKindV1,
    location: &EvaluationEvidenceLocationV1,
) -> Result<(), ContractError> {
    let compatible = matches!(
        (kind, location),
        (
            EvaluationEvidenceKindV1::Span,
            EvaluationEvidenceLocationV1::Span { .. }
        ) | (
            EvaluationEvidenceKindV1::Event,
            EvaluationEvidenceLocationV1::Event { .. }
        ) | (
            EvaluationEvidenceKindV1::TraceAttribute,
            EvaluationEvidenceLocationV1::TraceAttribute { .. }
        ) | (
            EvaluationEvidenceKindV1::SpanAttribute,
            EvaluationEvidenceLocationV1::SpanAttribute { .. }
        ) | (
            EvaluationEvidenceKindV1::InputSegment | EvaluationEvidenceKindV1::OutputSegment,
            EvaluationEvidenceLocationV1::Segment { .. }
        ) | (
            EvaluationEvidenceKindV1::StateObservation,
            EvaluationEvidenceLocationV1::TraceAttribute { .. }
                | EvaluationEvidenceLocationV1::SpanAttribute { .. }
        )
    );
    if !compatible {
        return Err(evaluation_error(
            "evidence kind is incompatible with its typed location",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedEvaluationV1 {
    pub schema_version: String,
    pub evaluator_release_id: String,
    pub target_key: String,
    pub target_revision: String,
    pub trace_context_binding_id: String,
    pub projection_hash: String,
    pub verdict: LearnedVerdictV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_reported_confidence: Option<f64>,
    pub explanation: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvaluationEvidenceCitationV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub criteria: Vec<EvaluationCriterionV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abstention_reason: Option<LearnedAbstentionReasonV1>,
}

impl LearnedEvaluationV1 {
    pub fn validate_against(
        &self,
        catalog: &EvaluationEvidenceCatalogV1,
    ) -> Result<(), ContractError> {
        catalog.validate()?;
        if self.schema_version != LEARNED_EVALUATION_SCHEMA_VERSION {
            return Err(evaluation_error(
                "unsupported learned evaluation schema version",
            ));
        }
        require_sha256(
            &self.evaluator_release_id,
            "evaluator_release_id",
            evaluation_error,
        )?;
        require_sha256(
            &self.trace_context_binding_id,
            "trace_context_binding_id",
            evaluation_error,
        )?;
        if self.target_key != catalog.target_key
            || self.target_revision != catalog.target_revision
            || self.projection_hash != catalog.projection_hash
        {
            return Err(evaluation_error(
                "evaluation target or projection does not match evidence catalog",
            ));
        }
        require_non_empty(&self.explanation, "explanation", evaluation_error)?;
        if self.explanation.chars().count() > 4_000 {
            return Err(evaluation_error("explanation exceeds 4000 characters"));
        }
        validate_probability(self.score, "score")?;
        validate_probability(self.model_reported_confidence, "model_reported_confidence")?;

        match self.verdict {
            LearnedVerdictV1::Pass | LearnedVerdictV1::Fail => {
                if self.abstention_reason.is_some() {
                    return Err(evaluation_error(
                        "pass/fail evaluation cannot include an abstention reason",
                    ));
                }
                if self.label.as_deref().is_none_or(str::is_empty) || self.score.is_none() {
                    return Err(evaluation_error(
                        "pass/fail evaluation requires a non-empty label and score",
                    ));
                }
            }
            LearnedVerdictV1::Abstain => {
                if self.abstention_reason.is_none() {
                    return Err(evaluation_error(
                        "abstention requires a typed abstention reason",
                    ));
                }
            }
        }

        if self.verdict == LearnedVerdictV1::Fail && self.evidence.is_empty() {
            return Err(evaluation_error("failed evaluation requires evidence"));
        }

        let mut criterion_ids = BTreeSet::new();
        for criterion in &self.criteria {
            require_non_empty(&criterion.criterion_id, "criterion_id", evaluation_error)?;
            require_non_empty(&criterion.label, "criterion label", evaluation_error)?;
            if !criterion_ids.insert(criterion.criterion_id.as_str()) {
                return Err(evaluation_error(format!(
                    "duplicate criterion {}",
                    criterion.criterion_id
                )));
            }
            validate_probability(criterion.score, "criterion score")?;
        }

        let mut cited_pairs = BTreeSet::new();
        for citation in &self.evidence {
            require_non_empty(&citation.evidence_key, "evidence_key", evaluation_error)?;
            citation.location.validate()?;
            let record = catalog.entries.get(&citation.evidence_key).ok_or_else(|| {
                evaluation_error(format!("unknown evidence key {}", citation.evidence_key))
            })?;
            if record.evidence_kind != citation.evidence_kind
                || record.location != citation.location
            {
                return Err(evaluation_error(format!(
                    "evidence citation {} does not match its catalog record",
                    citation.evidence_key
                )));
            }
            if let Some(criterion_id) = &citation.criterion_id {
                if !criterion_ids.contains(criterion_id.as_str()) {
                    return Err(evaluation_error(format!(
                        "evidence cites unknown criterion {criterion_id}"
                    )));
                }
                if !record.applicable_criterion_ids.is_empty()
                    && !record.applicable_criterion_ids.contains(criterion_id)
                {
                    return Err(evaluation_error(format!(
                        "evidence {} is not applicable to criterion {criterion_id}",
                        citation.evidence_key
                    )));
                }
            }
            if !cited_pairs.insert((
                citation.evidence_key.as_str(),
                citation.criterion_id.as_deref(),
            )) {
                return Err(evaluation_error(format!(
                    "duplicate evidence citation {}",
                    citation.evidence_key
                )));
            }
        }

        for criterion in &self.criteria {
            for evidence_key in &criterion.evidence_keys {
                let record = catalog.entries.get(evidence_key).ok_or_else(|| {
                    evaluation_error(format!(
                        "criterion {} cites unknown evidence key {evidence_key}",
                        criterion.criterion_id
                    ))
                })?;
                if !record.applicable_criterion_ids.is_empty()
                    && !record
                        .applicable_criterion_ids
                        .contains(&criterion.criterion_id)
                {
                    return Err(evaluation_error(format!(
                        "evidence {evidence_key} is not applicable to criterion {}",
                        criterion.criterion_id
                    )));
                }
                if !self.evidence.iter().any(|citation| {
                    citation.evidence_key == *evidence_key
                        && citation.criterion_id.as_deref() == Some(&criterion.criterion_id)
                }) {
                    return Err(evaluation_error(format!(
                        "criterion {} evidence {evidence_key} lacks a matching typed citation",
                        criterion.criterion_id
                    )));
                }
            }
        }
        Ok(())
    }
}

fn validate_probability(value: Option<f64>, field: &str) -> Result<(), ContractError> {
    if let Some(value) = value
        && (!value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(evaluation_error(format!(
            "{field} must be finite and between 0 and 1"
        )));
    }
    Ok(())
}

fn evaluation_error(message: impl Into<String>) -> ContractError {
    ContractError::InvalidEvaluation(message.into())
}
