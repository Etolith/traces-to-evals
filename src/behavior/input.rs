use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::model::{FactQuality, SourceSpanStatus, SpanKind, Trace};
use crate::{Result, TraceEvalError};

pub const BEHAVIOR_INPUT_SCHEMA_VERSION: &str = "traceeval.behavior_input.v1";
pub const SAFE_BEHAVIOR_PROJECTION_VERSION: &str = "traceeval.safe_behavior_projection.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BehaviorInputPrivacyV1 {
    SafeIdentitiesOnly,
    LegacyInline,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BehaviorInputProvenanceV1 {
    pub projection_version: String,
    pub source_id: String,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub decoder_versions: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub semantic_mapping_versions: BTreeSet<String>,
}

impl BehaviorInputProvenanceV1 {
    pub fn legacy() -> Self {
        Self {
            projection_version: "traceeval.legacy_trace_projection.v1".into(),
            source_id: "legacy".into(),
            decoder_versions: BTreeSet::new(),
            semantic_mapping_versions: BTreeSet::new(),
        }
    }
}

impl Default for BehaviorInputProvenanceV1 {
    fn default() -> Self {
        Self::legacy()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BehaviorInputCoverageV1 {
    pub span_count: u64,
    pub explicit_status_spans: u64,
    pub numeric_timestamp_spans: u64,
    pub payload_identity_spans: u64,
    pub event_count: u64,
    pub link_count: u64,
    #[serde(default)]
    pub operation_identity: FactQuality,
    #[serde(default)]
    pub final_outcome: FactQuality,
    #[serde(default)]
    pub verifier: FactQuality,
}

impl BehaviorInputCoverageV1 {
    pub fn from_trace(trace: &Trace) -> Self {
        let mut coverage = Self {
            span_count: trace.spans.len() as u64,
            ..Self::default()
        };
        let mut operation = false;
        let mut final_outcome = false;
        let mut verifier = false;
        for span in &trace.spans {
            coverage.explicit_status_spans = coverage
                .explicit_status_spans
                .saturating_add(u64::from(span.source_status != SourceSpanStatus::Unset));
            coverage.numeric_timestamp_spans = coverage.numeric_timestamp_spans.saturating_add(
                u64::from(span.start_time_unix_nano.is_some() && span.end_time_unix_nano.is_some()),
            );
            coverage.payload_identity_spans = coverage
                .payload_identity_spans
                .saturating_add(u64::from(!span.payload_identities.is_empty()));
            coverage.event_count = coverage
                .event_count
                .saturating_add(span.events.len() as u64);
            coverage.link_count = coverage.link_count.saturating_add(span.links.len() as u64);
            operation |= [
                "gen_ai.operation.name",
                "agent.operation",
                "tool.operation",
                "operation",
                "operation.name",
            ]
            .iter()
            .any(|key| span.attributes.contains_key(*key));
            final_outcome |= ["agent.final.status", "final.status", "gen_ai.output.type"]
                .iter()
                .any(|key| span.attributes.contains_key(*key));
            verifier |= span.kind == SpanKind::Evaluator
                || span
                    .attributes
                    .get("agent.role")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|role| role.eq_ignore_ascii_case("verifier"));
        }
        coverage.operation_identity = explicit_or_missing(operation);
        coverage.final_outcome = explicit_or_missing(final_outcome);
        coverage.verifier = explicit_or_missing(verifier);
        coverage
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehaviorInputV1 {
    pub schema_version: String,
    pub privacy: BehaviorInputPrivacyV1,
    pub provenance: BehaviorInputProvenanceV1,
    pub coverage: BehaviorInputCoverageV1,
    pub trace: Trace,
}

impl BehaviorInputV1 {
    pub fn safe(trace: Trace, provenance: BehaviorInputProvenanceV1) -> Result<Self> {
        let input = Self {
            schema_version: BEHAVIOR_INPUT_SCHEMA_VERSION.into(),
            privacy: BehaviorInputPrivacyV1::SafeIdentitiesOnly,
            coverage: BehaviorInputCoverageV1::from_trace(&trace),
            provenance,
            trace,
        };
        input.validate()?;
        Ok(input)
    }

    pub fn legacy(trace: Trace) -> Self {
        Self {
            schema_version: BEHAVIOR_INPUT_SCHEMA_VERSION.into(),
            privacy: BehaviorInputPrivacyV1::LegacyInline,
            coverage: BehaviorInputCoverageV1::from_trace(&trace),
            provenance: BehaviorInputProvenanceV1::legacy(),
            trace,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let invalid = |message: &str| TraceEvalError::InvalidBehaviorInput {
            trace_id: self.trace.id.clone(),
            message: message.into(),
        };
        if self.schema_version != BEHAVIOR_INPUT_SCHEMA_VERSION {
            return Err(invalid("unsupported behavior input schema version"));
        }
        if self.trace.id.trim().is_empty() {
            return Err(invalid("trace identity cannot be empty"));
        }
        if self.provenance.projection_version.trim().is_empty()
            || self.provenance.source_id.trim().is_empty()
        {
            return Err(invalid("projection and source provenance are required"));
        }
        for span in &self.trace.spans {
            if span.id.trim().is_empty() {
                return Err(invalid("span identity cannot be empty"));
            }
            if self.privacy == BehaviorInputPrivacyV1::SafeIdentitiesOnly
                && (span.input.is_some() || span.output.is_some() || span.error.is_some())
            {
                return Err(invalid(
                    "safe behavior input cannot contain inline input, output, or error bodies",
                ));
            }
            if self.privacy == BehaviorInputPrivacyV1::SafeIdentitiesOnly
                && span
                    .attributes
                    .keys()
                    .any(|key| is_private_payload_attribute(key))
            {
                return Err(invalid(
                    "safe behavior input contains an inline private payload attribute",
                ));
            }
            if let (Some(start), Some(end)) = (span.start_time_unix_nano, span.end_time_unix_nano) {
                if end < start {
                    return Err(invalid("span end timestamp precedes its start timestamp"));
                }
                if let Some(duration) = span.duration_nano
                    && duration.abs_diff(end - start) > 1_000_000
                {
                    return Err(invalid(
                        "span duration differs from source timestamps by more than 1 ms",
                    ));
                }
            }
            for (key, payload) in &span.payload_identities {
                if key.trim().is_empty() || payload.quality == FactQuality::Missing {
                    return Err(invalid(
                        "payload identities require a key and non-missing quality",
                    ));
                }
                if !is_sha256_identity(&payload.fingerprint) {
                    return Err(invalid(
                        "payload fingerprint is not an opaque SHA-256 identity",
                    ));
                }
            }
            if span
                .events
                .iter()
                .any(|event| event.identity.trim().is_empty())
            {
                return Err(invalid("span event evidence identity cannot be empty"));
            }
            if span.links.iter().any(|link| {
                link.identity.trim().is_empty()
                    || link.trace_id.trim().is_empty()
                    || link.span_id.trim().is_empty()
            }) {
                return Err(invalid("span link evidence identities cannot be empty"));
            }
        }
        Ok(())
    }
}

fn is_private_payload_attribute(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "input"
            | "input.value"
            | "output"
            | "output.value"
            | "prompt"
            | "prompt.value"
            | "reasoning"
            | "reasoning.content"
            | "source.code"
            | "code"
            | "tool.arguments"
            | "tool.result"
            | "gen_ai.tool.call.arguments"
            | "gen_ai.tool.call.result"
            | "gen_ai.prompt"
            | "gen_ai.completion"
            | "llm.input_messages"
            | "llm.output_messages"
    )
}

fn explicit_or_missing(present: bool) -> FactQuality {
    if present {
        FactQuality::Explicit
    } else {
        FactQuality::Missing
    }
}

fn is_sha256_identity(value: &str) -> bool {
    let Some(value) = value.strip_prefix("sha256:") else {
        return false;
    };
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::model::{PayloadIdentity, Span};

    use super::*;

    #[test]
    fn safe_input_rejects_inline_private_payloads() {
        let trace =
            Trace::new("trace-1").with_span(Span::new("span-1", "tool").with_input("secret"));

        assert!(BehaviorInputV1::safe(trace, BehaviorInputProvenanceV1::legacy()).is_err());
    }

    #[test]
    fn safe_input_preserves_numeric_time_status_and_payload_identity() {
        let mut span = Span::new("span-1", "tool");
        span.source_status = SourceSpanStatus::Ok;
        span.start_time_unix_nano = Some(10);
        span.end_time_unix_nano = Some(20);
        span.duration_nano = Some(10);
        span.payload_identities = BTreeMap::from([(
            "input.value".into(),
            PayloadIdentity {
                fingerprint:
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
                blob_id: Some("blob-1".into()),
                original_bytes: 42,
                quality: FactQuality::Explicit,
            },
        )]);
        let input = BehaviorInputV1::safe(
            Trace::new("trace-1").with_span(span),
            BehaviorInputProvenanceV1::legacy(),
        )
        .unwrap();

        assert_eq!(input.coverage.explicit_status_spans, 1);
        assert_eq!(input.coverage.numeric_timestamp_spans, 1);
        assert_eq!(input.coverage.payload_identity_spans, 1);
    }

    #[test]
    fn safe_input_rejects_known_inline_payload_attributes() {
        let mut span = Span::new("span-1", "tool");
        span.attributes.insert(
            "gen_ai.tool.call.arguments".into(),
            serde_json::json!({"secret": 7}),
        );

        assert!(
            BehaviorInputV1::safe(
                Trace::new("trace-1").with_span(span),
                BehaviorInputProvenanceV1::legacy(),
            )
            .is_err()
        );
    }
}
