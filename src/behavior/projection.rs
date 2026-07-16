use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::model::EvalCase;

use super::BehaviorFinding;

pub const FINDING_PROJECTION_SCHEMA_VERSION: &str = "traceeval.finding_projection.v1";
pub const DEFAULT_FINDING_PROJECTION_VERSION: &str = "traceeval.finding_text.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FindingProjection {
    pub schema_version: String,
    pub finding_id: String,
    pub trace_id: String,
    pub projection_version: String,
    pub text: String,
    pub text_hash: String,
    pub truncated: bool,
    pub included_fields: Vec<String>,
}

pub trait FindingRedactor: Send + Sync {
    fn project_value(&self, field: &str, value: &Value) -> Option<String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ScalarFindingRedactor;

impl FindingRedactor for ScalarFindingRedactor {
    fn project_value(&self, _field: &str, value: &Value) -> Option<String> {
        match value {
            Value::String(value) => Some(single_line(value)),
            Value::Number(value) => Some(value.to_string()),
            Value::Bool(value) => Some(value.to_string()),
            Value::Null | Value::Array(_) | Value::Object(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafeFindingProjector {
    projection_version: String,
    allowed_metadata_fields: BTreeSet<String>,
    max_bytes: usize,
}

impl Default for SafeFindingProjector {
    fn default() -> Self {
        Self {
            projection_version: DEFAULT_FINDING_PROJECTION_VERSION.to_string(),
            allowed_metadata_fields: [
                "subject",
                "tool_name",
                "operation",
                "error_kind",
                "policy_id",
                "reason_code",
                "approval_outcome",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            max_bytes: 4_096,
        }
    }
}

impl SafeFindingProjector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_projection_version(mut self, version: impl Into<String>) -> Self {
        self.projection_version = version.into();
        self
    }

    pub fn with_allowed_metadata_field(mut self, field: impl Into<String>) -> Self {
        self.allowed_metadata_fields.insert(field.into());
        self
    }

    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes.max(1);
        self
    }

    pub fn projection_version(&self) -> &str {
        &self.projection_version
    }

    pub fn project(&self, finding: &BehaviorFinding) -> FindingProjection {
        self.project_with_context_and_redactor(finding, &BTreeMap::new(), &ScalarFindingRedactor)
    }

    pub fn project_with_context(
        &self,
        finding: &BehaviorFinding,
        context: &BTreeMap<String, Value>,
    ) -> FindingProjection {
        self.project_with_context_and_redactor(finding, context, &ScalarFindingRedactor)
    }

    pub fn project_with_context_and_redactor(
        &self,
        finding: &BehaviorFinding,
        context: &BTreeMap<String, Value>,
        redactor: &dyn FindingRedactor,
    ) -> FindingProjection {
        let mut fields = BTreeMap::from([
            ("detector".to_string(), finding.detector_id.clone()),
            ("kind".to_string(), finding.kind.clone()),
            (
                "severity".to_string(),
                enum_name(&finding.severity).unwrap_or_else(|| "unknown".to_string()),
            ),
            (
                "recovery".to_string(),
                enum_name(&finding.recovery).unwrap_or_else(|| "unknown".to_string()),
            ),
        ]);

        for field in &self.allowed_metadata_fields {
            if forbidden_projection_field(field) {
                continue;
            }
            let value = finding.metadata.get(field).or_else(|| context.get(field));
            let Some(value) = value.and_then(|value| redactor.project_value(field, value)) else {
                continue;
            };
            fields.insert(field.clone(), value);
        }

        let included_fields = fields.keys().cloned().collect::<Vec<_>>();
        let full_text = fields
            .iter()
            .map(|(field, value)| format!("{field}: {value}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (text, truncated) = truncate_utf8(&full_text, self.max_bytes);
        let text_hash = format!("sha256:{:x}", Sha256::digest(text.as_bytes()));

        FindingProjection {
            schema_version: FINDING_PROJECTION_SCHEMA_VERSION.to_string(),
            finding_id: finding.finding_id.clone(),
            trace_id: finding.trace_id.clone(),
            projection_version: self.projection_version.clone(),
            text,
            text_hash,
            truncated,
            included_fields,
        }
    }
}

impl FindingProjection {
    pub fn to_eval_case(&self) -> EvalCase {
        let mut case = EvalCase::new(&self.finding_id, &self.trace_id, &self.text);
        case.metadata.insert(
            "artifact_kind".to_string(),
            Value::String("behavior_finding_projection".to_string()),
        );
        case.metadata.insert(
            "projection_version".to_string(),
            Value::String(self.projection_version.clone()),
        );
        case.metadata.insert(
            "text_hash".to_string(),
            Value::String(self.text_hash.clone()),
        );
        case
    }
}

pub fn finding_projection_cases(projections: &[FindingProjection]) -> Vec<EvalCase> {
    projections
        .iter()
        .map(FindingProjection::to_eval_case)
        .collect()
}

fn enum_name(value: &impl Serialize) -> Option<String> {
    serde_json::to_value(value)
        .ok()?
        .as_str()
        .map(str::to_string)
}

fn forbidden_projection_field(field: &str) -> bool {
    let normalized = field.to_ascii_lowercase();
    [
        "organization_id",
        "project_id",
        "service_id",
        "environment_id",
        "deployment_id",
        "revision_id",
    ]
    .contains(&normalized.as_str())
        || ["secret", "password", "credential", "api_key", "token"]
            .iter()
            .any(|marker| normalized.contains(marker))
}

fn single_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_utf8(value: &str, max_bytes: usize) -> (String, bool) {
    if value.len() <= max_bytes {
        return (value.to_string(), false);
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    (value[..end].to_string(), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::{BEHAVIOR_FINDING_SCHEMA_VERSION, FindingSeverity, RecoveryStatus};

    fn finding() -> BehaviorFinding {
        BehaviorFinding {
            schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
            finding_id: "finding-1".to_string(),
            detector_id: "false_success_claim".to_string(),
            detector_version: "2".to_string(),
            trace_id: "trace-1".to_string(),
            kind: "false_success_claim".to_string(),
            severity: FindingSeverity::High,
            recovery: RecoveryStatus::Unrecovered,
            confidence: Some(1.0),
            certainty: crate::behavior::FindingCertaintyV1::default(),
            failure_signature: "sha256:signature".to_string(),
            evidence: Vec::new(),
            created_at: "2026-07-10T12:00:00Z".to_string(),
            metadata: BTreeMap::from([
                (
                    "tool_name".to_string(),
                    Value::String("cancelCard".to_string()),
                ),
                (
                    "customer_note".to_string(),
                    Value::String("private note".to_string()),
                ),
            ]),
        }
    }

    #[test]
    fn excludes_arbitrary_context_until_explicitly_allowlisted() {
        let finding = finding();
        let context = BTreeMap::from([
            (
                "intent".to_string(),
                Value::String("cancel card".to_string()),
            ),
            (
                "deployment_id".to_string(),
                Value::String("deploy-secret".to_string()),
            ),
        ]);

        let default_projection =
            SafeFindingProjector::default().project_with_context(&finding, &context);
        assert!(!default_projection.text.contains("cancel card"));
        assert!(!default_projection.text.contains("private note"));

        let projection = SafeFindingProjector::default()
            .with_allowed_metadata_field("intent")
            .with_allowed_metadata_field("deployment_id")
            .project_with_context(&finding, &context);
        assert!(projection.text.contains("intent: cancel card"));
        assert!(!projection.text.contains("deploy-secret"));
    }

    #[test]
    fn projection_is_deterministic_bounded_and_hashed() {
        let finding = finding();
        let projector = SafeFindingProjector::default().with_max_bytes(40);

        let first = projector.project(&finding);
        let second = projector.project(&finding);

        assert_eq!(first, second);
        assert!(first.truncated);
        assert!(first.text.len() <= 40);
        assert!(first.text_hash.starts_with("sha256:"));
    }

    struct HashingRedactor;

    impl FindingRedactor for HashingRedactor {
        fn project_value(&self, field: &str, value: &Value) -> Option<String> {
            let value = value.as_str()?;
            Some(format!(
                "{field}:sha256:{:x}",
                Sha256::digest(value.as_bytes())
            ))
        }
    }

    #[test]
    fn supports_caller_supplied_redaction() {
        let finding = finding();
        let projection = SafeFindingProjector::default().project_with_context_and_redactor(
            &finding,
            &BTreeMap::new(),
            &HashingRedactor,
        );

        assert!(!projection.text.contains("cancelCard"));
        assert!(projection.text.contains("tool_name:sha256:"));
    }

    #[test]
    fn converts_safe_projection_to_clusterable_eval_case() {
        let projection = SafeFindingProjector::default().project(&finding());

        let case = projection.to_eval_case();

        assert_eq!(case.id, "finding-1");
        assert_eq!(case.trace_id, "trace-1");
        assert_eq!(case.input, projection.text);
        assert_eq!(
            case.metadata["artifact_kind"],
            "behavior_finding_projection"
        );
    }
}
