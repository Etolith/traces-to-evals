use super::*;

pub(super) fn finding_for_call(
    trace: &AgentBehaviorTrace,
    detector: &dyn TraceDetector,
    severity: FindingSeverity,
    recovery: RecoveryStatus,
    call: &ToolCallFact,
    mut metadata: BTreeMap<String, Value>,
) -> BehaviorFinding {
    metadata.insert("tool_name".to_string(), json!(call.tool_name));
    if let Some(operation) = &call.operation {
        metadata.insert("operation".to_string(), json!(operation));
    }
    metadata.insert("call_id".to_string(), json!(call.call_id));
    build_finding(
        trace,
        detector,
        severity,
        recovery,
        signature_subject(call),
        error_kind(call),
        call.evidence.clone(),
        metadata,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_finding(
    trace: &AgentBehaviorTrace,
    detector: &dyn TraceDetector,
    severity: FindingSeverity,
    recovery: RecoveryStatus,
    signature_subject: (String, Option<String>),
    error_kind: Option<String>,
    mut evidence: Vec<EvidenceRef>,
    mut metadata: BTreeMap<String, Value>,
) -> BehaviorFinding {
    metadata
        .entry("subject".to_string())
        .or_insert_with(|| json!(&signature_subject.0));
    if let Some(operation) = &signature_subject.1 {
        metadata
            .entry("operation".to_string())
            .or_insert_with(|| json!(operation));
    }
    if let Some(error_kind) = &error_kind {
        metadata
            .entry("error_kind".to_string())
            .or_insert_with(|| json!(error_kind));
    }
    for key in [
        "traceeval.behavior_adapter.id",
        "traceeval.behavior_adapter.version",
    ] {
        if let Some(value) = trace.metadata.get(key) {
            metadata
                .entry(key.to_string())
                .or_insert_with(|| value.clone());
        }
    }
    evidence.sort_by(|left, right| left.identity.cmp(&right.identity));
    evidence.dedup_by(|left, right| left.identity == right.identity);
    let evidence_ids = evidence
        .iter()
        .map(|evidence| evidence.identity.as_str())
        .collect::<Vec<_>>();
    let finding_id = hash_parts(
        [trace.trace_id.as_str(), detector.id(), detector.version()]
            .into_iter()
            .chain(evidence_ids),
    );
    let failure_signature = hash_parts([
        detector.id(),
        signature_subject.0.as_str(),
        signature_subject.1.as_deref().unwrap_or(""),
        error_kind.as_deref().unwrap_or(""),
    ]);
    let created_at = trace
        .metadata
        .get("observed_at")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            trace
                .tool_calls
                .last()
                .map(|call| call.started_at.clone())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string());

    BehaviorFinding {
        schema_version: BEHAVIOR_FINDING_SCHEMA_VERSION.to_string(),
        finding_id,
        detector_id: detector.id().to_string(),
        detector_version: detector.version().to_string(),
        trace_id: trace.trace_id.clone(),
        kind: detector.id().to_string(),
        severity,
        recovery,
        confidence: None,
        certainty: finding_certainty(trace, detector.id()),
        failure_signature,
        evidence,
        created_at,
        metadata,
    }
}

fn finding_certainty(trace: &AgentBehaviorTrace, detector_id: &str) -> FindingCertaintyV1 {
    if matches!(detector_id, "repeated_tool_failure" | "tool_call_loop") {
        return FindingCertaintyV1 {
            rule_match: RuleMatchCertaintyV1::Exact,
            semantic_coverage: 1.0,
            missing_facts: Vec::new(),
            calibrated_failure_risk: None,
        };
    }
    let mut missing_facts = Vec::new();
    if trace.coverage.explicit_status_spans == 0 {
        missing_facts.push("source_status".to_string());
    }
    if trace.coverage.operation_identity == FactQuality::Missing {
        missing_facts.push("operation_identity".to_string());
    }
    if trace.coverage.final_outcome == FactQuality::Missing {
        missing_facts.push("final_outcome".to_string());
    }
    let observed = 3usize.saturating_sub(missing_facts.len());
    let semantic_coverage = observed as f32 / 3.0;
    let rule_match = if missing_facts.is_empty() {
        RuleMatchCertaintyV1::Exact
    } else if observed == 0 {
        RuleMatchCertaintyV1::Inconclusive
    } else {
        RuleMatchCertaintyV1::BoundedInference
    };
    FindingCertaintyV1 {
        rule_match,
        semantic_coverage,
        missing_facts,
        calibrated_failure_risk: None,
    }
}

pub(super) fn signature_subject(call: &ToolCallFact) -> (String, Option<String>) {
    (call.tool_name.clone(), call.operation.clone())
}

pub(super) fn error_kind(call: &ToolCallFact) -> Option<String> {
    call.error.as_ref().map(|error| error.kind.clone())
}

fn hash_parts<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.len().to_be_bytes());
        hasher.update(part.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}
