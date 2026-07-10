use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::model::Trace;

use super::{
    AgentBehaviorNormalizer, AgentBehaviorTrace, BehaviorFinding, DeterministicDetectorSet,
    OperationEffect, RetrySafety, ToolCallStatus, ToolRequirement,
};

pub const DETECTION_CHECKPOINT_SCHEMA_VERSION: &str = "traceeval.detection_checkpoint.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEnvelope {
    pub trace: Trace,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_cursor: Option<String>,
}

impl TraceEnvelope {
    pub fn new(trace: Trace) -> Self {
        Self {
            trace,
            source_cursor: None,
        }
    }

    pub fn with_source_cursor(mut self, source_cursor: impl Into<String>) -> Self {
        self.source_cursor = Some(source_cursor.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingWriteStatus {
    Written,
    AlreadyPresent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectionCheckpoint {
    pub schema_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_cursor: Option<String>,
    pub trace_id: String,
    pub processed_trace_count: u64,
    pub written_finding_count: u64,
    pub skipped_finding_count: u64,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectionRunStats {
    pub processed_trace_count: u64,
    pub normalized_tool_call_count: u64,
    pub total_tool_duration_ms: u64,
    pub written_finding_count: u64,
    pub skipped_finding_count: u64,
    pub unknown_tool_status_count: u64,
    pub unknown_semantics_count: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub findings_by_detector: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub detector_versions: BTreeMap<String, String>,
}

#[async_trait]
pub trait AgentTraceSource: Send {
    async fn next_trace(&mut self) -> Result<Option<TraceEnvelope>>;
}

/// Sink implementations should durably write a finding and perform associated
/// event emission before the write method returns. A checkpoint is requested
/// only after every finding for the source trace has completed.
#[async_trait]
pub trait BehaviorFindingSink: Send {
    async fn write_behavior(&mut self, _behavior: &AgentBehaviorTrace) -> Result<()> {
        Ok(())
    }

    async fn write_finding(&mut self, finding: BehaviorFinding) -> Result<FindingWriteStatus>;

    async fn checkpoint(&mut self, checkpoint: DetectionCheckpoint) -> Result<()>;
}

pub struct DetectionRunner<'a> {
    normalizer: &'a dyn AgentBehaviorNormalizer,
    detectors: &'a DeterministicDetectorSet,
    completed_finding_ids: BTreeSet<String>,
}

impl<'a> DetectionRunner<'a> {
    pub fn new(
        normalizer: &'a dyn AgentBehaviorNormalizer,
        detectors: &'a DeterministicDetectorSet,
    ) -> Self {
        Self {
            normalizer,
            detectors,
            completed_finding_ids: BTreeSet::new(),
        }
    }

    pub fn with_completed_finding_ids(
        mut self,
        finding_ids: impl IntoIterator<Item = String>,
    ) -> Self {
        self.completed_finding_ids.extend(finding_ids);
        self
    }

    pub async fn run<S, K>(&mut self, source: &mut S, sink: &mut K) -> Result<DetectionRunStats>
    where
        S: AgentTraceSource,
        K: BehaviorFindingSink,
    {
        let mut stats = DetectionRunStats {
            detector_versions: self.detectors.detector_versions(),
            ..DetectionRunStats::default()
        };
        while let Some(envelope) = source.next_trace().await? {
            let behavior = self.normalizer.normalize(&envelope.trace)?;
            stats.processed_trace_count += 1;
            stats.normalized_tool_call_count += behavior.tool_calls.len() as u64;
            stats.total_tool_duration_ms += behavior
                .tool_calls
                .iter()
                .map(|call| call.duration_ms)
                .sum::<u64>();
            stats.unknown_tool_status_count += behavior
                .tool_calls
                .iter()
                .filter(|call| call.status == ToolCallStatus::Unknown)
                .count() as u64;
            stats.unknown_semantics_count += behavior
                .tool_calls
                .iter()
                .filter(|call| {
                    call.effect == OperationEffect::Unknown
                        || call.retry_safety == RetrySafety::Unknown
                        || call.requirement == ToolRequirement::Unknown
                })
                .count() as u64;

            let findings = self.detectors.detect(&behavior);
            sink.write_behavior(&behavior).await?;
            for finding in findings {
                if self.completed_finding_ids.contains(&finding.finding_id) {
                    stats.skipped_finding_count += 1;
                    continue;
                }
                let detector_id = finding.detector_id.clone();
                let finding_id = finding.finding_id.clone();
                match sink.write_finding(finding).await? {
                    FindingWriteStatus::Written => {
                        stats.written_finding_count += 1;
                        *stats.findings_by_detector.entry(detector_id).or_default() += 1;
                    }
                    FindingWriteStatus::AlreadyPresent => {
                        stats.skipped_finding_count += 1;
                    }
                }
                self.completed_finding_ids.insert(finding_id);
            }

            sink.checkpoint(DetectionCheckpoint {
                schema_version: DETECTION_CHECKPOINT_SCHEMA_VERSION.to_string(),
                source_cursor: envelope.source_cursor,
                trace_id: behavior.trace_id,
                processed_trace_count: stats.processed_trace_count,
                written_finding_count: stats.written_finding_count,
                skipped_finding_count: stats.skipped_finding_count,
            })
            .await?;
        }
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};

    use futures::executor::block_on;
    use serde_json::json;

    use crate::behavior::OpenInferenceBehaviorNormalizer;
    use crate::model::{Span, SpanKind};

    use super::*;

    struct VecTraceSource {
        traces: VecDeque<TraceEnvelope>,
    }

    #[async_trait]
    impl AgentTraceSource for VecTraceSource {
        async fn next_trace(&mut self) -> Result<Option<TraceEnvelope>> {
            Ok(self.traces.pop_front())
        }
    }

    #[derive(Default)]
    struct RecordingSink {
        finding_ids: Vec<String>,
        checkpoints: Vec<DetectionCheckpoint>,
        events: Vec<String>,
        fail_on_finding: bool,
    }

    #[async_trait]
    impl BehaviorFindingSink for RecordingSink {
        async fn write_behavior(&mut self, behavior: &AgentBehaviorTrace) -> Result<()> {
            self.events.push(format!("behavior:{}", behavior.trace_id));
            Ok(())
        }

        async fn write_finding(&mut self, finding: BehaviorFinding) -> Result<FindingWriteStatus> {
            self.events.push(format!("finding:{}", finding.finding_id));
            if self.fail_on_finding {
                return Err(crate::TraceEvalError::Provider(
                    "simulated finding sink failure".to_string(),
                ));
            }
            self.finding_ids.push(finding.finding_id);
            Ok(FindingWriteStatus::Written)
        }

        async fn checkpoint(&mut self, checkpoint: DetectionCheckpoint) -> Result<()> {
            self.events
                .push(format!("checkpoint:{}", checkpoint.trace_id));
            self.checkpoints.push(checkpoint);
            Ok(())
        }
    }

    fn failing_trace() -> Trace {
        let mut root_attributes = BTreeMap::new();
        root_attributes.insert("agent.final.status".to_string(), json!("completed"));
        root_attributes.insert(
            "agent.outcome.claims".to_string(),
            json!([{
                "operation": "cancel_card",
                "call_id": "call-1",
                "status": "succeeded"
            }]),
        );
        root_attributes.insert("agent.escalation.status".to_string(), json!("not_required"));
        let root = Span {
            id: "root".to_string(),
            trace_id: Some("trace-1".to_string()),
            parent_id: None,
            name: "agent".to_string(),
            kind: SpanKind::Agent,
            input: Some("cancel my card".to_string()),
            output: Some("done".to_string()),
            error: None,
            started_at: Some("2026-07-10T12:00:00Z".to_string()),
            ended_at: Some("2026-07-10T12:00:02Z".to_string()),
            attributes: root_attributes,
        };
        let mut tool_attributes = BTreeMap::new();
        tool_attributes.insert("gen_ai.tool.name".to_string(), json!("cancelCard"));
        tool_attributes.insert("gen_ai.tool.call.id".to_string(), json!("call-1"));
        tool_attributes.insert("agent.operation".to_string(), json!("cancel_card"));
        tool_attributes.insert("agent.operation.effect".to_string(), json!("mutating"));
        tool_attributes.insert(
            "agent.operation.retry_safety".to_string(),
            json!("non_idempotent"),
        );
        tool_attributes.insert("agent.tool.requirement".to_string(), json!("required"));
        tool_attributes.insert("agent.tool.status".to_string(), json!("timed_out"));
        tool_attributes.insert("agent.state.observation".to_string(), json!("ambiguous"));
        let tool = Span {
            id: "tool".to_string(),
            trace_id: Some("trace-1".to_string()),
            parent_id: Some("root".to_string()),
            name: "tool".to_string(),
            kind: SpanKind::Tool,
            input: None,
            output: None,
            error: Some("timed out".to_string()),
            started_at: Some("2026-07-10T12:00:01Z".to_string()),
            ended_at: Some("2026-07-10T12:00:02Z".to_string()),
            attributes: tool_attributes,
        };
        Trace::new("trace-1").with_span(root).with_span(tool)
    }

    #[test]
    fn duplicate_trace_delivery_writes_findings_once_and_checkpoints_after_writes() {
        let trace = failing_trace();
        let mut source = VecTraceSource {
            traces: VecDeque::from([
                TraceEnvelope::new(trace.clone()).with_source_cursor("offset-1"),
                TraceEnvelope::new(trace).with_source_cursor("offset-2"),
            ]),
        };
        let normalizer = OpenInferenceBehaviorNormalizer::default();
        let detectors = DeterministicDetectorSet::default();
        let mut runner = DetectionRunner::new(&normalizer, &detectors);
        let mut sink = RecordingSink::default();

        let stats = block_on(runner.run(&mut source, &mut sink)).unwrap();

        assert_eq!(stats.processed_trace_count, 2);
        assert_eq!(stats.written_finding_count, 3);
        assert_eq!(stats.skipped_finding_count, 3);
        assert_eq!(stats.detector_versions.len(), 10);
        assert_eq!(sink.finding_ids.len(), 3);
        assert_eq!(sink.checkpoints.len(), 2);
        assert_eq!(
            sink.events.last().map(String::as_str),
            Some("checkpoint:trace-1")
        );
        assert_eq!(
            sink.checkpoints[1].source_cursor.as_deref(),
            Some("offset-2")
        );
    }

    #[test]
    fn finding_write_failure_prevents_source_checkpoint() {
        let mut source = VecTraceSource {
            traces: VecDeque::from([
                TraceEnvelope::new(failing_trace()).with_source_cursor("offset-1")
            ]),
        };
        let normalizer = OpenInferenceBehaviorNormalizer::default();
        let detectors = DeterministicDetectorSet::default();
        let mut runner = DetectionRunner::new(&normalizer, &detectors);
        let mut sink = RecordingSink {
            fail_on_finding: true,
            ..RecordingSink::default()
        };

        let result = block_on(runner.run(&mut source, &mut sink));

        assert!(result.is_err());
        assert!(sink.checkpoints.is_empty());
    }
}
