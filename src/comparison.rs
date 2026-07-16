use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::SpanKind;

pub const TRACE_COMPARISON_SCHEMA_VERSION: &str = "traceeval.trace_comparison.v1";
pub const TRACE_COMPARISON_ENGINE_VERSION: &str = "bounded-structural-alignment-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub kind: SpanKind,
    pub status_code: i32,
    pub duration_nano: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub facts: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceComparisonInput {
    pub project_id: String,
    pub logical_trace_id: String,
    pub revision: u64,
    pub build_id: Option<String>,
    pub agent_id: Option<String>,
    pub steps: Vec<ExecutionStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentRelation {
    Exact,
    Equivalent,
    Changed,
    BaselineOnly,
    CandidateOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignedExecutionRow {
    pub baseline: Option<ExecutionStep>,
    pub candidate: Option<ExecutionStep>,
    pub relation: AlignmentRelation,
    pub meaningful: bool,
    pub reason: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DivergenceSummary {
    pub alignment_index: u64,
    pub baseline_step_index: u64,
    pub candidate_step_index: u64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceComparison {
    pub schema_version: String,
    pub engine_version: String,
    pub comparison_id: String,
    pub project_id: String,
    pub baseline_trace_id: String,
    pub baseline_revision: u64,
    pub candidate_trace_id: String,
    pub candidate_revision: u64,
    pub common_prefix_steps: u64,
    pub first_meaningful_divergence: Option<DivergenceSummary>,
    pub rows: Vec<AlignedExecutionRow>,
    pub baseline_unmatched_steps: u64,
    pub candidate_unmatched_steps: u64,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceAlignmentOptions {
    pub lookahead: usize,
    pub context_before: usize,
    pub maximum_rows: usize,
}

impl Default for TraceAlignmentOptions {
    fn default() -> Self {
        Self {
            lookahead: 32,
            context_before: 12,
            maximum_rows: 1_024,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StructuralTraceAligner;

impl StructuralTraceAligner {
    pub fn compare(
        &self,
        baseline: &TraceComparisonInput,
        candidate: &TraceComparisonInput,
        options: TraceAlignmentOptions,
    ) -> TraceComparison {
        self.compare_cancellable(baseline, candidate, options, || false)
            .expect("a comparison without cancellation always completes")
    }

    pub fn compare_cancellable(
        &self,
        baseline: &TraceComparisonInput,
        candidate: &TraceComparisonInput,
        options: TraceAlignmentOptions,
        cancelled: impl Fn() -> bool,
    ) -> Option<TraceComparison> {
        let maximum_rows = options.maximum_rows.max(1);
        let lookahead = options.lookahead.max(1);
        let mut baseline_index = 0_usize;
        let mut candidate_index = 0_usize;
        let mut alignment_index = 0_u64;
        let mut common_prefix_steps = 0_u64;
        let mut baseline_unmatched_steps = 0_u64;
        let mut candidate_unmatched_steps = 0_u64;
        let mut first_divergence = None;
        let mut prefix_context = VecDeque::with_capacity(options.context_before);
        let mut rows = Vec::with_capacity(maximum_rows.min(1_024));
        let mut truncated = false;

        while baseline_index < baseline.steps.len() || candidate_index < candidate.steps.len() {
            if cancelled() {
                return None;
            }
            let baseline_start = baseline_index;
            let candidate_start = candidate_index;
            let mut produced = Vec::new();
            match (
                baseline.steps.get(baseline_index),
                candidate.steps.get(candidate_index),
            ) {
                (Some(left), Some(right)) if comparable_key(left) == comparable_key(right) => {
                    produced.push(compare_pair(left, right));
                    baseline_index += 1;
                    candidate_index += 1;
                }
                (Some(left), Some(right)) => {
                    let candidate_match =
                        find_match(left, &candidate.steps, candidate_index + 1, lookahead);
                    let baseline_match =
                        find_match(right, &baseline.steps, baseline_index + 1, lookahead);
                    match (candidate_match, baseline_match) {
                        (Some(candidate_match), Some(baseline_match))
                            if candidate_match - candidate_index
                                <= baseline_match - baseline_index =>
                        {
                            for step in &candidate.steps[candidate_index..candidate_match] {
                                produced.push(one_sided(step, false));
                                candidate_unmatched_steps += 1;
                            }
                            candidate_index = candidate_match;
                        }
                        (Some(candidate_match), None) => {
                            for step in &candidate.steps[candidate_index..candidate_match] {
                                produced.push(one_sided(step, false));
                                candidate_unmatched_steps += 1;
                            }
                            candidate_index = candidate_match;
                        }
                        (_, Some(baseline_match)) => {
                            for step in &baseline.steps[baseline_index..baseline_match] {
                                produced.push(one_sided(step, true));
                                baseline_unmatched_steps += 1;
                            }
                            baseline_index = baseline_match;
                        }
                        (None, None) => {
                            produced.push(changed_pair(
                                left,
                                right,
                                "Operation or topology changed.",
                            ));
                            baseline_index += 1;
                            candidate_index += 1;
                        }
                    }
                }
                (Some(left), None) => {
                    produced.push(one_sided(left, true));
                    baseline_unmatched_steps += 1;
                    baseline_index += 1;
                }
                (None, Some(right)) => {
                    produced.push(one_sided(right, false));
                    candidate_unmatched_steps += 1;
                    candidate_index += 1;
                }
                (None, None) => break,
            }

            for row in produced {
                if first_divergence.is_none() && !row.meaningful {
                    common_prefix_steps += 1;
                    if options.context_before > 0 {
                        if prefix_context.len() == options.context_before {
                            prefix_context.pop_front();
                        }
                        prefix_context.push_back(row);
                    }
                    alignment_index += 1;
                    continue;
                }
                if first_divergence.is_none() {
                    first_divergence = Some(DivergenceSummary {
                        alignment_index,
                        baseline_step_index: baseline_start as u64,
                        candidate_step_index: candidate_start as u64,
                        reason: row
                            .reason
                            .clone()
                            .unwrap_or_else(|| "Execution shape diverged.".into()),
                    });
                    while let Some(prefix) = prefix_context.pop_front() {
                        if rows.len() == maximum_rows {
                            truncated = true;
                            break;
                        }
                        rows.push(prefix);
                    }
                }
                if rows.len() == maximum_rows {
                    truncated = true;
                } else {
                    rows.push(row);
                }
                alignment_index += 1;
            }
        }

        if first_divergence.is_none() {
            rows.extend(prefix_context.into_iter().take(maximum_rows));
        }
        Some(TraceComparison {
            schema_version: TRACE_COMPARISON_SCHEMA_VERSION.into(),
            engine_version: TRACE_COMPARISON_ENGINE_VERSION.into(),
            comparison_id: comparison_id(baseline, candidate),
            project_id: baseline.project_id.clone(),
            baseline_trace_id: baseline.logical_trace_id.clone(),
            baseline_revision: baseline.revision,
            candidate_trace_id: candidate.logical_trace_id.clone(),
            candidate_revision: candidate.revision,
            common_prefix_steps,
            first_meaningful_divergence: first_divergence,
            rows,
            baseline_unmatched_steps,
            candidate_unmatched_steps,
            truncated,
        })
    }
}

fn compare_pair(left: &ExecutionStep, right: &ExecutionStep) -> AlignedExecutionRow {
    let exact_identity = left.span_id == right.span_id;
    let status_changed = left.status_code != right.status_code;
    let facts_changed = left.facts != right.facts;
    if status_changed || facts_changed {
        let reason = if status_changed {
            format!(
                "Status changed from {} to {}.",
                left.status_code, right.status_code
            )
        } else {
            "Behavior facts changed.".into()
        };
        return changed_pair(left, right, &reason);
    }
    AlignedExecutionRow {
        baseline: Some(left.clone()),
        candidate: Some(right.clone()),
        relation: if exact_identity {
            AlignmentRelation::Exact
        } else {
            AlignmentRelation::Equivalent
        },
        meaningful: false,
        reason: None,
        confidence: if exact_identity { 1.0 } else { 0.9 },
    }
}

fn changed_pair(left: &ExecutionStep, right: &ExecutionStep, reason: &str) -> AlignedExecutionRow {
    AlignedExecutionRow {
        baseline: Some(left.clone()),
        candidate: Some(right.clone()),
        relation: AlignmentRelation::Changed,
        meaningful: true,
        reason: Some(reason.into()),
        confidence: 0.75,
    }
}

fn one_sided(step: &ExecutionStep, baseline: bool) -> AlignedExecutionRow {
    AlignedExecutionRow {
        baseline: baseline.then(|| step.clone()),
        candidate: (!baseline).then(|| step.clone()),
        relation: if baseline {
            AlignmentRelation::BaselineOnly
        } else {
            AlignmentRelation::CandidateOnly
        },
        meaningful: true,
        reason: Some(if baseline {
            "Step is absent from the candidate run.".into()
        } else {
            "Step was added in the candidate run.".into()
        }),
        confidence: 0.85,
    }
}

fn find_match(
    needle: &ExecutionStep,
    haystack: &[ExecutionStep],
    start: usize,
    lookahead: usize,
) -> Option<usize> {
    let key = comparable_key(needle);
    haystack
        .iter()
        .enumerate()
        .skip(start)
        .take(lookahead)
        .find_map(|(index, step)| (comparable_key(step) == key).then_some(index))
}

fn comparable_key(step: &ExecutionStep) -> String {
    let operation = step.operation.as_deref().unwrap_or(&step.name);
    format!(
        "{:?}|{}|{}|{}",
        step.kind,
        normalize(operation),
        step.agent_ref.as_deref().map(normalize).unwrap_or_default(),
        step.parent_span_id.is_some()
    )
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn comparison_id(baseline: &TraceComparisonInput, candidate: &TraceComparisonInput) -> String {
    let mut hasher = Sha256::new();
    hasher.update(TRACE_COMPARISON_ENGINE_VERSION.as_bytes());
    hasher.update(baseline.project_id.as_bytes());
    hasher.update(baseline.logical_trace_id.as_bytes());
    hasher.update(baseline.revision.to_le_bytes());
    hasher.update(candidate.logical_trace_id.as_bytes());
    hasher.update(candidate.revision.to_le_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(id: &str, name: &str, status: i32) -> ExecutionStep {
        ExecutionStep {
            span_id: id.into(),
            parent_span_id: Some("root".into()),
            name: name.into(),
            kind: SpanKind::Tool,
            status_code: status,
            duration_nano: 1,
            agent_ref: Some("planner".into()),
            operation: Some(name.into()),
            facts: BTreeMap::new(),
        }
    }

    fn input(id: &str, steps: Vec<ExecutionStep>) -> TraceComparisonInput {
        TraceComparisonInput {
            project_id: "checkout".into(),
            logical_trace_id: id.into(),
            revision: 1,
            build_id: None,
            agent_id: None,
            steps,
        }
    }

    #[test]
    fn late_divergence_is_retained_even_when_the_output_is_bounded() {
        let mut left = (0..10_000)
            .map(|index| step(&format!("l-{index}"), "lookup", 0))
            .collect::<Vec<_>>();
        let mut right = (0..10_000)
            .map(|index| step(&format!("r-{index}"), "lookup", 0))
            .collect::<Vec<_>>();
        left.push(step("left-final", "charge", 0));
        right.push(step("right-final", "charge", 2));

        let comparison = StructuralTraceAligner.compare(
            &input("left", left),
            &input("right", right),
            TraceAlignmentOptions {
                context_before: 3,
                maximum_rows: 8,
                ..TraceAlignmentOptions::default()
            },
        );

        assert_eq!(comparison.common_prefix_steps, 10_000);
        assert!(comparison.first_meaningful_divergence.is_some());
        assert_eq!(comparison.rows.len(), 4);
        assert_eq!(
            comparison.rows.last().unwrap().relation,
            AlignmentRelation::Changed
        );
    }

    #[test]
    fn bounded_lookahead_represents_insertions_without_merging_agents() {
        let left = input("left", vec![step("a", "lookup", 0), step("b", "charge", 0)]);
        let mut added = step("x", "explain", 0);
        added.agent_ref = Some("reviewer".into());
        let right = input(
            "right",
            vec![step("a2", "lookup", 0), added, step("b2", "charge", 0)],
        );

        let comparison =
            StructuralTraceAligner.compare(&left, &right, TraceAlignmentOptions::default());

        assert_eq!(comparison.candidate_unmatched_steps, 1);
        assert!(comparison.rows.iter().any(|row| {
            row.relation == AlignmentRelation::CandidateOnly
                && row.candidate.as_ref().unwrap().agent_ref.as_deref() == Some("reviewer")
        }));
    }

    #[test]
    fn identical_shapes_report_no_meaningful_divergence() {
        let left = input("left", vec![step("a", "lookup", 0)]);
        let right = input("right", vec![step("b", "lookup", 0)]);

        let comparison =
            StructuralTraceAligner.compare(&left, &right, TraceAlignmentOptions::default());

        assert!(comparison.first_meaningful_divergence.is_none());
        assert_eq!(comparison.common_prefix_steps, 1);
    }

    #[test]
    fn comparison_can_be_cancelled_during_alignment() {
        use std::cell::Cell;

        let baseline = input(
            "baseline",
            (0..10_000)
                .map(|index| step(&format!("b-{index}"), "lookup", 0))
                .collect(),
        );
        let candidate = input(
            "candidate",
            (0..10_000)
                .map(|index| step(&format!("c-{index}"), "lookup", 0))
                .collect(),
        );
        let checks = Cell::new(0_u32);
        let comparison = StructuralTraceAligner.compare_cancellable(
            &baseline,
            &candidate,
            TraceAlignmentOptions::default(),
            || {
                checks.set(checks.get() + 1);
                checks.get() > 8
            },
        );

        assert!(comparison.is_none());
        assert!(checks.get() <= 10);
    }
}
