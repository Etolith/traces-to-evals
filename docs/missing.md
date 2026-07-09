# Missing Work

This document tracks what remains after the composability and CLI workflow implementation.

For the detailed API/product cleanup plan, see [api-and-product-roadmap.md](api-and-product-roadmap.md).

The crate now has a usable non-ML evaluation loop:

```text
traces
  -> extract EvalCase JSONL
  -> validate cases/results
  -> grade into EvaluationResult JSONL
  -> calibrate from historical results + human ratings
  -> assign known cluster IDs
  -> report aggregate quality
```

The default crate is still intentionally lightweight. It has no default ML dependency, vector database dependency, embedding provider dependency, or persistent project database.

## Implemented

Core API:

- `EvaluationResult` as the canonical result type.
- `Evaluator` for sync evaluators.
- `AsyncEvaluator` for async evaluators.
- `EvaluationRun` for composing cases, results, and aggregate scoring.
- `WeightedAggregate` for evaluator and cluster weighted run scores.
- `EvaluationReport`, `EvaluatorScore`, and `ClusterScore`.
- `CalibrationModel::fit()` for simple bin-based calibration.
- `CalibrationModel::apply()` and `apply_run()` for calibrated result output.
- `EvalCluster`, `ClusterAssignment`, `ClusterAssigner`, and `RuleBasedClusterAssigner`.
- `ClusterAssignmentRule`, `MetadataAssignmentRule`, `KeywordAssignmentRule`, and `FnClusterAssignmentRule`.
- `ValidationReport` and fixture-backed validation checks.
- `GradeResult` and `JudgeResult` conversions into `EvaluationResult`.

CLI:

- `traceeval extract`
- `traceeval grade`
- `traceeval validate`
- `traceeval calibrate`
- `traceeval cluster`
- `traceeval report`

Fixtures:

- `fixtures/openinference/traces.jsonl`
- `fixtures/eval/cases.jsonl`
- `fixtures/eval/historical_results.jsonl`
- `fixtures/eval/human_ratings.jsonl`
- `fixtures/eval/new_results.jsonl`
- `fixtures/eval/clusters.jsonl`

Tests:

- Public API shape integration test.
- Fixture-backed CLI workflow test covering extract -> validate -> grade -> calibrate -> cluster -> report.
- Validation failure test for missing output cases.

## Remaining P0

There is no longer a blocking P0 item for a non-ML workflow. The crate can run the basic end-to-end loop from checked-in fixtures.

## P1: Report Quality

`EvaluationReport` now explains overall, per-evaluator, and per-cluster scores, failed cases, worst clusters, and calibration impact.

Implemented:

- List of failed cases.
- Worst clusters sorted by score.
- Score deltas before/after calibration.

Missing:

- Per-cluster case counts split by evaluator.
- Human-readable Markdown report output.

Useful next API:

```rust
pub struct FailedCase {
    pub case_id: String,
    pub trace_id: String,
    pub evaluator_name: String,
    pub cluster_id: Option<String>,
    pub score: f32,
    pub evaluation: String,
}
```

## P1: Public API Curation

The library still exposes more structure than ideal, but compatibility aliases and compatibility namespaces have been removed before release.

Current issue:

- `cli` is public because the binary uses `traces_to_evals::cli::run()`, but it is `#[doc(hidden)]`.
- Provider implementation details are still public under `providers`, but hidden from generated docs.

Target public shape:

```rust
pub mod calibration;
pub mod clustering;
pub mod evaluation;
pub mod export;
pub mod extractors;
pub mod graders;
pub mod io;
pub mod judge;
pub mod model;
pub mod report;
pub mod validation;

pub mod prelude;
```

Potential cleanup:

- Move CLI execution into an internal binary-support module if we want to remove public `cli` entirely.
- Keep `providers` public only if users need to inject provider clients.
- Keep the public API on canonical modules instead of adding pre-release compatibility shims.

## P1: Full Pipeline Builder

`EvaluationRun` composes cases and results, but it does not model the entire trace-to-report workflow.

Potential API:

```rust
let report = TraceEval::new()
    .extract_with(OpenInferenceExtractor)
    .evaluate_with(ExactMatchGrader)
    .calibrate_with(calibration_model)
    .assign_clusters_with(assigner)
    .aggregate_with(weights)
    .run_traces(traces)
    .await?;
```

This should now be feasible because report and cluster assignment APIs exist. It should be implemented only if it removes real ceremony from common workflows.

## P1: Real Cluster Discovery

The current implementation is rule-based assignment to an existing cluster taxonomy. It is not data-driven clustering.

Implemented:

- Exact assignment from metadata fields such as `cluster_id`.
- Keyword/lexical fallback against provided cluster definitions.
- `unclustered` fallback for unknown cases.

Missing:

- A separate `ClusterDiscovery` trait for fitting clusters from cases.
- A `ClusterModel` produced from feature vectors or embeddings.
- A separate `ClusterLabeler` trait for naming and describing discovered clusters.
- Evaluation of cluster quality.
- Optional embedding-based and ML-backed implementations.

Potential future API:

```rust
pub trait ClusterDiscovery {
    fn fit(&self, cases: &[EvalCase]) -> anyhow::Result<ClusterModel>;
}

pub struct ClusterModel {
    pub clusters: Vec<DiscoveredCluster>,
    pub assignments: Vec<ClusterAssignment>,
}

pub trait ClusterLabeler {
    async fn label_cluster(
        &self,
        cluster: &DiscoveredCluster,
        examples: &[EvalCase],
    ) -> anyhow::Result<ClusterLabel>;
}

pub struct ClusterLabel {
    pub label: String,
    pub description: String,
    pub suggested_rubric: Option<String>,
    pub known_failure_modes: Vec<String>,
}
```

Keep these responsibilities separate:

- `ClusterDiscovery` creates groups from historical cases.
- `ClusterLabeler` names and explains those groups, commonly with an LLM.
- `ClusterAssigner` maps new cases into a known taxonomy or discovered cluster model.

Expected future flow:

```text
historical EvalCase rows
  -> embeddings
  -> K-Means/HDBSCAN/BERTopic-style discovery
  -> representative examples per cluster
  -> LLM cluster labeling
  -> human approval/editing
  -> cluster-aware scoring and reporting
```

## P1: Calibration Semantics

Calibration is currently simple score-bin calibration.

Missing:

- Explicit warnings when applying a model to a different evaluator.
- Per-cluster calibration.
- Persisted calibration metadata such as created_at, source dataset names, and compared row count.
- Calibration quality in `EvaluationReport`.

Do not add a general optimizer until binned calibration proves insufficient.

## P1: Validation Semantics

Validation now supports profiles and warning severity.

Implemented:

- `DraftCases`
- `RunnableCases`
- `EvaluationResults`
- `CalibrationDataset`
- `ValidationSeverity::Error`
- `ValidationSeverity::Warning`
- CLI `traceeval validate --profile ...`

Missing:

- Schema version checks.
- Validation of calibration/report files.
- Direct validation of human-rating files as part of calibration datasets.

## P2: Typed Errors

The crate now exposes `traces_to_evals::Result<T>` and `TraceEvalError` for core library APIs. CLI command handlers still use `anyhow::Result` for command-specific context, and provider internals still use `anyhow` in a few places.

Implemented typed categories include:

- Invalid case data.
- Missing required output.
- Invalid score scale.
- Calibration overlap failure.
- Cluster assignment failure.
- Validation failure.
- Export failure.

Remaining work:

- Decide whether provider internals should map all errors into `TraceEvalError::Provider`.
- Review variant names and payloads before a stable release.
- Add more branch-on-error tests for `io`, `export`, and OpenAI judge paths.

## P2: ML Feature Layer

ML should stay optional and feature-gated.

Possible features:

```toml
embeddings-openai = [...]
embeddings-fastembed = [...]
clustering-linfa = [...]
vector-hnsw = [...]
vector-qdrant = [...]
```

Recommended order:

1. Improve rule-based assignment.
2. Add a separate discovery trait.
3. Add embedding generation.
4. Add local nearest-centroid assignment.
5. Add K-Means or HDBSCAN-style clustering.
6. Add approximate nearest-neighbor index or vector database.

Do not add a broad ML dependency until the non-ML assignment API is stable.

## P2: Compatibility Policy

Before publishing or asking users to depend on this crate, define what is stable.

Needed:

- Decide whether `GradeResult` and `JudgeResult` are long-term domain types or internal conversion records.
- Decide whether serialized JSON fields are stable.
- Add changelog entries for schema-affecting changes.

## Current Definition Of Useful

This sequence is now covered by integration tests:

```bash
traceeval extract \
  --format openinference \
  --traces fixtures/openinference/traces.jsonl \
  --out target/tmp/cases.jsonl

traceeval validate \
  --cases target/tmp/cases.jsonl \
  --out target/tmp/validation.json

traceeval grade \
  --cases target/tmp/cases.jsonl \
  --grader non-empty-output \
  --out target/tmp/evaluation_results.jsonl

traceeval calibrate \
  --human-ratings fixtures/eval/human_ratings.jsonl \
  --results fixtures/eval/historical_results.jsonl \
  --out target/tmp/calibration.json

traceeval cluster \
  --cases target/tmp/cases.jsonl \
  --clusters fixtures/eval/clusters.jsonl \
  --out target/tmp/cluster_assignments.jsonl \
  --results target/tmp/evaluation_results.jsonl \
  --results-out target/tmp/clustered_results.jsonl

traceeval report \
  --results target/tmp/clustered_results.jsonl \
  --calibration target/tmp/calibration.json \
  --clusters fixtures/eval/clusters.jsonl \
  --out target/tmp/report.json
```

The report answers:

- How good was this run overall?
- Which evaluator produced the score?
- Which clusters are weak?
- Which cases are unclustered?

It does not yet explain failed cases or calibration deltas in enough detail.

## Recommended Next Build Order

1. Add failed-case detail to `EvaluationReport`.
2. Add report Markdown output.
3. Add validation profiles.
4. Add evaluator mismatch warnings for calibration.
5. Revisit public API visibility.
6. Add a `TraceEval` pipeline builder if it clearly simplifies common usage.
7. Add typed errors.
8. Add optional ML features only after the non-ML workflow is stable.
