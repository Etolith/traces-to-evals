# Missing Work

This document tracks what remains after the composability and CLI workflow implementation.

The crate now has a usable non-ML evaluation loop:

```text
traces
  -> extract EvalCase JSONL
  -> validate cases/results
  -> grade into EvaluationResult JSONL
  -> calibrate from historical results + human ratings
  -> assign clusters
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
- `EvalCluster`, `ClusterAssignment`, `Clusterer`, and `MetadataClusterer`.
- `ValidationReport` and fixture-backed validation checks.
- `GradeResult` and `JudgeResult` conversions into `EvaluationResult`.

CLI:

- `traceeval extract`
- `traceeval grade`
- `traceeval judge`
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

`EvaluationReport` explains overall, per-evaluator, and per-cluster scores, but it is still minimal.

Missing:

- List of failed cases.
- Worst clusters sorted by score.
- Score deltas before/after calibration.
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

The library still exposes more structure than ideal.

Current issue:

- `cli` is public because the binary uses `traces_to_evals::cli::run()`.
- `exporters` is a compatibility namespace.
- Provider implementation details are still visible under `providers`.

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

- Move CLI execution into an internal binary-support module or keep `cli` public but mark it as binary API, not library API.
- Keep `providers` public only if users need to inject provider clients.
- Keep `exporters` for one release, then deprecate it if this becomes a published crate.

## P1: Full Pipeline Builder

`EvaluationRun` composes cases and results, but it does not model the entire trace-to-report workflow.

Potential API:

```rust
let report = TraceEval::new()
    .extract_with(OpenInferenceExtractor)
    .evaluate_with(ExactMatchGrader)
    .calibrate_with(calibration_model)
    .cluster_with(clusterer)
    .aggregate_with(weights)
    .run_traces(traces)
    .await?;
```

This should now be feasible because report and cluster APIs exist. It should be implemented only if it removes real ceremony from common workflows.

## P1: Calibration Semantics

Calibration is currently simple score-bin calibration.

Missing:

- Explicit warnings when applying a model to a different evaluator.
- Per-cluster calibration.
- Persisted calibration metadata such as created_at, source dataset names, and compared row count.
- Calibration quality in `EvaluationReport`.

Do not add a general optimizer until binned calibration proves insufficient.

## P1: Validation Semantics

Validation currently checks structural and score-range issues.

Missing:

- Optional validation profiles, because some export-only cases may not require `actual_output`.
- Schema version checks.
- Configurable severity: error vs warning.
- Validation of calibration/report files.

## P2: Typed Errors

The crate currently uses `anyhow::Result` broadly. That is acceptable while the API is changing, but a reusable library eventually needs typed errors for predictable caller behavior.

Add typed errors for:

- Invalid case data.
- Missing required output.
- Invalid score scale.
- Calibration overlap failure.
- Cluster assignment failure.
- Provider failure.
- Export failure.

This can be done with `thiserror` when the public API stabilizes.

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

1. Improve metadata clusterer.
2. Improve lexical clusterer.
3. Add embedding generation.
4. Add local nearest-centroid assignment.
5. Add K-Means or HDBSCAN-style clustering.
6. Add approximate nearest-neighbor index or vector database.

Do not add a broad ML dependency until the non-ML clustering API is stable.

## P2: Compatibility Policy

Before publishing or asking users to depend on this crate, define what is stable.

Needed:

- Decide whether `GradeResult` and `JudgeResult` are compatibility types or long-term domain types.
- Decide whether `ScoredResult` remains as an alias or is removed before release.
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
