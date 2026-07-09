# Missing Work

This document tracks what is still missing after the composability refactor.

The core library is now composition-ready: deterministic graders and LLM judges can both produce `EvaluationResult`, `EvaluationRun` can collect multiple evaluator passes, `CalibrationModel` can fit and apply simple score calibration, and `WeightedAggregate` can compute a run score.

The crate is not yet product-complete. The missing work is mostly around pipeline stages, CLI coverage, reporting, fixtures, and public API hygiene.

## Current Baseline

Implemented:

- `EvaluationResult` as the canonical result type.
- `Evaluator` for sync evaluators.
- `AsyncEvaluator` for async evaluators.
- `EvaluationRun` for composing cases, results, and aggregate scoring.
- `WeightedAggregate` for evaluator and cluster weighted run scores.
- `CalibrationModel::fit()` for simple bin-based calibration.
- `CalibrationModel::apply()` and `apply_run()` for calibrated result output.
- `GradeResult` and `JudgeResult` conversions into `EvaluationResult`.
- CLI `grade` and `judge` commands that write unified evaluation results.

Still intentionally lightweight:

- No default ML dependency.
- No vector database dependency.
- No embedding provider dependency.
- No persistent project/database format.

## P0: Missing End-to-End CLI Commands

The library now has the composable pieces, but the CLI does not expose the full workflow.

Missing commands:

```text
traceeval extract
traceeval validate
traceeval calibrate
traceeval cluster
traceeval report
```

Required behavior:

- `extract` reads traces and writes `EvalCase` JSONL.
- `validate` checks cases/results for missing IDs, missing outputs, invalid score ranges, duplicate case IDs, and schema drift.
- `calibrate` reads historical `EvaluationResult` JSONL plus `HumanRating` JSONL and writes a `CalibrationModel` JSON file.
- `cluster` assigns cluster IDs to cases/results using metadata-first rules before any ML features.
- `report` reads `EvaluationResult` JSONL and writes a run-level report with totals, pass rates, weighted score, evaluator breakdowns, and cluster breakdowns.

Acceptance criteria:

- A user can run extract -> grade -> calibrate -> report without writing Rust code.
- CLI outputs use the same library structs as the Rust API.
- All commands have fixture-backed tests.

## P0: Missing Report Model

`WeightedAggregate` returns a single `RunScore`, but there is no report object that explains a run.

Add:

```rust
pub struct EvaluationReport {
    pub run_score: RunScore,
    pub evaluator_scores: Vec<EvaluatorScore>,
    pub cluster_scores: Vec<ClusterScore>,
    pub total_cases: usize,
    pub total_results: usize,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

pub struct EvaluatorScore {
    pub evaluator_name: String,
    pub score: RunScore,
}

pub struct ClusterScore {
    pub cluster_id: String,
    pub score: RunScore,
}
```

Required behavior:

- Report generation should group by evaluator name.
- Report generation should group by cluster ID when present.
- Missing cluster IDs should be grouped as `"unclustered"`.
- Reports should serialize to stable JSON.

## P0: Missing Fixture Dataset

The crate needs checked-in fixtures that prove the workflow is useful.

Add fixtures:

```text
fixtures/openinference/traces.jsonl
fixtures/eval/cases.jsonl
fixtures/eval/historical_results.jsonl
fixtures/eval/human_ratings.jsonl
fixtures/eval/new_results.jsonl
fixtures/eval/clusters.jsonl
```

Required coverage:

- At least one simple successful answer.
- At least one wrong answer.
- At least one missing output.
- At least one tool/agent trace.
- At least two task clusters with different expected quality thresholds.

Acceptance criteria:

- `cargo test` exercises the fixtures.
- README examples can be reproduced with fixture files.

## P1: Missing Cluster Assignment API

`EvaluationResult` has `cluster_id`, and `WeightedAggregate` can weight clusters, but there is no first-class cluster assignment stage.

Add:

```rust
pub struct EvalCluster {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub weight: f32,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

pub struct ClusterAssignment {
    pub case_id: String,
    pub trace_id: String,
    pub cluster_id: String,
    pub confidence: f32,
    pub method: String,
}

pub trait Clusterer {
    fn assign_case(&self, case: &EvalCase) -> anyhow::Result<ClusterAssignment>;
}
```

First implementation:

- Metadata-based assignment.
- Optional lexical fallback.
- Unknown cases assigned to `"unclustered"` or `"novel"` with low confidence.

Do not add embeddings until metadata and lexical assignment are implemented and tested.

## P1: Missing Calibration CLI And Persistence

`CalibrationModel` exists in the library, but the CLI cannot create or consume it yet.

Add:

```bash
traceeval calibrate \
  --human-ratings fixtures/eval/human_ratings.jsonl \
  --results fixtures/eval/historical_results.jsonl \
  --pass-threshold 3 \
  --out calibration.json
```

Then extend grading/reporting:

```bash
traceeval report \
  --results evaluation_results.jsonl \
  --calibration calibration.json \
  --out report.json
```

Required behavior:

- Calibration should reject empty overlap between ratings and results.
- Calibration should record evaluator name when all historical results come from one evaluator.
- Applying calibration should be skipped or warned when evaluator names do not match.

## P1: Missing Public API Curation

The library still exposes more structure than ideal.

Current issue:

- `cli` is public because the binary uses `traces_to_evals::cli::run()`.
- `exporters` is a compatibility namespace.
- Provider implementation details are still visible under `providers`.

Target shape:

```rust
pub mod calibration;
pub mod evaluation;
pub mod export;
pub mod extractors;
pub mod graders;
pub mod io;
pub mod judge;
pub mod model;
pub mod report;

pub mod prelude;
```

Potential cleanup:

- Move CLI execution into an internal binary-support module or keep `cli` public but mark it as binary API, not library API.
- Keep `providers` public only if users need to inject provider clients.
- Keep `exporters` for one release, then deprecate it if this becomes a published crate.

## P1: Missing Builder For Full Pipelines

`EvaluationRun` composes results, but it does not yet model the full trace-to-report workflow.

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

This should wait until report and cluster APIs exist. Otherwise it will become a shallow wrapper.

## P2: Missing Better Error Types

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

## P2: Missing ML Feature Layer

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

1. Metadata clusterer.
2. Lexical clusterer.
3. Embedding generation.
4. Local nearest-centroid assignment.
5. K-Means or HDBSCAN-style clustering.
6. Approximate nearest-neighbor index or vector database.

Do not add a broad ML dependency until the non-ML clustering API is stable.

## P2: Missing Compatibility Policy

Before publishing or asking users to depend on this crate, define what is stable.

Needed:

- Decide whether `GradeResult` and `JudgeResult` are compatibility types or long-term domain types.
- Decide whether `ScoredResult` remains as an alias or is removed before release.
- Decide whether serialized JSON fields are stable.
- Add changelog entries for schema-affecting changes.

## Definition Of Useful

The crate becomes clearly useful when this works with checked-in fixtures:

```bash
traceeval extract \
  --format openinference \
  --traces fixtures/openinference/traces.jsonl \
  --out target/tmp/cases.jsonl

traceeval grade \
  --cases target/tmp/cases.jsonl \
  --grader exact-match \
  --out target/tmp/evaluation_results.jsonl

traceeval calibrate \
  --human-ratings fixtures/eval/human_ratings.jsonl \
  --results fixtures/eval/historical_results.jsonl \
  --out target/tmp/calibration.json

traceeval report \
  --results target/tmp/evaluation_results.jsonl \
  --calibration target/tmp/calibration.json \
  --clusters fixtures/eval/clusters.jsonl \
  --out target/tmp/report.json
```

The report should answer:

- How good was this run overall?
- Which evaluator produced the score?
- Which clusters are weak?
- How much did calibration change the score?
- Which cases failed and why?

## Recommended Next Build Order

1. Add fixtures.
2. Add `EvaluationReport`.
3. Add `traceeval report`.
4. Add `traceeval calibrate`.
5. Add metadata-based clustering.
6. Add `traceeval extract`.
7. Add validation.
8. Revisit public API visibility.
9. Add optional ML features only after the non-ML workflow is useful.
