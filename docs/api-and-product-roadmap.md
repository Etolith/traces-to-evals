# API And Product Roadmap

This document defines the next cleanup and productization work for `traces-to-evals`.

The crate is now composition-ready, but it is not yet API-stable. The work below should happen before publishing a stable crate or asking external users to build against it.

## 1. Public API Curation

Current issue:

- `cli` is still public because the binary currently calls `traces_to_evals::cli::run()`, but it is hidden from generated public docs.
- `providers` exposes implementation details needed by the OpenAI adapter and test injection, and is hidden from generated public docs.
- Compatibility namespaces and aliases were removed before release:
  - `exporters`
  - `scoring`
  - `Clusterer`
  - `MetadataClusterer`

Target public modules:

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

Policy:

- `prelude` should contain the recommended user-facing API only.
- Compatibility aliases should not be added before the first release.
- `cli` should be treated as binary API, not library API.
- `providers` should be public only if users need direct dependency injection.
- Canonical modules should be preferred over compatibility namespaces.

Proposed implementation steps:

1. Move binary execution code behind `pub(crate)` command modules where possible. Partially done: command modules are private, but `cli` remains public for the binary entrypoint.
2. Keep a small public `cli` module only if downstream users realistically embed the CLI. Partially done: `cli` is `#[doc(hidden)]`.
3. Remove compatibility aliases and namespaces before release. Done for clustering aliases, `exporters`, and `scoring`.
4. Add API-shape integration tests for `prelude`. Done.
5. Before a stable release, avoid adding aliases unless there is an actual external compatibility contract.

Acceptance criteria:

- A downstream user can build common workflows from `prelude`.
- Internal command/provider plumbing does not dominate generated public docs. Partially done.
- Compatibility aliases and namespaces are not part of the public API. Done.

## 2. Typed Errors

Current issue:

- Core library APIs now return `traces_to_evals::Result<T>`.
- CLI command handlers still return `anyhow::Result` so they can add command-specific context.
- Provider-specific internals still use `anyhow` in a few places.
- The typed error surface is useful, but not yet fully curated for a stable release.

Target:

```rust
pub type Result<T> = std::result::Result<T, TraceEvalError>;

#[derive(Debug, thiserror::Error)]
pub enum TraceEvalError {
    #[error("invalid eval case {case_id}: {message}")]
    InvalidCase { case_id: String, message: String },

    #[error("missing required actual_output for case {case_id}")]
    MissingActualOutput { case_id: String },

    #[error("missing required expected_output for case {case_id}")]
    MissingExpectedOutput { case_id: String },

    #[error("invalid score {score} for scale {scale}")]
    InvalidScore { score: f32, scale: String },

    #[error("invalid threshold {threshold} for scale {scale}")]
    InvalidThreshold { threshold: u8, scale: String },

    #[error("cannot calibrate without overlapping case IDs")]
    CalibrationOverlap,

    #[error("cluster assignment failed for case {case_id}: {message}")]
    ClusterAssignment { case_id: String, message: String },

    #[error("provider error: {0}")]
    Provider(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Csv(#[from] csv::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
```

Policy:

- CLI code may continue to use `anyhow`.
- Library/domain modules should move toward typed errors.
- Typed errors should be introduced at module boundaries first, not everywhere at once.

Proposed implementation order:

1. Add `src/error.rs` with `TraceEvalError` and `Result<T>`. Done.
2. Convert validation, calibration, evaluation, extractors, deterministic graders, evaluation traits, clustering, `io`, and `export` helpers. Done.
3. Keep CLI command handlers on `anyhow` for user-facing context. Done.
4. Convert or wrap provider-specific async internals where useful. Not done.
5. Review error variants before a stable release. Not done.

Acceptance criteria:

- Library users can match on common failure categories. Done for missing outputs, extraction failures, invalid scores, validation failures, and calibration overlap.
- Error messages remain clear in the CLI. Preserved by keeping command handlers on `anyhow`.
- No large mechanical rewrite without behavior tests. Partially done with API-shape tests for typed errors.

## 3. Richer Reports

Current issue:

- `EvaluationReport` now has overall score, evaluator scores, cluster scores, failed cases, weak clusters, and calibration impact.

Remaining issue:

- Report output is JSON-only.
- It does not yet have a human-readable Markdown format.
- It does not yet include per-cluster case counts split by evaluator.

Target additions:

```rust
pub struct FailedCase {
    pub case_id: String,
    pub trace_id: String,
    pub evaluator_name: String,
    pub cluster_id: Option<String>,
    pub score: f32,
    pub calibrated_score: Option<f32>,
    pub evaluation: String,
}

pub struct CalibrationImpact {
    pub uncalibrated_score: f32,
    pub calibrated_score: f32,
    pub delta: f32,
}

pub struct ClusterIssue {
    pub cluster_id: String,
    pub score: RunScore,
    pub failed_cases: Vec<FailedCase>,
}
```

Policy:

- JSON report should remain stable and machine-readable.
- Markdown report can be added as a CLI output format later.
- Failed cases should be sorted by cluster, evaluator, and score.
- Reports should include enough detail to answer “what should I fix first?”

Implementation status:

1. Add `failed_cases` to `EvaluationReport`. Done.
2. Add `worst_clusters` sorted by weighted score. Done.
3. Add calibration impact when calibrated scores are present. Done.
4. Add `traceeval report --format json|markdown`. Not done.
5. Add fixture-backed snapshot-style tests for report output. Partially done with CLI JSON assertions.

Acceptance criteria:

- Report identifies failed cases and weak clusters. Done.
- Report is useful without opening raw JSONL. Partially done through structured JSON.
- Existing JSON fields remain backwards compatible where possible.

## 4. Validation Profiles

Current issue:

- Validation now supports profiles and warning severity.
- `draft-cases` allows missing `actual_output` as a warning.
- `runnable-cases` treats missing `actual_output` as an error.
- `evaluation-results` validates result rows and score ranges.
- `calibration-dataset` validates result rows and case/result overlap when cases are provided.

Remaining issue:

- Calibration validation does not yet read human ratings directly.
- There is no schema-version validation yet.

Target:

```rust
pub enum ValidationProfile {
    DraftCases,
    RunnableCases,
    EvaluationResults,
    CalibrationDataset,
}

pub enum ValidationSeverity {
    Error,
    Warning,
}

pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub code: String,
    pub message: String,
    pub case_id: Option<String>,
    pub trace_id: Option<String>,
}
```

Policy:

- `DraftCases`: require IDs and input, allow missing actual/expected output.
- `RunnableCases`: require IDs, input, and actual output.
- `CalibrationDataset`: require overlapping ratings/results and valid score scales.
- `EvaluationResults`: require valid scores, evaluator names, and case IDs.

Implemented CLI:

```bash
traceeval validate \
  --profile runnable-cases \
  --cases eval_cases.jsonl \
  --out validation.json
```

Implementation status:

1. Add `ValidationProfile`. Done.
2. Add `ValidationSeverity`. Done.
3. Update `ValidationReport::is_valid()` to fail only on errors. Done.
4. Add CLI `--profile`. Done.
5. Add fixture tests for draft vs runnable case behavior. Done.

Acceptance criteria:

- Users can validate intermediate datasets without false failures. Done for draft cases.
- CI can still fail hard on production-ready datasets. Done for runnable cases.

## 5. Real Cluster Discovery And LLM Labeling

Authoritative spec: [cluster-discovery.md](cluster-discovery.md).

Current issue:

- Current clustering support is rule-based assignment into known clusters.
- It now has default-build cluster discovery schemas, projection, validation, and nearest-centroid assignment from a manually constructed `ClusterModel`.
- It still does not fit/discover new clusters from embeddings.
- LLM-based cluster naming is specified but not implemented.

The specified architecture stays split:

```text
EmbeddingProvider -> creates CaseEmbedding rows
ClusterDiscovery  -> creates a ClusterModel from cases and embeddings
ClusterLabeler    -> names/explains discovered clusters, often with an LLM
ClusterAssigner   -> maps new cases to known or discovered clusters
```

Policy:

- Do not use LLMs as the primary clustering algorithm by default.
- Use LLMs to label, explain, summarize, and inspect discovered clusters.
- Keep embeddings and ML dependencies feature-gated.
- Keep the default crate capable of rule-based assignment without ML.
- Persist versioned schemas for embeddings, cluster models, and assignments.

Recommended feature flags:

```toml
embeddings-openai = ["openai_dive", "tokio"]
embeddings-local = ["fastembed"]
clustering-linfa = ["linfa", "linfa-clustering", "ndarray"]
cluster-label-openai = ["openai_dive", "schemars", "tokio"]
```

Implementation contract:

- Add `CaseEmbedding`, `ClusterModel`, `DiscoveredCluster`, `ClusterLabel`, and quality types. Done.
- Add validation profiles for embedding datasets, cluster models, and cluster assignments. Done for library and `traceeval validate` inputs.
- Add `ClusterTextProjector`, `EmbeddingProvider`, `ClusterDiscovery`, `ClusterLabeler`, and embedding-aware assignment APIs. Done for traits and nearest-centroid assignment; provider implementations do not exist yet.
- Add CLI subcommands: `cluster embed`, `cluster discover`, `cluster label`, and expanded `cluster assign`. Done for the command contracts; only `cluster assign` has a functional backend today.
- Add K-Means with `linfa-clustering` first; DBSCAN/HDBSCAN-style work is later.
- Add OpenAI embeddings and OpenAI cluster labeling only behind feature flags.

Acceptance criteria:

- Users can discover clusters from historical cases.
- Users can label clusters with an LLM.
- Users can review/edit labels before using them for reports.
- New cases can be assigned to discovered clusters without rerunning discovery.
- Default builds do not include ML, embedding, ANN, or vector database dependencies.

## Recommended Order

1. Public API curation.
2. Validation profiles.
3. Richer reports.
4. Typed errors.
5. Cluster discovery model.
6. LLM cluster labeling.
7. Optional embedding/ML backends.

This order keeps the current non-ML workflow stable before introducing heavier dependencies.
