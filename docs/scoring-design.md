# Scoring and Calibration Design

This document defines the next scoring layer for `traces-to-evals`.

The goal is to make this workflow first-class:

```text
historical eval dataset + human labels
    -> calibration and task clusters

new traces
    -> EvalCase
    -> grade/judge
    -> calibrated score
    -> cluster-aware aggregate report
```

The important constraint is that the default crate should stay lightweight. ML, embeddings, clustering, and vector stores should be optional feature-gated layers.

## 1. Unified `traceeval grade` CLI

The binary now has a `traceeval grade` command for deterministic graders and feature-gated LLM judging. The next step is to extend that command with calibrated scoring inputs and cluster-aware context.

Target commands:

```bash
traceeval grade \
  --cases eval_cases.jsonl \
  --grader exact-match \
  --out grade_results.jsonl
```

```bash
traceeval grade \
  --cases eval_cases.jsonl \
  --grader contains \
  --contains "expected phrase" \
  --out grade_results.jsonl
```

```bash
traceeval grade \
  --cases eval_cases.jsonl \
  --judge openai-dive \
  --model gpt-4o \
  --out judge_results.jsonl
```

```bash
traceeval grade \
  --cases new_eval_cases.jsonl \
  --judge openai-dive \
  --model gpt-4o \
  --calibration historical_calibration.json \
  --clusters eval_clusters.jsonl \
  --out scored_results.jsonl
```

Recommended CLI library:

```toml
clap = { version = "4", features = ["derive"] }
```

`clap` gives us generated help, typed subcommands, validation, shell-friendly errors, and a cleaner path to replacing the current manual argument parsing.

Proposed CLI shape:

```rust
#[derive(clap::Parser)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
pub enum Command {
    Extract(ExtractArgs),
    Validate(ValidateArgs),
    Grade(GradeArgs),
    Calibrate(CalibrateArgs),
    Cluster(ClusterArgs),
    Report(ReportArgs),
}
```

LLM judging goes through the canonical `grade` command:

```bash
traceeval grade --judge openai-dive ...
```

## 2. Result Types

Deterministic graders, LLM judges, and future ML scorers now compose through one canonical result type: `EvaluationResult`.

```rust
pub struct EvaluationResult {
    pub case_id: String,
    pub trace_id: String,
    pub evaluator_name: String,
    pub raw_score: f32,
    pub normalized_score: f32,
    pub score_scale: ScoreScale,
    pub calibrated_score: Option<f32>,
    pub passed: bool,
    pub confidence: Option<f32>,
    pub cluster_id: Option<String>,
    pub evaluation: String,
    pub criteria: Option<EvaluationCriteria>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}
```

Score conventions:

```text
deterministic pass/fail:
  raw_score = 0 or 1
  normalized_score = 0.0 or 1.0

LLM judge:
  raw_score = 1..4
  normalized_score = (raw_score - 1) / 3

calibrated score:
  calibrated_score = probability-like 0.0..1.0 value derived from historical human labels
```

`EvaluationResult` is the canonical scored result type. `GradeResult` and `JudgeResult` are still available for callers that need the raw grader/judge-specific records, but the composable path converts both into `EvaluationResult`:

```rust
impl From<GradeResult> for EvaluationResult
impl From<JudgeResult> for EvaluationResult
```

The composable run API is:

```rust
EvaluationRun::new(cases)
    .evaluate_with(&ExactMatchGrader)?
    .evaluate_with(&NonEmptyOutputGrader)?
    .aggregate_with(&WeightedAggregate::default());
```

## 3. Calibrated Score Based On Previous Eval Datasets

The previous eval dataset should be used for calibration, not treated as training data by default.

Inputs:

```text
historical_eval_cases.jsonl
human_ratings.jsonl
historical_judge_results.jsonl
```

Outputs:

```text
historical_calibration.json
```

Target command:

```bash
traceeval calibrate \
  --cases historical_eval_cases.jsonl \
  --human-ratings human_ratings.jsonl \
  --judge-results historical_judge_results.jsonl \
  --out historical_calibration.json
```

`CalibrationModel::fit()` produces a reusable calibration table from historical `EvaluationResult` rows and human ratings:

```rust
pub struct CalibrationModel {
    pub scorer_name: String,
    pub pass_threshold: f32,
    pub bins: Vec<CalibrationBin>,
    pub global_pass_rate: f32,
    pub mean_absolute_error: f32,
}

pub struct CalibrationBin {
    pub raw_score_min: f32,
    pub raw_score_max: f32,
    pub count: usize,
    pub human_mean_score: f32,
    pub human_pass_rate: f32,
}
```

For v1, use binning:

```text
raw judge score 1 -> historical human pass rate for score 1
raw judge score 2 -> historical human pass rate for score 2
raw judge score 3 -> historical human pass rate for score 3
raw judge score 4 -> historical human pass rate for score 4
```

This needs no ML dependency and is easy to explain.

Later options:

```text
isotonic calibration
logistic calibration / Platt scaling
per-cluster calibration
```

Do not add a general ML optimizer until the simple binning model is insufficient.

## 4. Cluster-Aware Scoring

Clustering should select context and thresholds. It should not be the grader.

The current crate implements rule-based assignment to a known cluster taxonomy. True cluster discovery is a separate future feature.

LLMs should usually be used after discovery to label and explain clusters, not as the primary clustering algorithm. A typical future flow is:

```text
EvalCase text
  -> embeddings
  -> clustering algorithm
  -> representative examples
  -> LLM-generated cluster label, description, rubric, and failure modes
  -> optional human approval
  -> cluster-aware scoring
```

Historical cases become task clusters:

```text
EvalCase + HumanRating + optional embedding
    -> EvalCluster
```

New cases are assigned to the nearest cluster before scoring:

```text
new EvalCase
    -> embedding or lexical features
    -> nearest cluster
    -> cluster rubric / examples / threshold
    -> scorer
    -> calibrated score
```

Proposed types:

```rust
pub struct EvalCluster {
    pub id: String,
    pub label: Option<String>,
    pub case_ids: Vec<String>,
    pub centroid: Option<Vec<f32>>,
    pub rubric: Option<String>,
    pub pass_threshold: f32,
    pub weight: f32,
    pub mean_human_score: f32,
    pub human_pass_rate: f32,
    pub known_failure_modes: Vec<String>,
}

pub struct ClusterAssignment {
    pub case_id: String,
    pub cluster_id: Option<String>,
    pub distance: Option<f32>,
    pub confidence: f32,
    pub novelty: bool,
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

Target commands:

```bash
traceeval cluster build \
  --cases historical_eval_cases.jsonl \
  --human-ratings human_ratings.jsonl \
  --embeddings historical_embeddings.jsonl \
  --algorithm kmeans \
  --k 12 \
  --out eval_clusters.jsonl
```

```bash
traceeval cluster assign \
  --cases new_eval_cases.jsonl \
  --clusters eval_clusters.jsonl \
  --embeddings new_embeddings.jsonl \
  --out cluster_assignments.jsonl
```

Start with three assignment modes:

```text
exact-tag:
  Use case metadata like task_id, route, product area, or scenario tag.

lexical:
  Use normalized input tokens, tool names, and rubric text.

embedding:
  Use cosine similarity over embeddings.
```

Clustering rule:

```text
Cluster on task intent, input, rubric, and tool sequence.
Do not cluster primarily on actual_output.
```

The reason is practical: bad outputs can form coherent clusters and hide failures.

## 5. Libraries For ML Use Cases

Use feature flags so the default crate stays small.

Recommended feature plan:

```toml
[features]
default = []
cli = ["clap"]
llm-judge-openai = ["openai_dive", "async-trait", "tokio"]
embeddings-openai = ["openai_dive"]
embeddings-local = ["fastembed"]
clustering = ["linfa", "linfa-clustering", "ndarray"]
ann-local = ["hnsw_rs"]
vector-qdrant = ["qdrant-client"]

[dependencies]
clap = { version = "4", features = ["derive"], optional = true }
linfa = { version = "0.8", optional = true }
linfa-clustering = { version = "0.8", optional = true }
ndarray = { version = "0.17", optional = true }
fastembed = { version = "5", optional = true }
hnsw_rs = { version = "0.3", optional = true }
qdrant-client = { version = "1", optional = true }
```

Recommended order:

```text
1. clap
   Needed immediately for the unified CLI.

2. No new ML dependency for calibration v1
   Implement score bins directly.

3. embeddings-openai
   Reuse openai_dive to generate embeddings for EvalCase input/rubric/tool summary.

4. clustering
   Add linfa-clustering + ndarray for K-Means first.

5. ann-local or vector-qdrant
   Add only when brute-force cosine search is too slow.

6. embeddings-local
   Add fastembed only when offline/local embedding generation is a product requirement.
```

Library choices:

```text
clap:
  CLI subcommands, validation, help output, and typed argument parsing.

openai_dive:
  Existing OpenAI dependency. Use for judge and OpenAI embeddings.

linfa-clustering:
  Pure Rust clustering algorithms: K-Means, DBSCAN, GMM, OPTICS.

ndarray:
  Matrix/vector representation used by linfa.

fastembed:
  Local ONNX-based embeddings. Useful for offline mode, but adds model download/cache/runtime concerns.

hnsw_rs:
  Local approximate nearest-neighbor index when in-memory brute force is too slow.

qdrant-client:
  Persistent vector database integration for service deployments.
```

## 6. Weighted Aggregate Run Scores

A single mean score is usually too weak. The report should support weighted scoring by cluster, criticality, and scenario coverage.

Inputs:

```text
scored_results.jsonl
cluster_assignments.jsonl
eval_clusters.jsonl
weights.json
```

Target command:

```bash
traceeval report \
  --scores scored_results.jsonl \
  --clusters eval_clusters.jsonl \
  --assignments cluster_assignments.jsonl \
  --weights weights.json \
  --out run_report.json
```

Weights file:

```json
{
  "default_case_weight": 1.0,
  "clusters": {
    "safety": 3.0,
    "tool_execution": 2.0,
    "basic_qa": 1.0
  },
  "criteria": {
    "safety": 3.0,
    "correctness": 2.0,
    "completeness": 1.0,
    "relevance": 1.0
  }
}
```

Aggregate formula:

```text
case_effective_score =
  calibrated_score if present
  else normalized_score

case_weight =
  default_case_weight
  * cluster_weight
  * criticality_weight

run_score =
  sum(case_effective_score * case_weight) / sum(case_weight)
```

Report shape:

```rust
pub struct RunReport {
    pub run_id: String,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub unscored_cases: usize,
    pub run_score: f32,
    pub pass_rate: f32,
    pub weighted_pass_rate: f32,
    pub cluster_scores: Vec<ClusterScore>,
    pub failure_modes: Vec<FailureModeSummary>,
}

pub struct ClusterScore {
    pub cluster_id: String,
    pub label: Option<String>,
    pub total_cases: usize,
    pub run_score: f32,
    pub pass_rate: f32,
    pub weighted_pass_rate: f32,
}
```

Minimum useful report:

```json
{
  "total_cases": 100,
  "passed_cases": 82,
  "run_score": 0.84,
  "weighted_pass_rate": 0.79,
  "clusters": [
    {
      "cluster_id": "tool_execution",
      "total_cases": 18,
      "run_score": 0.71,
      "pass_rate": 0.67
    }
  ]
}
```

## 7. Implementation Phases

Phase 1: CLI and result unification

```text
- Add clap. Done.
- Add traceeval grade. Done.
- Add EvaluationResult. Done.
- Remove pre-release scored-result aliases. Done.
- Convert GradeResult and JudgeResult into EvaluationResult. Done.
- Add EvaluationRun and weighted aggregation. Done.
- Add traceeval report. Done.
```

Phase 2: calibration v1

```text
- Add CalibrationModel. Done for library API.
- Add traceeval calibrate. Done.
- Apply calibration in traceeval report. Done.
- Use score bins first. Done.
```

Phase 3: cluster assignment without ML

```text
- Add EvalCluster and ClusterAssignment. Done.
- Build clusters from explicit metadata tags. Done.
- Assign by exact tag and lexical similarity. Done.
- Add per-cluster aggregate reports. Done.
```

Phase 4: embeddings and ML clustering

```text
- Add embeddings-openai.
- Add embedding JSONL format.
- Add brute-force nearest-cluster assignment by cosine similarity.
- Add linfa-clustering K-Means.
```

Phase 5: scale and production options

```text
- Add hnsw_rs for local ANN.
- Add qdrant-client for persistent vector search.
- Add fastembed for local/offline embeddings if required.
```

## 8. Acceptance Criteria

The feature is useful when this command sequence works on checked-in fixtures:

```bash
traceeval grade \
  --cases fixtures/scoring/new_eval_cases.jsonl \
  --grader exact-match \
  --out target/tmp/grade_results.jsonl

traceeval calibrate \
  --human-ratings fixtures/scoring/human_ratings.jsonl \
  --judge-results fixtures/scoring/historical_judge_results.jsonl \
  --out target/tmp/calibration.json

traceeval cluster assign \
  --cases fixtures/scoring/new_eval_cases.jsonl \
  --clusters fixtures/scoring/eval_clusters.jsonl \
  --out target/tmp/cluster_assignments.jsonl

traceeval report \
  --scores target/tmp/grade_results.jsonl \
  --clusters fixtures/scoring/eval_clusters.jsonl \
  --assignments target/tmp/cluster_assignments.jsonl \
  --out target/tmp/run_report.json
```

Tests should prove:

```text
- deterministic graders emit EvaluationResult
- LLM judge results normalize to 0.0..1.0
- calibration bins change the final score
- unknown clusters are marked as novelty
- weighted run score changes when critical cluster weights change
- report output is stable JSON
```

## 9. References

- `clap` CLI parser: https://docs.rs/clap/latest/clap/
- `openai_dive` OpenAI client and embeddings support: https://docs.rs/openai_dive/latest/openai_dive/
- `linfa-clustering`: https://docs.rs/linfa-clustering/latest/linfa_clustering/
- `ndarray`: https://docs.rs/ndarray/latest/ndarray/
- `fastembed`: https://docs.rs/fastembed/latest/fastembed/
- `hnsw_rs`: https://docs.rs/hnsw_rs/latest/hnsw_rs/
- `qdrant-client`: https://docs.rs/qdrant-client/latest/qdrant_client/
