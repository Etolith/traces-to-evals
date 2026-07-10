# Cluster Discovery, Embeddings, And LLM Labeling Spec

This document is the implementation contract for real cluster discovery in
`traces-to-evals`.

The current crate supports rule-based assignment into an existing `EvalCluster`
taxonomy. This spec defines the next layer:

```text
EvalCase rows
  -> text projection
  -> embeddings or local feature vectors
  -> cluster discovery
  -> representative examples
  -> optional LLM labels
  -> reviewable cluster model
  -> assignment of new cases
  -> cluster-aware scoring and reports
```

## Goals

- Discover clusters from historical eval cases instead of requiring a predefined taxonomy.
- Keep cluster discovery, cluster labeling, and cluster assignment as separate APIs.
- Keep ML dependencies optional and feature-gated.
- Keep the default crate capable of rule-based assignment with no embedding or ML runtime.
- Make all file formats versioned and validation-friendly.
- Use LLMs for labeling and explanation, not as the default primary clustering algorithm.

## Non-Goals

- No broad ML dependency in the default crate.
- No vector database dependency in the first discovery implementation.
- No automatic human-rating optimizer in the first discovery implementation.
- No hidden network calls in `ClusterDiscovery`; networked embedding and labeling providers stay explicit.
- No compatibility aliases before the first release.

## Project Namespace

`traceeval` is the default CLI/example name and the default artifact namespace,
not a value that should be baked into every implementation point.

Generated versioned artifacts use a project namespace:

```rust
pub struct ProjectName(String);

impl ProjectName {
    pub fn new(name: impl Into<String>) -> Result<Self>;
    pub fn case_embedding_schema_version(&self) -> String;
    pub fn cluster_model_schema_version(&self) -> String;
    pub fn cluster_text_projection_version(&self, include_output: bool) -> String;
}
```

Default artifact versions are:

- `traceeval.case_embedding.v1`
- `traceeval.cluster_model.v1`
- `traceeval.cluster_text.v1`

Callers can override the namespace when generating artifacts:

```rust
let project = ProjectName::new("acme-evals")?;
let projector = DefaultClusterTextProjector::new().with_project_name(project.clone());
let projected = projector.project_case(&case);
let embedding = CaseEmbedding::new_with_project(
    &project,
    &projected,
    "provider",
    "model",
    vector,
    projector.projection_version(),
);
let model = ClusterModel::new_with_project(
    &project,
    "model-1",
    created_at,
    source,
    clusters,
    assignments,
    quality,
);
```

Validation must accept any valid project namespace while still enforcing the
artifact kind and schema version. For example, both
`traceeval.case_embedding.v1` and `acme-evals.case_embedding.v1` are valid case
embedding schemas; `acme-evals.case_embedding.v2` is not valid for v1 readers.

The clap command name must not be hardcoded in the parser. The installed binary
or wrapper script owns the command name shown to users; docs use `traceeval` as
the current repository binary.

## Implementation Status

Implemented in the default crate:

- Extended `ClusterAssignment` with `distance`, `novelty`, and `metadata`.
- Added `CaseEmbedding`, `ClusterText`, `ProjectedField`, and `DefaultClusterTextProjector`.
- Added deterministic default projection that excludes `actual_output`.
- Added SHA-256 projected text hashes for embedding rows.
- Added `ClusterModel`, `DiscoveredCluster`, `ClusterModelSource`, `ClusterLabel`, and cluster quality structs.
- Added `EmbeddingProvider`, `ClusterDiscovery`, `ClusterLabeler`, and `EmbeddingClusterAssigner` traits.
- Added `ProjectName` so schema/projection namespaces can be overridden without editing every type.
- Added `ClusterModelAssigner` for brute-force nearest-centroid assignment.
- Added validation profiles and library checks for embeddings, cluster models, and cluster assignments.
- Added `traceeval validate` inputs for `--embeddings`, `--cluster-model`, and `--assignments`.
- Added `traceeval cluster assign` for rule-based assignment and discovered-model nearest-centroid assignment.
- Added `traceeval cluster discover --algorithm kmeans` behind `clustering-linfa`; it writes `ClusterModel`, assignment JSONL, and report-compatible cluster JSONL artifacts.
- Added `traceeval cluster embed --provider openai` behind `embeddings-openai`; it writes valid `CaseEmbedding` JSONL.
- Added `traceeval cluster label --provider openai` behind `cluster-label-openai`; it writes a labeled `ClusterModel` and report-compatible cluster JSONL.
- Added `clustering-linfa` with `linfa`, `linfa-clustering`, `ndarray`, and seeded `rand`.
- Added `embeddings-openai` and `cluster-label-openai` feature flags. The OpenAI embedding provider and OpenAI cluster labeler are implemented.
- Added report enrichment for exported cluster labels/descriptions, novelty/unclustered counts, and failed-case cluster confidence.

Future optional work:

- Local embedding provider.
- DBSCAN/HDBSCAN-style discovery.
- ANN/vector database assignment backends.
  See [vector-index.md](vector-index.md) for the proposed vector-index trait
  and Paimon backend design.

## Module Shape

All public cluster APIs stay under `traces_to_evals::clustering`.

Recommended implementation split:

```text
src/clustering.rs                 existing rule-based assignment API
src/clustering/discovery.rs       discovery traits and model types
src/clustering/embedding.rs       embedding traits and embedding row schema
src/clustering/labeling.rs        LLM labeler traits and payload schema
src/clustering/quality.rs         cluster quality metrics
```

If the crate keeps a single `src/clustering.rs` file for now, the public API
must still expose the same names listed below.

## Existing Types To Keep

`EvalCluster` remains the cluster descriptor used by reports, weights, and
known-taxonomy assignment:

```rust
pub struct EvalCluster {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub weight: f32,
    pub metadata: BTreeMap<String, serde_json::Value>,
}
```

`ClusterAssignment` remains the assignment row used by CLI and reports. It now
includes discovery metadata:

```rust
pub struct ClusterAssignment {
    pub case_id: String,
    pub trace_id: String,
    pub cluster_id: String,
    pub confidence: f32,
    pub method: String,
    pub distance: Option<f32>,
    pub novelty: bool,
    pub metadata: BTreeMap<String, serde_json::Value>,
}
```

Older rows with only the first five fields remain readable through serde
defaults.

`ClusterAssigner` remains the rule-based assignment trait for assigning cases
to an existing taxonomy:

```rust
pub trait ClusterAssigner {
    fn assign_case(&self, case: &EvalCase) -> Result<ClusterAssignment>;
}
```

Embedding-based assignment needs a separate method because a bare `EvalCase`
does not contain an embedding:

```rust
pub trait EmbeddingClusterAssigner {
    fn assign_case_embedding(
        &self,
        case: &EvalCase,
        embedding: &[f32],
    ) -> Result<ClusterAssignment>;

    fn assign_case_embeddings(
        &self,
        cases: &[EvalCase],
        embeddings: &[CaseEmbedding],
    ) -> Result<Vec<ClusterAssignment>>;
}
```

## Text Projection

Discovery never embeds raw JSON directly. It first projects each `EvalCase`
into deterministic text.

```rust
pub trait ClusterTextProjector {
    fn projection_version(&self) -> String;
    fn project_case(&self, case: &EvalCase) -> ClusterText;
}

pub struct ClusterText {
    pub case_id: String,
    pub trace_id: String,
    pub text: String,
    pub fields: Vec<ProjectedField>,
}

pub struct ProjectedField {
    pub name: String,
    pub value: String,
}
```

Default projection version: `traceeval.cluster_text.v1`. With a custom
`ProjectName`, this becomes `{project}.cluster_text.v1`.

Default included fields, in order:

1. `input`
2. `rubric`, when present
3. `expected_output`, when present
4. selected metadata keys:
   - `route`
   - `task`
   - `task_id`
   - `scenario`
   - `tool`
   - `tool_name`
   - `product_area`
   - `cluster_id`
   - `tags`

Default excluded fields:

- `actual_output`
- evaluator output
- calibrated score
- pass/fail
- human rating notes

Rationale: clusters should represent task intent and scenario shape. Bad
outputs can form coherent clusters and hide failure modes if used as primary
cluster input.

Optional CLI override:

```bash
--projection include-output
```

This may include `actual_output`, but it must set a different projection
version such as `traceeval.cluster_text.v1.include_output`.

## Embedding API

Embedding generation is separate from discovery.

```rust
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn provider_name(&self) -> String;
    fn model_name(&self) -> String;

    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    async fn embed_cases<P>(
        &self,
        projector: &P,
        cases: &[EvalCase],
    ) -> Result<Vec<CaseEmbedding>>
    where
        P: ClusterTextProjector + Send + Sync;
}
```

Serialized embedding row:

```rust
pub struct CaseEmbedding {
    pub schema_version: String,
    pub case_id: String,
    pub trace_id: String,
    pub provider: String,
    pub model: String,
    pub dimensions: usize,
    pub vector: Vec<f32>,
    pub projection_version: String,
    pub text_hash: String,
    pub metadata: BTreeMap<String, serde_json::Value>,
}
```

Required JSONL schema values:

- `schema_version`: `{project}.case_embedding.v1`; default is `traceeval.case_embedding.v1`
- `dimensions`: must equal `vector.len()`
- `text_hash`: lowercase hex SHA-256 of projected text bytes

Validation rules:

- All embeddings in one discovery run must have identical dimensions.
- All embeddings must have finite values.
- Every embedding `case_id` must match exactly one input `EvalCase`.
- Duplicate `case_id` values are invalid.
- Missing embeddings are invalid for embedding-based discovery.
- Extra embeddings not present in cases are invalid unless `--allow-extra-embeddings` is passed.

## Discovery Input

Discovery consumes cases and precomputed features.

```rust
pub struct ClusterDiscoveryInput<'a> {
    pub cases: &'a [EvalCase],
    pub embeddings: Option<&'a [CaseEmbedding]>,
    pub human_ratings: Option<&'a [HumanRating]>,
    pub previous_results: Option<&'a [EvaluationResult]>,
    pub options: ClusterDiscoveryOptions,
}

pub struct ClusterDiscoveryOptions {
    pub model_id: Option<String>,
    pub project_name: ProjectName,
    pub algorithm: ClusterAlgorithm,
    pub distance_metric: DistanceMetric,
    pub representative_count: usize,
    pub random_seed: u64,
    pub novelty_distance_threshold: Option<f32>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

pub enum ClusterAlgorithm {
    KMeans { k: usize, max_iterations: usize, tolerance: f32 },
    Dbscan { min_points: usize, epsilon: f32 },
}

pub enum DistanceMetric {
    Cosine,
    Euclidean,
}
```

MVP requirements:

- Implement `KMeans` first.
- Require `--k` for `KMeans`; do not guess `k` by default.
- Use cosine distance for embedding assignment by default.
- Use deterministic `random_seed`, default `42`.
- Use `representative_count`, default `5`.
- Reject discovery when `k == 0`.
- Reject discovery when `cases.len() < k`.

DBSCAN/HDBSCAN can be added after K-Means. If HDBSCAN is used later, it must
be feature-gated and documented separately because `linfa-clustering` does not
provide HDBSCAN as the first obvious Rust path.

## Discovery API

```rust
pub trait ClusterDiscovery {
    fn algorithm_name(&self) -> &'static str;
    fn fit(&self, input: ClusterDiscoveryInput<'_>) -> Result<ClusterModel>;
}
```

First implementation:

```rust
pub struct KMeansClusterDiscovery {
    pub k: usize,
    pub max_iterations: usize,
    pub tolerance: f32,
    pub random_seed: u64,
}
```

The implementation must:

- Validate embedding dimensions.
- Normalize vectors for cosine distance.
- Fit clusters.
- Assign every input case to a cluster.
- Compute centroids.
- Pick representative cases nearest to each centroid.
- Compute cluster quality metrics.
- Produce a versioned `ClusterModel`.

## Cluster Model

`ClusterModel` is the persisted discovery artifact.

```rust
pub struct ClusterModel {
    pub schema_version: String,
    pub model_id: String,
    pub created_at: String,
    pub source: ClusterModelSource,
    pub clusters: Vec<DiscoveredCluster>,
    pub assignments: Vec<ClusterAssignment>,
    pub quality: ClusterQualityReport,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

pub struct ClusterModelSource {
    pub case_count: usize,
    pub embedding_provider: Option<String>,
    pub embedding_model: Option<String>,
    pub embedding_dimensions: Option<usize>,
    pub projection_version: Option<String>,
    pub algorithm: String,
    pub distance_metric: String,
    pub random_seed: u64,
}

pub struct DiscoveredCluster {
    pub id: String,
    pub size: usize,
    pub centroid: Option<Vec<f32>>,
    pub representative_case_ids: Vec<String>,
    pub radius: Option<f32>,
    pub mean_distance: Option<f32>,
    pub quality: ClusterQuality,
    pub label: Option<ClusterLabel>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}
```

Required schema values:

- `schema_version`: `{project}.cluster_model.v1`; default is `traceeval.cluster_model.v1`
- `model_id`: caller-provided or generated as `cluster-model-{unix_seconds}`
- cluster ids: `cluster-0001`, `cluster-0002`, sorted by descending size unless labels are manually edited later

`ClusterModel` must be serializable as pretty JSON. Assignments may also be
written as JSONL for existing CLI/report workflows.

## Cluster Quality

```rust
pub struct ClusterQualityReport {
    pub cluster_count: usize,
    pub assigned_case_count: usize,
    pub mean_distance: Option<f32>,
    pub silhouette_score: Option<f32>,
    pub clusters: Vec<ClusterQuality>,
}

pub struct ClusterQuality {
    pub cluster_id: String,
    pub size: usize,
    pub mean_distance: Option<f32>,
    pub max_distance: Option<f32>,
    pub silhouette_score: Option<f32>,
    pub representative_case_ids: Vec<String>,
}
```

MVP quality metrics:

- Always compute size.
- Always compute mean and max distance when embeddings exist.
- Compute silhouette score only when cheap enough for the dataset size.
- For MVP, silhouette may be omitted above 10,000 cases unless a faster path is implemented.

Quality validation:

- Empty clusters are invalid in persisted `ClusterModel`.
- A cluster with no representatives is invalid.
- A model with assignments referencing unknown cluster ids is invalid.

## LLM Cluster Labeling

LLM labeling consumes a discovered cluster and representative cases. It does
not run clustering.

```rust
#[async_trait::async_trait]
pub trait ClusterLabeler: Send + Sync {
    fn labeler_name(&self) -> String;

    async fn label_cluster(
        &self,
        cluster: &DiscoveredCluster,
        examples: &[EvalCase],
    ) -> Result<ClusterLabel>;

    async fn label_model(
        &self,
        model: ClusterModel,
        cases: &[EvalCase],
    ) -> Result<ClusterModel>;
}

pub struct ClusterLabel {
    pub label: String,
    pub description: String,
    pub suggested_rubric: Option<String>,
    pub known_failure_modes: Vec<String>,
    pub confidence: f32,
    pub needs_review: bool,
    pub metadata: BTreeMap<String, serde_json::Value>,
}
```

OpenAI implementation:

```rust
pub struct OpenAiClusterLabeler<C = OpenAiChatClient> {
    chat_client: C,
    model: String,
}
```

Feature flag:

```toml
cluster-label-openai = ["openai_dive", "schemars", "tokio"]
```

Structured output payload:

```rust
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterLabelPayload {
    pub label: String,
    pub description: String,
    pub suggested_rubric: String,
    pub known_failure_modes: Vec<String>,
    pub confidence: f32,
    pub needs_review: bool,
}
```

`suggested_rubric` is an empty string when the model has no useful rubric
suggestion; conversion into `ClusterLabel` trims it into `None`.

Prompt inputs:

- cluster id
- cluster size
- representative examples
- projected text fields
- rubric, when present
- metadata keys selected by the projector
- optional failed-case snippets only when explicitly enabled

Prompt exclusions by default:

- full raw traces
- secrets or credentials
- large actual outputs
- hidden chain-of-thought

Label validation:

- `label` must be non-empty and no longer than 80 characters.
- `description` must be non-empty and no longer than 600 characters.
- `confidence` must be finite and in `0.0..=1.0`.
- `known_failure_modes` entries must be non-empty.

Human review workflow:

1. `traceeval cluster label` writes a labeled model.
2. Users edit labels/descriptions/rubrics directly in JSON.
3. `traceeval validate --profile cluster-model --cluster-model model.json` validates the edited file.
4. Reports consume the reviewed model.

## Assignment From A Discovered Model

```rust
pub struct ClusterModelAssigner {
    pub model: ClusterModel,
    pub distance_metric: DistanceMetric,
    pub novelty_distance_threshold: Option<f32>,
}
```

For embeddings:

```rust
impl EmbeddingClusterAssigner for ClusterModelAssigner {
    fn assign_case_embedding(
        &self,
        case: &EvalCase,
        embedding: &[f32],
    ) -> Result<ClusterAssignment>;
}
```

Assignment rules:

- Compute distance from case embedding to every cluster centroid.
- Choose the closest centroid.
- Set `method = "embedding_nearest_centroid"`.
- Set `distance` to the selected distance.
- Set `confidence = 1.0 - normalized_distance`, clamped to `0.0..=1.0`.
- Set `novelty = true` when distance exceeds `novelty_distance_threshold`.
- Set `cluster_id = "unclustered"` when no centroid is available or novelty policy rejects assignment.

MVP nearest-neighbor implementation:

- Brute-force scan over centroids.
- No ANN dependency.
- Add ANN only when benchmarked assignment latency requires it.

## CLI Contract

Current rule-based `traceeval cluster` behavior is exposed as the `assign`
subcommand:

```bash
traceeval cluster assign \
  --cases eval_cases.jsonl \
  --clusters clusters.jsonl \
  --out cluster_assignments.jsonl \
  --results eval_results.jsonl \
  --results-out clustered_results.jsonl
```

The default metadata rule only reads explicit cluster taxonomy fields:
`cluster_id`, `cluster`, `task_cluster`, and `tags`. Application metadata is
opt-in and may be selected with a repeatable flag:

```bash
traceeval cluster assign \
  --cases eval_cases.jsonl \
  --clusters clusters.jsonl \
  --metadata-key route \
  --metadata-key product_area \
  --out cluster_assignments.jsonl
```

All extractors preserve arbitrary trace metadata, and the OpenInference
extractor also preserves arbitrary root-span attributes. No domain semantics
are assigned to those fields. `--metadata-key` applies only to rule-based
assignment with `--clusters`; discovered-model assignment continues to use
embeddings.

Embedding generation:

```bash
traceeval cluster embed \
  --cases historical_eval_cases.jsonl \
  --provider openai \
  --model text-embedding-3-small \
  --dimensions 512 \
  --project-name acme-evals \
  --out historical_embeddings.jsonl
```

Cluster discovery from precomputed embeddings:

```bash
traceeval cluster discover \
  --cases historical_eval_cases.jsonl \
  --embeddings historical_embeddings.jsonl \
  --algorithm kmeans \
  --k 12 \
  --representatives 5 \
  --project-name acme-evals \
  --out-model cluster_model.json \
  --out-assignments cluster_assignments.jsonl \
  --out-clusters clusters.jsonl
```

LLM labeling:

```bash
traceeval cluster label \
  --model cluster_model.json \
  --cases historical_eval_cases.jsonl \
  --provider openai \
  --llm-model <openai-chat-model> \
  --out-model labeled_cluster_model.json \
  --out-clusters labeled_clusters.jsonl
```

Assignment from a discovered model:

```bash
traceeval cluster assign \
  --cases new_eval_cases.jsonl \
  --model labeled_cluster_model.json \
  --embeddings new_embeddings.jsonl \
  --out cluster_assignments.jsonl
```

CLI validation:

- `cluster discover --algorithm kmeans` requires `--k`.
- `cluster discover` requires `--embeddings` for MVP.
- `cluster label` requires `cluster-label-openai` for `--provider openai`.
- `cluster embed --provider openai` requires `embeddings-openai`.
- `cluster assign --model` plus `--embeddings` uses discovered-model assignment.
- `cluster assign --clusters` uses existing rule-based assignment.
- `cluster assign --metadata-key` is repeatable and requires `--clusters`.
- `--model` and `--clusters` conflict.

## Feature Flags And Dependencies

Final feature flag names:

```toml
[features]
default = []
llm-judge-openai = ["openai_dive", "schemars", "tokio"]
embeddings-openai = ["openai_dive", "tokio"]
embeddings-local = ["fastembed"]
clustering-linfa = ["linfa", "linfa-clustering", "ndarray", "rand"]
cluster-label-openai = ["openai_dive", "schemars", "tokio"]
ann-paimon = ["paimon-vindex-core", "roaring"]

[dependencies]
linfa = { version = "0.8", optional = true }
linfa-clustering = { version = "0.8", optional = true }
ndarray = { version = "0.16", optional = true }
rand = { version = "0.8", features = ["small_rng"], optional = true }
fastembed = { version = "5", optional = true }
paimon-vindex-core = { git = "https://github.com/apache/paimon-vector-index", rev = "93753f7dc8fea0402f7a5c8ee9f080168b553219", package = "paimon-vindex-core", optional = true }
roaring = { version = "0.11", optional = true }
```

Dependency policy:

- `clustering-linfa` is required for K-Means.
- `embeddings-openai` is the first network embedding provider.
- `embeddings-local` is optional and should not be added until local/offline embedding is required.
- `ann-paimon` is optional and should not be added until brute-force centroid search is too slow or case-level nearest-neighbor search is required.
- No feature should enable both OpenAI judging and OpenAI embedding unless it explicitly needs both.

## Validation Profiles

Implemented profiles:

```rust
pub enum ValidationProfile {
    DraftCases,
    RunnableCases,
    EvaluationResults,
    CalibrationDataset,
    EmbeddingDataset,
    ClusterModel,
    ClusterAssignments,
}
```

CLI additions:

```bash
traceeval validate --profile embedding-dataset --embeddings embeddings.jsonl --cases cases.jsonl
traceeval validate --profile cluster-model --cluster-model cluster_model.json
traceeval validate --profile cluster-assignments --assignments assignments.jsonl --cases cases.jsonl
```

Validation rules:

- `EmbeddingDataset`: enforce schema, dimensions, finite values, duplicate case ids, and overlap with cases.
- `ClusterModel`: enforce schema, non-empty clusters, valid assignments, representatives, and label constraints.
- `ClusterAssignments`: enforce case overlap, cluster id references, finite confidence, and valid novelty fields.

## Errors

Implemented typed errors:

```rust
pub enum TraceEvalError {
    InvalidEmbedding { case_id: String, message: String },
    EmbeddingProvider { provider: String, message: String },
    ClusterDiscovery { algorithm: String, message: String },
    ClusterLabeling { provider: String, cluster_id: String, message: String },
    ClusterModelValidation { model_id: String, message: String },
}
```

Provider/network failures must be mapped into these variants or
`TraceEvalError::Provider`; HTTP APIs must not expose raw `anyhow` failures.

## Report Integration

`EvaluationReport` should use discovered clusters exactly like manually defined
clusters after assignment has populated `EvaluationResult.cluster_id`.

Additional report fields:

```rust
pub struct ClusterScore {
    pub cluster_id: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub score: RunScore,
}

pub struct FailedCase {
    pub cluster_label: Option<String>,
    pub cluster_confidence: Option<f32>,
}
```

Report rules:

- Use cluster labels from exported `EvalCluster` rows when available.
- Include novelty/unclustered counts.
- Include worst clusters using weighted score.
- Include failed cases with cluster label and assignment confidence when available.

## Acceptance Tests

Minimum test matrix before implementation is considered complete:

- Unit: default text projection is deterministic and excludes `actual_output`.
- Unit: embedding validation rejects mixed dimensions.
- Unit: K-Means discovery with fixed seed produces stable cluster ids and assignments on a fixture.
- Unit: representative cases are nearest to centroid.
- Unit: nearest-centroid assignment sets distance, confidence, method, and novelty.
- Unit: LLM label payload rejects unknown fields and invalid confidence.
- CLI: `cluster embed` writes valid `CaseEmbedding` JSONL behind feature flag.
- CLI: `cluster discover` writes `ClusterModel`, `ClusterAssignment` JSONL, and `EvalCluster` JSONL.
- CLI: `cluster label` writes a labeled model using an injected fake chat client.
- CLI: `validate --profile cluster-model` fails on invalid representatives and unknown assignment cluster ids.
- Integration: discovered cluster assignments flow into `traceeval report`.
- Feature: default build does not compile or pull `linfa`, `fastembed`, or ANN dependencies.
- Feature: `cargo test --features clustering-linfa`.
- Feature: `cargo test --features embeddings-openai,cluster-label-openai` with fake providers only.

## Implementation Order

1. Add schema structs only: `CaseEmbedding`, `ClusterModel`, `DiscoveredCluster`, `ClusterLabel`, quality types.
2. Add validation profiles for embeddings, cluster models, and cluster assignments.
3. Add `ClusterTextProjector` and deterministic default projection tests.
4. Add brute-force centroid assignment from a manually constructed `ClusterModel`.
5. Add `EmbeddingProvider` trait and fake provider tests.
6. Add OpenAI embedding provider behind `embeddings-openai`.
7. Add `clustering-linfa` feature and K-Means discovery.
8. Add `ClusterLabeler` trait and fake labeler tests.
9. Add OpenAI cluster labeler behind `cluster-label-openai`.
10. Add CLI subcommands: `embed`, `discover`, `label`, and the expanded `assign`.
11. Integrate discovered labels/confidence into reports.
12. Benchmark assignment; add ANN only if brute-force centroid search is insufficient.

Steps 1 through 5 and step 8 are implemented in the default crate. Step 6 is
implemented behind `embeddings-openai`. Step 7 is implemented behind
`clustering-linfa`. Step 9 is implemented behind `cluster-label-openai`. Step
10 is implemented for OpenAI embeddings, K-Means discovery, OpenAI labeling,
and rule/model assignment. Step 11 is implemented for exported `EvalCluster`
rows and assignment metadata. Step 12 remains future work and should be driven
by benchmark evidence.

## Done Definition

Cluster discovery is done when:

- The default crate still builds without ML dependencies.
- A user can generate embeddings, discover clusters, label clusters, review labels, assign new cases, and produce a cluster-aware report using documented commands.
- All persisted artifacts have versioned schemas and validation profiles.
- LLM labeling can be replaced by a fake labeler in tests.
- OpenAI embedding and labeling providers are feature-gated and never required for local rule-based clustering.
- Provider failures are typed and do not leak raw `anyhow` from public APIs.
