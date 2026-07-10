# Vector Index Trait And Paimon Backend Design

This document specifies the vector-index abstraction needed before adding an
ANN backend such as Apache Paimon Vector Index.

The current crate stores embeddings as `Vec<f32>` in `CaseEmbedding` and uses a
brute-force scan over discovered cluster centroids for model assignment. That is
good for the MVP because the number of clusters is normally small. A vector
index is useful when assignment needs to search many centroids, many historical
case vectors, or a future nearest-neighbor explainability layer.

## Goals

- Keep vector indexing separate from embedding generation and cluster discovery.
- Keep the default crate free of ANN dependencies.
- Allow multiple backends behind one trait: brute-force default, Paimon later.
- Preserve our public `case_id` and `cluster_id` string IDs while supporting
  backend-owned 64-bit row IDs.
- Support build-time training, persisted index files, search warm-up, single
  query search, batch search, and optional row-id filtering.

## Non-Goals

- Do not replace `EmbeddingProvider`; vector indexes consume vectors, they do
  not generate them.
- Do not replace `ClusterDiscovery`; K-Means and future HDBSCAN/DBSCAN still
  produce `ClusterModel`.
- Do not use an ANN index automatically for small centroid counts.
- Do not add Paimon, HNSW, FAISS, or vector database dependencies to the default
  crate.

## Public Trait Shape

The trait should live under `traces_to_evals::clustering::vector_index`.

```rust
pub type VectorRowId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorMetric {
    Cosine,
    Euclidean,
    InnerProduct,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorRecord<'a> {
    pub row_id: VectorRowId,
    pub external_id: &'a str,
    pub vector: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchOptions {
    pub top_k: usize,
    pub metric: VectorMetric,
    pub nprobe: Option<usize>,
    pub ef_search: Option<usize>,
    pub allowed_row_ids: Option<Vec<VectorRowId>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchHit {
    pub row_id: VectorRowId,
    pub distance: f32,
}

pub trait VectorIndex: Send + Sync {
    fn metric(&self) -> VectorMetric;
    fn dimensions(&self) -> usize;

    fn optimize_for_search(&mut self) -> Result<()> {
        Ok(())
    }

    fn search(
        &mut self,
        query: &[f32],
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchHit>>;

    fn search_batch(
        &mut self,
        queries: &[Vec<f32>],
        options: &VectorSearchOptions,
    ) -> Result<Vec<Vec<VectorSearchHit>>> {
        queries
            .iter()
            .map(|query| self.search(query, options))
            .collect()
    }
}

pub trait VectorIndexBuilder: Send + Sync {
    type Index: VectorIndex;

    fn build(&self, records: &[VectorRecord<'_>]) -> Result<Self::Index>;
}
```

The trait intentionally returns row IDs only. Higher-level code owns the mapping
from row IDs to `cluster_id`, `case_id`, or any future entity ID.

## Default Backend

The default crate should provide a simple exact backend:

```rust
pub struct BruteForceVectorIndex {
    metric: VectorMetric,
    dimensions: usize,
    records: Vec<OwnedVectorRecord>,
}

pub struct BruteForceVectorIndexBuilder {
    metric: VectorMetric,
}
```

This backend:

- Stores vectors in memory as `Vec<f32>`.
- Validates a single fixed dimensionality.
- Supports `Cosine`, `Euclidean`, and `InnerProduct`.
- Honors `top_k` and `allowed_row_ids`.
- Ignores ANN-specific knobs such as `nprobe` and `ef_search`.
- Is used by tests and small cluster models.

## Paimon Backend

Feature flag:

```toml
[features]
ann-paimon = ["paimon-vindex-core", "roaring"]

[dependencies]
paimon-vindex-core = { git = "https://github.com/apache/paimon-vector-index", rev = "93753f7dc8fea0402f7a5c8ee9f080168b553219", package = "paimon-vindex-core", optional = true }
roaring = { version = "0.11", optional = true }
```

If Paimon publishes a crates.io package, prefer a pinned semver dependency. If
not, use a pinned git revision instead of tracking `main`.

The trait uses `&mut self` for search because Paimon's Rust reader mutates
reader-side caches and cursor state during search.

Paimon's current Rust API shape uses:

- `paimon_vindex_core::index::VectorIndexConfig`
- `VectorIndexTrainer::train(config, &training_vectors, training_count)`
- `VectorIndexWriter::new(training)`
- `writer.add_vectors(&row_ids, &vectors, vector_count)`
- `writer.write(...)`
- `VectorIndexReader::open(...)`
- `reader.optimize_for_search()`
- `reader.search(&query, VectorSearchParams)`

The adapter should be private to `src/clustering/vector_index/paimon.rs` and
export only our backend type:

```rust
#[cfg(feature = "ann-paimon")]
pub struct PaimonVectorIndexBuilder {
    pub config: PaimonVectorIndexConfig,
    pub output_path: PathBuf,
}

#[cfg(feature = "ann-paimon")]
pub struct PaimonVectorIndex {
    reader: paimon_vindex_core::index::VectorIndexReader<File>,
    metric: VectorMetric,
    dimensions: usize,
}
```

Suggested config:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum PaimonVectorIndexKind {
    IvfFlat { nlist: usize },
    IvfPq { nlist: usize, m: usize, use_opq: bool },
    IvfHnswFlat { nlist: usize, hnsw: PaimonHnswOptions },
    IvfHnswSq { nlist: usize, hnsw: PaimonHnswOptions },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PaimonVectorIndexConfig {
    pub metric: VectorMetric,
    pub dimensions: usize,
    pub kind: PaimonVectorIndexKind,
    pub optimize_on_open: bool,
}
```

The mapping from our config to Paimon config is straightforward:

- `VectorMetric::Euclidean` -> Paimon `MetricType::L2`
- `VectorMetric::InnerProduct` -> Paimon inner-product metric
- `VectorMetric::Cosine` -> Paimon cosine metric
- `IvfFlat` -> `VectorIndexConfig::IvfFlat`
- `IvfPq` -> `VectorIndexConfig::IvfPq`
- `IvfHnswFlat` -> `VectorIndexConfig::IvfHnswFlat`
- `IvfHnswSq` -> `VectorIndexConfig::IvfHnswSq`

## Row ID Mapping

Paimon search returns numeric row IDs, while this crate uses string IDs.

For cluster assignment from a `ClusterModel`, build the index over cluster
centroids:

```text
row_id 0 -> cluster-0001
row_id 1 -> cluster-0002
row_id 2 -> cluster-0003
```

For future historical-case nearest-neighbor search, build the index over case
embeddings:

```text
row_id 0 -> case-a
row_id 1 -> case-b
row_id 2 -> case-c
```

The persisted sidecar should be explicit JSON:

```rust
pub struct VectorIndexRowMap {
    pub schema_version: String,
    pub index_id: String,
    pub rows: Vec<VectorIndexRow>,
}

pub struct VectorIndexRow {
    pub row_id: VectorRowId,
    pub external_id: String,
}
```

Do not derive row IDs from vector position at query time unless the index and
row map are built in the same process and never persisted.

## Cluster Assignment Integration

Keep `EmbeddingClusterAssigner` as the high-level API. Add a new assigner that
uses the vector index trait:

```rust
pub struct VectorIndexClusterAssigner<I> {
    pub model: ClusterModel,
    pub index: I,
    pub row_map: BTreeMap<VectorRowId, String>,
    pub novelty_distance_threshold: Option<f32>,
}

impl<I> EmbeddingClusterAssigner for VectorIndexClusterAssigner<I>
where
    I: VectorIndex,
{
    fn assign_case_embedding(
        &self,
        case: &EvalCase,
        embedding: &[f32],
    ) -> Result<ClusterAssignment>;
}
```

Assignment rules stay the same as brute-force nearest centroid:

- Query the index with `top_k = 1`.
- Map the returned row ID to a `cluster_id`.
- Store `distance`.
- Compute confidence with the same metric-specific normalization policy used by
  the exact assigner.
- Apply `novelty_distance_threshold`.
- Put `nearest_cluster_id` in metadata when novelty sends the case to
  `unclustered`.

## CLI Shape

Do not change the default `cluster assign --model --embeddings` behavior yet.
Add an explicit optional backend flag:

```bash
traceeval cluster assign \
  --cases new_eval_cases.jsonl \
  --model labeled_cluster_model.json \
  --embeddings new_embeddings.jsonl \
  --vector-index paimon \
  --index-file cluster_centroids.pvindex \
  --index-row-map cluster_centroids.rows.json \
  --nprobe 16 \
  --ef-search 80 \
  --out cluster_assignments.jsonl
```

Index build can be a separate command because Paimon has train/write semantics:

```bash
traceeval cluster index \
  --model labeled_cluster_model.json \
  --backend paimon \
  --index-kind ivf-hnsw-sq \
  --metric cosine \
  --nlist 256 \
  --out-index cluster_centroids.pvindex \
  --out-row-map cluster_centroids.rows.json
```

For the first implementation, only model-centroid indexes are required. Case
embedding indexes can come later.

## Error Handling

Add typed errors instead of leaking backend errors directly:

```rust
pub enum TraceEvalError {
    VectorIndex {
        backend: String,
        message: String,
    },
}
```

Paimon adapter errors should map into `TraceEvalError::VectorIndex {
backend: "paimon".to_string(), ... }`.

## Tests

Required tests before enabling `ann-paimon`:

- Unit: `BruteForceVectorIndex` returns the same nearest row as exact distance.
- Unit: row-map lookup rejects unknown row IDs.
- Unit: `VectorIndexClusterAssigner` matches `ClusterModelAssigner` on a small
  fixed fixture.
- Feature: `cargo test --features ann-paimon` builds without enabling OpenAI or
  `clustering-linfa`.
- Feature: Paimon backend uses fake/temp index files only; no network.
- CLI: building an index from a `ClusterModel` writes both index file and row
  map.
- CLI: assignment with `--vector-index paimon` produces the same cluster IDs as
  brute-force on a deterministic fixture.

## Implementation Order

1. Add `vector_index` module with trait types and `BruteForceVectorIndex`.
2. Add `VectorIndexClusterAssigner<I>`.
3. Add row-map schema and validation.
4. Add `traceeval cluster index` for centroid indexes.
5. Add `ann-paimon` feature and Paimon adapter.
6. Add CLI backend selection for `cluster assign`.
7. Benchmark brute-force centroid search versus Paimon for realistic cluster
   counts before making Paimon the recommended path.

## Open Questions

- Is Paimon published to crates.io, or should the feature use a pinned git
  dependency?
- Should the first Paimon backend use Rust core directly or the C FFI for API
  stability?
- Do we want persisted centroid indexes at all, or only case-level nearest
  neighbor indexes where vector count is much larger?
- Should `VectorMetric::InnerProduct` be added to public scoring/cluster APIs,
  or kept private to vector-index search?
