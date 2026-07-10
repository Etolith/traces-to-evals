use std::path::PathBuf;

use clap::{ArgGroup, Args, Subcommand, ValueEnum};

#[derive(Debug, Clone, Args)]
pub struct ClusterArgs {
    #[command(subcommand)]
    pub command: ClusterCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ClusterCommand {
    /// Generate case embeddings for cluster discovery.
    Embed(ClusterEmbedArgs),
    /// Discover clusters from historical cases and embeddings.
    Discover(ClusterDiscoverArgs),
    /// Label a discovered cluster model.
    Label(ClusterLabelArgs),
    /// Build a vector index from a discovered cluster model.
    Index(ClusterIndexArgs),
    /// Assign cases to known clusters or a discovered cluster model.
    Assign(ClusterAssignArgs),
}

#[derive(Debug, Clone, Args)]
#[command(group(
    ArgGroup::new("cluster_source")
        .required(true)
        .args(["clusters", "model"])
))]
pub struct ClusterAssignArgs {
    /// Input eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Cluster definitions JSONL file.
    #[arg(long, conflicts_with = "model")]
    pub clusters: Option<PathBuf>,
    /// Additional case metadata key to use for rule-based assignment. Repeatable.
    #[arg(long = "metadata-key", requires = "clusters")]
    pub metadata_keys: Vec<String>,
    /// Discovered cluster model JSON file.
    #[arg(long, conflicts_with = "clusters", requires = "embeddings")]
    pub model: Option<PathBuf>,
    /// Case embeddings JSONL file for discovered-model assignment.
    #[arg(long, requires = "model")]
    pub embeddings: Option<PathBuf>,
    /// Distance threshold above which model assignment marks novelty.
    #[arg(long = "novelty-distance-threshold", requires = "model")]
    pub novelty_distance_threshold: Option<f32>,
    /// Optional vector index backend for discovered-model assignment.
    #[arg(long = "vector-index", value_enum, requires = "model")]
    pub vector_index: Option<VectorIndexBackendName>,
    /// Persisted vector index file for indexed discovered-model assignment.
    #[arg(long = "index-file", requires = "vector_index")]
    pub index_file: Option<PathBuf>,
    /// Persisted vector index row-map JSON file.
    #[arg(long = "index-row-map", requires = "vector_index")]
    pub index_row_map: Option<PathBuf>,
    /// Number of IVF lists to probe for ANN vector indexes.
    #[arg(long, requires = "vector_index")]
    pub nprobe: Option<usize>,
    /// HNSW search breadth for ANN vector indexes.
    #[arg(long = "ef-search", requires = "vector_index")]
    pub ef_search: Option<usize>,
    /// Output cluster assignments JSONL file.
    #[arg(long)]
    pub out: PathBuf,
    /// Optional evaluation results JSONL file to annotate with cluster IDs.
    #[arg(long)]
    pub results: Option<PathBuf>,
    /// Optional output path for annotated evaluation results.
    #[arg(long = "results-out", requires = "results")]
    pub results_out: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct ClusterIndexArgs {
    /// Input discovered cluster model JSON file.
    #[arg(long)]
    pub model: PathBuf,
    /// Vector index backend to build.
    #[arg(long, value_enum)]
    pub backend: VectorIndexBackendName,
    /// Distance metric for vector search.
    #[arg(long, value_enum, default_value = "cosine")]
    pub metric: VectorMetricName,
    /// Paimon index kind.
    #[arg(long = "index-kind", value_enum, default_value = "ivf-flat")]
    pub index_kind: PaimonIndexKindName,
    /// Number of IVF lists for Paimon indexes.
    #[arg(long, default_value_t = 1)]
    pub nlist: usize,
    /// Product-quantization sub-vector count for IVF-PQ.
    #[arg(long = "pq-m")]
    pub pq_m: Option<usize>,
    /// Enable OPQ rotation for IVF-PQ.
    #[arg(long = "use-opq", default_value_t = false)]
    pub use_opq: bool,
    /// HNSW max connections per node.
    #[arg(long = "hnsw-m")]
    pub hnsw_m: Option<usize>,
    /// HNSW construction breadth.
    #[arg(long = "hnsw-ef-construction")]
    pub hnsw_ef_construction: Option<usize>,
    /// HNSW maximum level.
    #[arg(long = "hnsw-max-level")]
    pub hnsw_max_level: Option<usize>,
    /// Project namespace for generated row-map schema version.
    #[arg(long = "project-name")]
    pub project_name: Option<String>,
    /// Output vector index file.
    #[arg(long = "out-index")]
    pub out_index: PathBuf,
    /// Output vector index row-map JSON file.
    #[arg(long = "out-row-map")]
    pub out_row_map: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct ClusterEmbedArgs {
    /// Input eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Embedding provider to run.
    #[arg(long, value_enum)]
    pub provider: ClusterEmbeddingProviderName,
    /// Embedding model name.
    #[arg(long)]
    pub model: String,
    /// Optional embedding dimensions override for providers that support it.
    #[arg(long)]
    pub dimensions: Option<u32>,
    /// Project namespace for generated artifact schema versions.
    #[arg(long = "project-name")]
    pub project_name: Option<String>,
    /// Output case embeddings JSONL file.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct ClusterDiscoverArgs {
    /// Input historical eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Input case embeddings JSONL file.
    #[arg(long)]
    pub embeddings: PathBuf,
    /// Discovery algorithm.
    #[arg(long, value_enum)]
    pub algorithm: ClusterAlgorithmName,
    /// Number of clusters for k-means.
    #[arg(long)]
    pub k: Option<usize>,
    /// Representative examples per discovered cluster.
    #[arg(long, default_value_t = 5)]
    pub representatives: usize,
    /// Project namespace for generated artifact schema versions.
    #[arg(long = "project-name")]
    pub project_name: Option<String>,
    /// Output discovered cluster model JSON file.
    #[arg(long = "out-model")]
    pub out_model: PathBuf,
    /// Output cluster assignments JSONL file.
    #[arg(long = "out-assignments")]
    pub out_assignments: PathBuf,
    /// Output report-compatible cluster definitions JSONL file.
    #[arg(long = "out-clusters")]
    pub out_clusters: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct ClusterLabelArgs {
    /// Input discovered cluster model JSON file.
    #[arg(long)]
    pub model: PathBuf,
    /// Input historical eval cases JSONL file.
    #[arg(long)]
    pub cases: PathBuf,
    /// Label provider to run.
    #[arg(long, value_enum)]
    pub provider: ClusterLabelProviderName,
    /// LLM model name.
    #[arg(long = "llm-model")]
    pub llm_model: String,
    /// Output labeled cluster model JSON file.
    #[arg(long = "out-model")]
    pub out_model: PathBuf,
    /// Output report-compatible labeled cluster definitions JSONL file.
    #[arg(long = "out-clusters")]
    pub out_clusters: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ClusterEmbeddingProviderName {
    Openai,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ClusterLabelProviderName {
    Openai,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ClusterAlgorithmName {
    Kmeans,
    Dbscan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VectorIndexBackendName {
    BruteForce,
    Paimon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VectorMetricName {
    Cosine,
    Euclidean,
    InnerProduct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PaimonIndexKindName {
    IvfFlat,
    IvfPq,
    IvfHnswFlat,
    IvfHnswSq,
}
