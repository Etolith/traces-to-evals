pub mod assignment;
pub mod discovery;
pub mod embedding;
pub mod labeling;
#[cfg(any(feature = "embeddings-openai", feature = "cluster-label-openai"))]
pub mod openai;
pub mod quality;
pub mod vector_index;

pub use assignment::{
    ClusterAssigner, ClusterAssignment, ClusterAssignmentRule, ClusterRuleMatch, EvalCluster,
    FnClusterAssignmentRule, KeywordAssignmentRule, MetadataAssignmentRule,
    RuleBasedClusterAssigner, UNCLUSTERED, apply_assignments_to_results,
};
pub use discovery::{
    ClusterAlgorithm, ClusterDiscovery, ClusterDiscoveryInput, ClusterDiscoveryOptions,
    ClusterModel, ClusterModelAssigner, ClusterModelSource, DiscoveredCluster, DistanceMetric,
    EmbeddingClusterAssigner, KMeansClusterDiscovery,
};
pub use embedding::{
    CaseEmbedding, ClusterText, ClusterTextProjector, DefaultClusterTextProjector,
    EmbeddingProvider, ProjectedField,
};
pub use labeling::{ClusterLabel, ClusterLabelPayload, ClusterLabelPrompt, ClusterLabeler};
#[cfg(feature = "cluster-label-openai")]
pub use openai::{OPENAI_CLUSTER_LABEL_PROVIDER_NAME, OpenAiClusterLabeler};
#[cfg(feature = "embeddings-openai")]
pub use openai::{
    OPENAI_EMBEDDING_PROVIDER_NAME, OpenAiEmbeddingClient, OpenAiEmbeddingProvider,
    TextEmbeddingClient,
};
pub use quality::{ClusterQuality, ClusterQualityReport};
pub use vector_index::{
    BruteForceVectorIndex, BruteForceVectorIndexBuilder, OwnedVectorRecord, VectorIndex,
    VectorIndexBuilder, VectorIndexClusterAssigner, VectorIndexRow, VectorIndexRowMap,
    VectorMetric, VectorRecord, VectorRowId, VectorSearchHit, VectorSearchOptions,
    borrowed_records, case_embedding_records, cluster_centroid_records,
};
#[cfg(feature = "ann-paimon")]
pub use vector_index::{
    PaimonHnswOptions, PaimonVectorIndex, PaimonVectorIndexBuilder, PaimonVectorIndexConfig,
    PaimonVectorIndexKind,
};
