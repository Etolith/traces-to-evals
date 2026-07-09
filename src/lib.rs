pub mod calibration;
#[doc(hidden)]
pub mod cli;
pub mod clustering;
mod commands;
pub mod error;
pub mod evaluation;
pub mod export;
pub mod extractors;
pub mod graders;
pub mod io;
pub mod judge;
pub mod model;
pub mod project;
#[cfg(any(feature = "llm-judge-openai", feature = "cluster-label-openai"))]
#[doc(hidden)]
pub mod providers;
pub mod report;
pub mod validation;

pub use clustering::{
    CaseEmbedding, ClusterAlgorithm, ClusterAssigner, ClusterAssignment, ClusterAssignmentRule,
    ClusterDiscovery, ClusterDiscoveryInput, ClusterDiscoveryOptions, ClusterLabel,
    ClusterLabelPayload, ClusterLabelPrompt, ClusterLabeler, ClusterModel, ClusterModelAssigner,
    ClusterModelSource, ClusterQuality, ClusterQualityReport, ClusterRuleMatch, ClusterText,
    ClusterTextProjector, DefaultClusterTextProjector, DiscoveredCluster, DistanceMetric,
    EmbeddingClusterAssigner, EmbeddingProvider, EvalCluster, FnClusterAssignmentRule,
    KMeansClusterDiscovery, KeywordAssignmentRule, MetadataAssignmentRule, ProjectedField,
    RuleBasedClusterAssigner,
};
#[cfg(feature = "cluster-label-openai")]
pub use clustering::{OPENAI_CLUSTER_LABEL_PROVIDER_NAME, OpenAiClusterLabeler};
#[cfg(feature = "embeddings-openai")]
pub use clustering::{
    OPENAI_EMBEDDING_PROVIDER_NAME, OpenAiEmbeddingClient, OpenAiEmbeddingProvider,
    TextEmbeddingClient,
};
pub use error::{Result, TraceEvalError};
pub use evaluation::{
    AsyncEvaluator, EvaluationCriteria, EvaluationResult, EvaluationRun, Evaluator, RunScore,
    ScoreScale, WeightedAggregate,
};
pub use model::{EvalCase, Span, SpanKind, Trace};
pub use project::{DEFAULT_PROJECT_NAME, ProjectName};
pub use report::{
    CalibrationImpact, ClusterIssue, ClusterScore, EvaluationReport, EvaluatorScore, FailedCase,
};
pub use validation::{ValidationIssue, ValidationProfile, ValidationReport, ValidationSeverity};

pub mod prelude {
    pub use crate::clustering::{
        CaseEmbedding, ClusterAlgorithm, ClusterAssigner, ClusterAssignment, ClusterAssignmentRule,
        ClusterDiscovery, ClusterDiscoveryInput, ClusterDiscoveryOptions, ClusterLabel,
        ClusterLabelPayload, ClusterLabelPrompt, ClusterLabeler, ClusterModel,
        ClusterModelAssigner, ClusterModelSource, ClusterQuality, ClusterQualityReport,
        ClusterRuleMatch, ClusterText, ClusterTextProjector, DefaultClusterTextProjector,
        DiscoveredCluster, DistanceMetric, EmbeddingClusterAssigner, EmbeddingProvider,
        EvalCluster, FnClusterAssignmentRule, KMeansClusterDiscovery, KeywordAssignmentRule,
        MetadataAssignmentRule, ProjectedField, RuleBasedClusterAssigner,
    };
    #[cfg(feature = "cluster-label-openai")]
    pub use crate::clustering::{OPENAI_CLUSTER_LABEL_PROVIDER_NAME, OpenAiClusterLabeler};
    #[cfg(feature = "embeddings-openai")]
    pub use crate::clustering::{
        OPENAI_EMBEDDING_PROVIDER_NAME, OpenAiEmbeddingClient, OpenAiEmbeddingProvider,
        TextEmbeddingClient,
    };
    pub use crate::error::{Result, TraceEvalError};
    pub use crate::evaluation::{
        AsyncEvaluator, EvaluationCriteria, EvaluationResult, EvaluationRun, Evaluator, RunScore,
        ScoreScale, WeightedAggregate,
    };
    pub use crate::extractors::{EvalCaseExtractor, OpenInferenceExtractor, SimpleExtractor};
    pub use crate::graders::{
        ContainsGrader, DeterministicGrader, ExactMatchGrader, NonEmptyOutputGrader,
    };
    pub use crate::model::{EvalCase, Span, SpanKind, Trace};
    pub use crate::project::{DEFAULT_PROJECT_NAME, ProjectName};
    pub use crate::report::{
        CalibrationImpact, ClusterIssue, ClusterScore, EvaluationReport, EvaluatorScore, FailedCase,
    };
    pub use crate::validation::{
        ValidationIssue, ValidationProfile, ValidationReport, ValidationSeverity,
    };
}
