pub mod behavior;
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

pub use behavior::{
    AGENT_BEHAVIOR_TRACE_SCHEMA_VERSION, AgentBehaviorNormalizer, AgentBehaviorTrace, AgentRole,
    AgentTurn, ApprovalBypassDetector, ApprovalOutcome, BEHAVIOR_FINDING_SCHEMA_VERSION,
    BehaviorFinding, CandidateGenerator, DETERMINISTIC_DETECTOR_VERSION, DeterministicDetectorSet,
    EVAL_CANDIDATE_SCHEMA_VERSION, EvalCandidate, EvalCandidateGenerator, EvalCandidateStatus,
    EvidenceRef, ExcessiveToolUsageDetector, FalseSuccessClaimDetector, FinalOutcome,
    FinalOutcomeStatus, FindingEvalCandidateGenerator, FindingSeverity, MissingResolutionDetector,
    NormalizedToolError, OpenInferenceBehaviorNormalizer, PolicyDecision, PolicyDecisionOutcome,
    PolicyViolationDetector, RecoveryAnalyzer, RecoveryStatus, RepeatedToolFailureDetector,
    StateChangeRef, TerminalToolFailureDetector, ToolCallFact, ToolCallLoopDetector,
    ToolCallStatus, TraceDetector, UncertainMutationStateDetector, UnresolvedEscalationDetector,
};
pub use clustering::{
    BruteForceVectorIndex, BruteForceVectorIndexBuilder, CaseEmbedding, ClusterAlgorithm,
    ClusterAssigner, ClusterAssignment, ClusterAssignmentRule, ClusterDiscovery,
    ClusterDiscoveryInput, ClusterDiscoveryOptions, ClusterLabel, ClusterLabelPayload,
    ClusterLabelPrompt, ClusterLabeler, ClusterModel, ClusterModelAssigner, ClusterModelSource,
    ClusterQuality, ClusterQualityReport, ClusterRuleMatch, ClusterText, ClusterTextProjector,
    DefaultClusterTextProjector, DiscoveredCluster, DistanceMetric, EmbeddingClusterAssigner,
    EmbeddingProvider, EvalCluster, FnClusterAssignmentRule, KMeansClusterDiscovery,
    KeywordAssignmentRule, MetadataAssignmentRule, OwnedVectorRecord, ProjectedField,
    RuleBasedClusterAssigner, VectorIndex, VectorIndexBuilder, VectorIndexClusterAssigner,
    VectorIndexRow, VectorIndexRowMap, VectorMetric, VectorRecord, VectorRowId, VectorSearchHit,
    VectorSearchOptions, borrowed_records, case_embedding_records, cluster_centroid_records,
};
#[cfg(feature = "cluster-label-openai")]
pub use clustering::{OPENAI_CLUSTER_LABEL_PROVIDER_NAME, OpenAiClusterLabeler};
#[cfg(feature = "embeddings-openai")]
pub use clustering::{
    OPENAI_EMBEDDING_PROVIDER_NAME, OpenAiEmbeddingClient, OpenAiEmbeddingProvider,
    TextEmbeddingClient,
};
#[cfg(feature = "ann-paimon")]
pub use clustering::{
    PaimonHnswOptions, PaimonVectorIndex, PaimonVectorIndexBuilder, PaimonVectorIndexConfig,
    PaimonVectorIndexKind,
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
    pub use crate::behavior::{
        AGENT_BEHAVIOR_TRACE_SCHEMA_VERSION, AgentBehaviorNormalizer, AgentBehaviorTrace,
        AgentRole, AgentTurn, ApprovalBypassDetector, ApprovalOutcome,
        BEHAVIOR_FINDING_SCHEMA_VERSION, BehaviorFinding, CandidateGenerator,
        DETERMINISTIC_DETECTOR_VERSION, DeterministicDetectorSet, EVAL_CANDIDATE_SCHEMA_VERSION,
        EvalCandidate, EvalCandidateGenerator, EvalCandidateStatus, EvidenceRef,
        ExcessiveToolUsageDetector, FalseSuccessClaimDetector, FinalOutcome, FinalOutcomeStatus,
        FindingEvalCandidateGenerator, FindingSeverity, MissingResolutionDetector,
        NormalizedToolError, OpenInferenceBehaviorNormalizer, PolicyDecision,
        PolicyDecisionOutcome, PolicyViolationDetector, RecoveryAnalyzer, RecoveryStatus,
        RepeatedToolFailureDetector, StateChangeRef, TerminalToolFailureDetector, ToolCallFact,
        ToolCallLoopDetector, ToolCallStatus, TraceDetector, UncertainMutationStateDetector,
        UnresolvedEscalationDetector,
    };
    pub use crate::clustering::{
        BruteForceVectorIndex, BruteForceVectorIndexBuilder, CaseEmbedding, ClusterAlgorithm,
        ClusterAssigner, ClusterAssignment, ClusterAssignmentRule, ClusterDiscovery,
        ClusterDiscoveryInput, ClusterDiscoveryOptions, ClusterLabel, ClusterLabelPayload,
        ClusterLabelPrompt, ClusterLabeler, ClusterModel, ClusterModelAssigner, ClusterModelSource,
        ClusterQuality, ClusterQualityReport, ClusterRuleMatch, ClusterText, ClusterTextProjector,
        DefaultClusterTextProjector, DiscoveredCluster, DistanceMetric, EmbeddingClusterAssigner,
        EmbeddingProvider, EvalCluster, FnClusterAssignmentRule, KMeansClusterDiscovery,
        KeywordAssignmentRule, MetadataAssignmentRule, OwnedVectorRecord, ProjectedField,
        RuleBasedClusterAssigner, VectorIndex, VectorIndexBuilder, VectorIndexClusterAssigner,
        VectorIndexRow, VectorIndexRowMap, VectorMetric, VectorRecord, VectorRowId,
        VectorSearchHit, VectorSearchOptions, borrowed_records, case_embedding_records,
        cluster_centroid_records,
    };
    #[cfg(feature = "cluster-label-openai")]
    pub use crate::clustering::{OPENAI_CLUSTER_LABEL_PROVIDER_NAME, OpenAiClusterLabeler};
    #[cfg(feature = "embeddings-openai")]
    pub use crate::clustering::{
        OPENAI_EMBEDDING_PROVIDER_NAME, OpenAiEmbeddingClient, OpenAiEmbeddingProvider,
        TextEmbeddingClient,
    };
    #[cfg(feature = "ann-paimon")]
    pub use crate::clustering::{
        PaimonHnswOptions, PaimonVectorIndex, PaimonVectorIndexBuilder, PaimonVectorIndexConfig,
        PaimonVectorIndexKind,
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
