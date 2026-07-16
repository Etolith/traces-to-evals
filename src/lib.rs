pub mod behavior;
pub mod calibration;
#[doc(hidden)]
pub mod cli;
pub mod clustering;
mod commands;
pub mod comparison;
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
    AgentTraceSource, AgentTurn, ApprovalBypassDetector, ApprovalOutcome,
    BEHAVIOR_ADAPTER_SCHEMA_VERSION, BEHAVIOR_FINDING_SCHEMA_VERSION,
    BEHAVIOR_INPUT_SCHEMA_VERSION, BehaviorAdapterConfig, BehaviorFinding, BehaviorFindingSink,
    BehaviorInputCoverageV1, BehaviorInputPrivacyV1, BehaviorInputProvenanceV1, BehaviorInputV1,
    BudgetRegressionGate, CandidateGenerator, CandidateReview, CandidateReviewDecision,
    ClaimedOutcomeStatus, DEFAULT_FINDING_PROJECTION_VERSION, DEFAULT_SEMANTIC_BEHAVIOR_RUBRIC,
    DEFAULT_SEMANTIC_BEHAVIOR_RUBRIC_VERSION, DETECTION_CHECKPOINT_SCHEMA_VERSION,
    DETECTION_REPORT_SCHEMA_VERSION, DETERMINISTIC_DETECTOR_VERSION, DetectionCheckpoint,
    DetectionReportV1, DetectionRunStats, DetectionRunner, DetectorCoverageV1,
    DetectorEvaluationStatusV1, DetectorProfileIdentityV1, DetectorProfileV1,
    DeterministicDetectorSet, EVAL_CANDIDATE_SCHEMA_VERSION, EVIDENCE_PACKET_SCHEMA_VERSION,
    EscalationStatus, EvalCandidate, EvalCandidateGenerator, EvalCandidateStatus, EvidencePacket,
    EvidencePacketBuilder, EvidenceRef, ExcessiveToolUsageDetector, ExecutionBudget,
    FINDING_PRESENTATION_SCHEMA_VERSION, FINDING_PROJECTION_SCHEMA_VERSION,
    FINDING_RECURRENCE_COMPARATOR_VERSION, FINDING_RECURRENCE_COMPARISON_SCHEMA_VERSION,
    FINDING_RECURRENCE_REQUEST_SCHEMA_VERSION, FalseSuccessClaimDetector, FinalOutcome,
    FinalOutcomeStatus, FindingCertaintyV1, FindingEvalCandidateGenerator, FindingEvidenceRoleV1,
    FindingKindRate, FindingPresentationV1, FindingPresenter, FindingProjection,
    FindingRecurrenceComparator, FindingRecurrenceComparison, FindingRecurrenceRequest,
    FindingRedactor, FindingSeverity, FindingWindow, FindingWriteStatus, IncidentRegressionGate,
    KNOWN_SIGNATURE_GROUP_SCHEMA_VERSION, KnownSignatureGroup, KnownSignatureGrouper,
    MissingResolutionDetector, NormalizedToolError, OpenInferenceBehaviorNormalizer,
    OperationEffect, OutcomeClaim, PAIRED_FINDING_VERIFICATION_SCHEMA_VERSION,
    PAIRED_FINDING_VERIFIER_VERSION, PairedEvaluationComparison, PairedEvaluationKey,
    PairedFindingVerification, PairedFindingVerifier, PolicyDecision, PolicyDecisionOutcome,
    PolicyViolationDetector, PopulationBasis, PresentedEvidenceV1,
    REMEDIATION_VERIFICATION_REQUEST_SCHEMA_VERSION, REMEDIATION_VERIFICATION_SCHEMA_VERSION,
    REMEDIATION_VERIFIER_VERSION, RecoveryAnalyzer, RecoveryStatus, RedactedCandidateInput,
    RemediationInputArtifacts, RemediationVerificationPolicy, RemediationVerificationReport,
    RemediationVerificationRequest, RemediationVerifier, RepeatedFailurePolicyV1,
    RepeatedToolFailureDetector, RetrySafety, RuleMatchCertaintyV1,
    SAFE_BEHAVIOR_PROJECTION_VERSION, SEMANTIC_BEHAVIOR_DETECTOR_ID,
    SEMANTIC_BEHAVIOR_DETECTOR_VERSION, SEMANTIC_BEHAVIOR_EVALUATION_SCHEMA_VERSION,
    SEMANTIC_BEHAVIOR_PROJECTION_SCHEMA_VERSION, SEMANTIC_BEHAVIOR_PROJECTION_VERSION,
    SafeFindingProjector, ScalarFindingRedactor, SemanticBehaviorDetectionRun,
    SemanticBehaviorDetector, SemanticBehaviorEvaluation, SemanticBehaviorEvaluator,
    SemanticBehaviorFinalOutcome, SemanticBehaviorJudgment, SemanticBehaviorPolicy,
    SemanticBehaviorProjection, SemanticBehaviorProjector, SemanticContentPolicy,
    SemanticEvidenceRef, SemanticFinalClaim, SemanticPolicyDecision, SemanticToolCall,
    SemanticVerdict, StateChangeRef, StateObservation, SuiteRegressionGate,
    TelemetryDiagnosticSeverityV1, TelemetryDiagnosticV1, TerminalToolFailureDetector,
    ToolCallFact, ToolCallLoopDetector, ToolCallStatus, ToolLoopPolicyV1, ToolRequirement,
    ToolSemanticMapping, ToolUsageBudgetV1, TraceDetector, TraceEnvelope,
    UncertainMutationStateDetector, UnresolvedEscalationDetector, VerificationArtifactDigest,
    VerificationGate, VerificationGateStatus, finding_projection_cases,
};
#[cfg(feature = "llm-judge-openai")]
pub use behavior::{OPENAI_SEMANTIC_BEHAVIOR_EVALUATOR_VERSION, OpenAiSemanticBehaviorEvaluator};
pub use clustering::{
    BruteForceVectorIndex, BruteForceVectorIndexBuilder, CaseEmbedding, ClusterAlgorithm,
    ClusterAssigner, ClusterAssignment, ClusterAssignmentRule, ClusterDiscovery,
    ClusterDiscoveryInput, ClusterDiscoveryOptions, ClusterLabel, ClusterLabelPayload,
    ClusterLabelPrompt, ClusterLabeler, ClusterModel, ClusterModelAssigner, ClusterModelSource,
    ClusterQuality, ClusterQualityEvaluation, ClusterQualityReport, ClusterRuleMatch, ClusterText,
    ClusterTextProjector, DefaultClusterTextProjector, DiscoveredCluster, DistanceMetric,
    EmbeddingClusterAssigner, EmbeddingProvider, EvalCluster, FnClusterAssignmentRule,
    KMeansClusterDiscovery, KeywordAssignmentRule, MetadataAssignmentRule, OwnedVectorRecord,
    ProjectedField, RuleBasedClusterAssigner, VectorIndex, VectorIndexBuilder,
    VectorIndexClusterAssigner, VectorIndexRow, VectorIndexRowMap, VectorMetric, VectorRecord,
    VectorRowId, VectorSearchHit, VectorSearchOptions, borrowed_records, case_embedding_records,
    cluster_centroid_records,
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
pub use comparison::{
    AlignedExecutionRow, AlignmentRelation, DivergenceSummary, ExecutionStep,
    StructuralTraceAligner, TRACE_COMPARISON_ENGINE_VERSION, TRACE_COMPARISON_SCHEMA_VERSION,
    TraceAlignmentOptions, TraceComparison, TraceComparisonInput,
};
pub use error::{Result, TraceEvalError};
pub use evaluation::{
    AsyncEvaluator, EvaluationCriteria, EvaluationResult, EvaluationRun, Evaluator, RunScore,
    ScoreScale, WeightedAggregate,
};
pub use model::{
    EvalCase, FactQuality, PayloadIdentity, SourceSpanStatus, Span, SpanEvent, SpanKind, SpanLink,
    SpanProvenance, Trace,
};
pub use project::{DEFAULT_PROJECT_NAME, ProjectName};
pub use report::{
    CalibrationImpact, ClusterIssue, ClusterScore, EvaluationReport, EvaluatorScore, FailedCase,
};
pub use validation::{ValidationIssue, ValidationProfile, ValidationReport, ValidationSeverity};

pub mod prelude {
    pub use crate::behavior::{
        AGENT_BEHAVIOR_TRACE_SCHEMA_VERSION, AgentBehaviorNormalizer, AgentBehaviorTrace,
        AgentRole, AgentTraceSource, AgentTurn, ApprovalBypassDetector, ApprovalOutcome,
        BEHAVIOR_ADAPTER_SCHEMA_VERSION, BEHAVIOR_FINDING_SCHEMA_VERSION,
        BEHAVIOR_INPUT_SCHEMA_VERSION, BehaviorAdapterConfig, BehaviorFinding, BehaviorFindingSink,
        BehaviorInputCoverageV1, BehaviorInputPrivacyV1, BehaviorInputProvenanceV1,
        BehaviorInputV1, BudgetRegressionGate, CandidateGenerator, CandidateReview,
        CandidateReviewDecision, ClaimedOutcomeStatus, DEFAULT_FINDING_PROJECTION_VERSION,
        DEFAULT_SEMANTIC_BEHAVIOR_RUBRIC, DEFAULT_SEMANTIC_BEHAVIOR_RUBRIC_VERSION,
        DETECTION_CHECKPOINT_SCHEMA_VERSION, DETECTION_REPORT_SCHEMA_VERSION,
        DETERMINISTIC_DETECTOR_VERSION, DetectionCheckpoint, DetectionReportV1, DetectionRunStats,
        DetectionRunner, DetectorCoverageV1, DetectorEvaluationStatusV1, DetectorProfileIdentityV1,
        DetectorProfileV1, DeterministicDetectorSet, EVAL_CANDIDATE_SCHEMA_VERSION,
        EVIDENCE_PACKET_SCHEMA_VERSION, EscalationStatus, EvalCandidate, EvalCandidateGenerator,
        EvalCandidateStatus, EvidencePacket, EvidencePacketBuilder, EvidenceRef,
        ExcessiveToolUsageDetector, ExecutionBudget, FINDING_PRESENTATION_SCHEMA_VERSION,
        FINDING_PROJECTION_SCHEMA_VERSION, FINDING_RECURRENCE_COMPARATOR_VERSION,
        FINDING_RECURRENCE_COMPARISON_SCHEMA_VERSION, FINDING_RECURRENCE_REQUEST_SCHEMA_VERSION,
        FalseSuccessClaimDetector, FinalOutcome, FinalOutcomeStatus, FindingCertaintyV1,
        FindingEvalCandidateGenerator, FindingEvidenceRoleV1, FindingKindRate,
        FindingPresentationV1, FindingPresenter, FindingProjection, FindingRecurrenceComparator,
        FindingRecurrenceComparison, FindingRecurrenceRequest, FindingRedactor, FindingSeverity,
        FindingWindow, FindingWriteStatus, IncidentRegressionGate,
        KNOWN_SIGNATURE_GROUP_SCHEMA_VERSION, KnownSignatureGroup, KnownSignatureGrouper,
        MissingResolutionDetector, NormalizedToolError, OpenInferenceBehaviorNormalizer,
        OperationEffect, OutcomeClaim, PAIRED_FINDING_VERIFICATION_SCHEMA_VERSION,
        PAIRED_FINDING_VERIFIER_VERSION, PairedEvaluationComparison, PairedEvaluationKey,
        PairedFindingVerification, PairedFindingVerifier, PolicyDecision, PolicyDecisionOutcome,
        PolicyViolationDetector, PopulationBasis, PresentedEvidenceV1,
        REMEDIATION_VERIFICATION_REQUEST_SCHEMA_VERSION, REMEDIATION_VERIFICATION_SCHEMA_VERSION,
        REMEDIATION_VERIFIER_VERSION, RecoveryAnalyzer, RecoveryStatus, RedactedCandidateInput,
        RemediationInputArtifacts, RemediationVerificationPolicy, RemediationVerificationReport,
        RemediationVerificationRequest, RemediationVerifier, RepeatedFailurePolicyV1,
        RepeatedToolFailureDetector, RetrySafety, RuleMatchCertaintyV1,
        SAFE_BEHAVIOR_PROJECTION_VERSION, SEMANTIC_BEHAVIOR_DETECTOR_ID,
        SEMANTIC_BEHAVIOR_DETECTOR_VERSION, SEMANTIC_BEHAVIOR_EVALUATION_SCHEMA_VERSION,
        SEMANTIC_BEHAVIOR_PROJECTION_SCHEMA_VERSION, SEMANTIC_BEHAVIOR_PROJECTION_VERSION,
        SafeFindingProjector, ScalarFindingRedactor, SemanticBehaviorDetectionRun,
        SemanticBehaviorDetector, SemanticBehaviorEvaluation, SemanticBehaviorEvaluator,
        SemanticBehaviorFinalOutcome, SemanticBehaviorJudgment, SemanticBehaviorPolicy,
        SemanticBehaviorProjection, SemanticBehaviorProjector, SemanticContentPolicy,
        SemanticEvidenceRef, SemanticFinalClaim, SemanticPolicyDecision, SemanticToolCall,
        SemanticVerdict, StateChangeRef, StateObservation, SuiteRegressionGate,
        TelemetryDiagnosticSeverityV1, TelemetryDiagnosticV1, TerminalToolFailureDetector,
        ToolCallFact, ToolCallLoopDetector, ToolCallStatus, ToolLoopPolicyV1, ToolRequirement,
        ToolSemanticMapping, ToolUsageBudgetV1, TraceDetector, TraceEnvelope,
        UncertainMutationStateDetector, UnresolvedEscalationDetector, VerificationArtifactDigest,
        VerificationGate, VerificationGateStatus, finding_projection_cases,
    };
    #[cfg(feature = "llm-judge-openai")]
    pub use crate::behavior::{
        OPENAI_SEMANTIC_BEHAVIOR_EVALUATOR_VERSION, OpenAiSemanticBehaviorEvaluator,
    };
    pub use crate::clustering::{
        BruteForceVectorIndex, BruteForceVectorIndexBuilder, CaseEmbedding, ClusterAlgorithm,
        ClusterAssigner, ClusterAssignment, ClusterAssignmentRule, ClusterDiscovery,
        ClusterDiscoveryInput, ClusterDiscoveryOptions, ClusterLabel, ClusterLabelPayload,
        ClusterLabelPrompt, ClusterLabeler, ClusterModel, ClusterModelAssigner, ClusterModelSource,
        ClusterQuality, ClusterQualityEvaluation, ClusterQualityReport, ClusterRuleMatch,
        ClusterText, ClusterTextProjector, DefaultClusterTextProjector, DiscoveredCluster,
        DistanceMetric, EmbeddingClusterAssigner, EmbeddingProvider, EvalCluster,
        FnClusterAssignmentRule, KMeansClusterDiscovery, KeywordAssignmentRule,
        MetadataAssignmentRule, OwnedVectorRecord, ProjectedField, RuleBasedClusterAssigner,
        VectorIndex, VectorIndexBuilder, VectorIndexClusterAssigner, VectorIndexRow,
        VectorIndexRowMap, VectorMetric, VectorRecord, VectorRowId, VectorSearchHit,
        VectorSearchOptions, borrowed_records, case_embedding_records, cluster_centroid_records,
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
    pub use crate::model::{
        EvalCase, FactQuality, PayloadIdentity, SourceSpanStatus, Span, SpanEvent, SpanKind,
        SpanLink, SpanProvenance, Trace,
    };
    pub use crate::project::{DEFAULT_PROJECT_NAME, ProjectName};
    pub use crate::report::{
        CalibrationImpact, ClusterIssue, ClusterScore, EvaluationReport, EvaluatorScore, FailedCase,
    };
    pub use crate::validation::{
        ValidationIssue, ValidationProfile, ValidationReport, ValidationSeverity,
    };
}
