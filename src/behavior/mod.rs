mod adapter;
mod candidate;
mod detectors;
mod evidence;
mod grouping;
mod model;
mod normalizer;
mod projection;
mod recurrence;
mod runner;
mod semantic;
mod verification;

pub use adapter::{BEHAVIOR_ADAPTER_SCHEMA_VERSION, BehaviorAdapterConfig, ToolSemanticMapping};
pub use candidate::{EvalCandidateGenerator, FindingEvalCandidateGenerator};
pub use detectors::{
    ApprovalBypassDetector, DETERMINISTIC_DETECTOR_VERSION, DeterministicDetectorSet,
    ExcessiveToolUsageDetector, FalseSuccessClaimDetector, MissingResolutionDetector,
    PolicyViolationDetector, RecoveryAnalyzer, RepeatedToolFailureDetector,
    TerminalToolFailureDetector, ToolCallLoopDetector, TraceDetector,
    UncertainMutationStateDetector, UnresolvedEscalationDetector,
};
pub use evidence::{EVIDENCE_PACKET_SCHEMA_VERSION, EvidencePacket, EvidencePacketBuilder};
pub use grouping::{
    KNOWN_SIGNATURE_GROUP_SCHEMA_VERSION, KnownSignatureGroup, KnownSignatureGrouper,
};
pub use model::{
    AGENT_BEHAVIOR_TRACE_SCHEMA_VERSION, AgentBehaviorTrace, AgentRole, AgentTurn, ApprovalOutcome,
    BEHAVIOR_FINDING_SCHEMA_VERSION, BehaviorFinding, CandidateGenerator, CandidateReview,
    CandidateReviewDecision, ClaimedOutcomeStatus, EVAL_CANDIDATE_SCHEMA_VERSION, EscalationStatus,
    EvalCandidate, EvalCandidateStatus, EvidenceRef, FinalOutcome, FinalOutcomeStatus,
    FindingSeverity, NormalizedToolError, OperationEffect, OutcomeClaim, PolicyDecision,
    PolicyDecisionOutcome, RecoveryStatus, RedactedCandidateInput, RetrySafety, StateChangeRef,
    StateObservation, ToolCallFact, ToolCallStatus, ToolRequirement,
};
pub use normalizer::{AgentBehaviorNormalizer, OpenInferenceBehaviorNormalizer};
pub use projection::{
    DEFAULT_FINDING_PROJECTION_VERSION, FINDING_PROJECTION_SCHEMA_VERSION, FindingProjection,
    FindingRedactor, SafeFindingProjector, ScalarFindingRedactor, finding_projection_cases,
};
pub use recurrence::{
    FINDING_RECURRENCE_COMPARATOR_VERSION, FINDING_RECURRENCE_COMPARISON_SCHEMA_VERSION,
    FINDING_RECURRENCE_REQUEST_SCHEMA_VERSION, FindingKindRate, FindingRecurrenceComparator,
    FindingRecurrenceComparison, FindingRecurrenceRequest, FindingWindow, PopulationBasis,
};
pub use runner::{
    AgentTraceSource, BehaviorFindingSink, DETECTION_CHECKPOINT_SCHEMA_VERSION,
    DetectionCheckpoint, DetectionRunStats, DetectionRunner, FindingWriteStatus, TraceEnvelope,
};
pub use semantic::{
    DEFAULT_SEMANTIC_BEHAVIOR_RUBRIC, DEFAULT_SEMANTIC_BEHAVIOR_RUBRIC_VERSION,
    SEMANTIC_BEHAVIOR_DETECTOR_ID, SEMANTIC_BEHAVIOR_DETECTOR_VERSION,
    SEMANTIC_BEHAVIOR_EVALUATION_SCHEMA_VERSION, SEMANTIC_BEHAVIOR_PROJECTION_SCHEMA_VERSION,
    SEMANTIC_BEHAVIOR_PROJECTION_VERSION, SemanticBehaviorDetectionRun, SemanticBehaviorDetector,
    SemanticBehaviorEvaluation, SemanticBehaviorEvaluator, SemanticBehaviorFinalOutcome,
    SemanticBehaviorJudgment, SemanticBehaviorPolicy, SemanticBehaviorProjection,
    SemanticBehaviorProjector, SemanticContentPolicy, SemanticEvidenceRef, SemanticFinalClaim,
    SemanticPolicyDecision, SemanticToolCall, SemanticVerdict,
};
#[cfg(feature = "llm-judge-openai")]
pub use semantic::{OPENAI_SEMANTIC_BEHAVIOR_EVALUATOR_VERSION, OpenAiSemanticBehaviorEvaluator};
pub use verification::{
    BudgetRegressionGate, ExecutionBudget, IncidentRegressionGate,
    PAIRED_FINDING_VERIFICATION_SCHEMA_VERSION, PAIRED_FINDING_VERIFIER_VERSION,
    PairedEvaluationComparison, PairedEvaluationKey, PairedFindingVerification,
    PairedFindingVerifier, REMEDIATION_VERIFICATION_REQUEST_SCHEMA_VERSION,
    REMEDIATION_VERIFICATION_SCHEMA_VERSION, REMEDIATION_VERIFIER_VERSION,
    RemediationInputArtifacts, RemediationVerificationPolicy, RemediationVerificationReport,
    RemediationVerificationRequest, RemediationVerifier, SuiteRegressionGate,
    VerificationArtifactDigest, VerificationGate, VerificationGateStatus,
};
