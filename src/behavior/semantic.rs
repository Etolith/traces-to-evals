mod detection;
mod model;
mod projection;

#[cfg(feature = "llm-judge-openai")]
mod openai;

pub use detection::SemanticBehaviorDetector;
pub use model::{
    DEFAULT_SEMANTIC_BEHAVIOR_RUBRIC, DEFAULT_SEMANTIC_BEHAVIOR_RUBRIC_VERSION,
    SEMANTIC_BEHAVIOR_DETECTOR_ID, SEMANTIC_BEHAVIOR_DETECTOR_VERSION,
    SEMANTIC_BEHAVIOR_EVALUATION_SCHEMA_VERSION, SEMANTIC_BEHAVIOR_PROJECTION_SCHEMA_VERSION,
    SEMANTIC_BEHAVIOR_PROJECTION_VERSION, SemanticBehaviorDetectionRun, SemanticBehaviorEvaluation,
    SemanticBehaviorEvaluator, SemanticBehaviorFinalOutcome, SemanticBehaviorJudgment,
    SemanticBehaviorPolicy, SemanticBehaviorProjection, SemanticContentPolicy, SemanticEvidenceRef,
    SemanticFinalClaim, SemanticPolicyDecision, SemanticToolCall, SemanticVerdict,
};
pub use projection::SemanticBehaviorProjector;

#[cfg(feature = "llm-judge-openai")]
pub use openai::{OPENAI_SEMANTIC_BEHAVIOR_EVALUATOR_VERSION, OpenAiSemanticBehaviorEvaluator};

#[cfg(test)]
mod tests;
