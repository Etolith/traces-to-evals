mod agreement;
mod calibration;
mod compact_projector;
mod task_completion;

pub use traceeval_contracts::*;

pub use agreement::{AgreementLabelScaleV1, AgreementRatingV1, HumanAgreementReportV1};
pub use calibration::{
    BINARY_CALIBRATION_MODEL_SCHEMA_VERSION, BinaryCalibrationExampleV1,
    BinaryCalibrationFitOptionsV1, BinaryCalibrationModelV1, BinaryCalibrationReportV1,
    BinaryPredictionV1, BinomialRateIntervalV1, CalibrationBinV1, CalibrationDataSplitV1,
    ConfusionMatrixV1, GROUPED_BOOTSTRAP_MACRO_F1_ITERATIONS_V1,
    GROUPED_BOOTSTRAP_MACRO_F1_METHOD_V1, GROUPED_BOOTSTRAP_MACRO_F1_SEED_V1,
    GroupedBootstrapIntervalV1, LearnedCalibrationFeaturesV1, SelectiveRiskPointV1,
};
pub use compact_projector::{
    COMPACT_TASK_COMPLETION_PROJECTOR_VERSION, CompactTaskCompletionProjector,
    CompactTaskCompletionProjectorError, DEFAULT_COMPACT_TASK_COMPLETION_RUBRIC,
    TaskCompletionTokenCounter,
};
#[cfg(feature = "llm-judge-openai")]
pub use task_completion::OpenAiTaskCompletionEvaluator;
pub use task_completion::{
    TASK_COMPLETION_EVIDENCE_SYSTEM_PROMPT_V2, TASK_COMPLETION_JUDGMENT_SCHEMA_VERSION,
    TASK_COMPLETION_PROJECTION_SCHEMA_VERSION, TASK_COMPLETION_PROJECTOR_VERSION,
    TaskCompletionCapabilityV1, TaskCompletionContentPolicyV1, TaskCompletionCriterionJudgmentV1,
    TaskCompletionCriterionOutcomeV1, TaskCompletionCriterionSpecV1, TaskCompletionDeclaredFieldV1,
    TaskCompletionEvaluator, TaskCompletionExecutionV1, TaskCompletionJudgmentV1,
    TaskCompletionOutcomeV1, TaskCompletionProjectionV1, TaskCompletionProjectorV1,
    TaskCompletionToolObservationV1, TaskCompletionTraceObservationV1,
    task_completion_judgment_response_schema,
};
fn require_non_empty(
    value: &str,
    field: &str,
    error: fn(String) -> ContractError,
) -> Result<(), ContractError> {
    if value.trim().is_empty() {
        return Err(error(format!("{field} must not be empty")));
    }
    Ok(())
}

fn require_sha256(
    value: &str,
    field: &str,
    error: fn(String) -> ContractError,
) -> Result<(), ContractError> {
    let digest = value.strip_prefix("sha256:").unwrap_or_default();
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(error(format!("{field} must be a sha256: content identity")));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
