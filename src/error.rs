pub type Result<T> = std::result::Result<T, TraceEvalError>;

#[derive(Debug, thiserror::Error)]
pub enum TraceEvalError {
    #[error("invalid eval case {case_id}: {message}")]
    InvalidCase { case_id: String, message: String },

    #[error("missing required actual_output for case {case_id}")]
    MissingActualOutput { case_id: String },

    #[error("missing required expected_output for case {case_id}")]
    MissingExpectedOutput { case_id: String },

    #[error("trace {trace_id} does not contain input for {extractor}")]
    MissingTraceInput { trace_id: String, extractor: String },

    #[error("span {span_id} has no input")]
    MissingSpanInput { span_id: String },

    #[error("invalid score {score} for scale {scale}")]
    InvalidScore { score: f32, scale: String },

    #[error("invalid threshold {threshold} for scale {scale}")]
    InvalidThreshold { threshold: u8, scale: String },

    #[error("cannot calibrate without overlapping case IDs")]
    CalibrationOverlap,

    #[error("cluster assignment failed for case {case_id}: {message}")]
    ClusterAssignment { case_id: String, message: String },

    #[error("invalid embedding for case {case_id}: {message}")]
    InvalidEmbedding { case_id: String, message: String },

    #[error("embedding provider {provider} failed: {message}")]
    EmbeddingProvider { provider: String, message: String },

    #[error("cluster discovery with {algorithm} failed: {message}")]
    ClusterDiscovery { algorithm: String, message: String },

    #[error("cluster labeling with {provider} failed for cluster {cluster_id}: {message}")]
    ClusterLabeling {
        provider: String,
        cluster_id: String,
        message: String,
    },

    #[error("cluster model {model_id} is invalid: {message}")]
    ClusterModelValidation { model_id: String, message: String },

    #[error("invalid project name {name:?}: {message}")]
    InvalidProjectName { name: String, message: String },

    #[error("validation failed with {error_count} errors and {warning_count} warnings")]
    ValidationFailed {
        error_count: usize,
        warning_count: usize,
    },

    #[error("provider error: {0}")]
    Provider(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Csv(#[from] csv::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl TraceEvalError {
    pub fn invalid_case(case_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidCase {
            case_id: case_id.into(),
            message: message.into(),
        }
    }

    pub fn invalid_score(score: impl Into<f32>, scale: impl Into<String>) -> Self {
        Self::InvalidScore {
            score: score.into(),
            scale: scale.into(),
        }
    }
}
