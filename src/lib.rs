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
#[cfg(feature = "llm-judge-openai")]
#[doc(hidden)]
pub mod providers;
pub mod report;
pub mod validation;

pub use clustering::{
    ClusterAssigner, ClusterAssignment, ClusterAssignmentRule, ClusterRuleMatch, EvalCluster,
    FnClusterAssignmentRule, KeywordAssignmentRule, MetadataAssignmentRule,
    RuleBasedClusterAssigner,
};
pub use error::{Result, TraceEvalError};
pub use evaluation::{
    AsyncEvaluator, EvaluationCriteria, EvaluationResult, EvaluationRun, Evaluator, RunScore,
    ScoreScale, WeightedAggregate,
};
pub use model::{EvalCase, Span, SpanKind, Trace};
pub use report::{
    CalibrationImpact, ClusterIssue, ClusterScore, EvaluationReport, EvaluatorScore, FailedCase,
};
pub use validation::{ValidationIssue, ValidationProfile, ValidationReport, ValidationSeverity};

pub mod prelude {
    pub use crate::clustering::{
        ClusterAssigner, ClusterAssignment, ClusterAssignmentRule, ClusterRuleMatch, EvalCluster,
        FnClusterAssignmentRule, KeywordAssignmentRule, MetadataAssignmentRule,
        RuleBasedClusterAssigner,
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
    pub use crate::report::{
        CalibrationImpact, ClusterIssue, ClusterScore, EvaluationReport, EvaluatorScore, FailedCase,
    };
    pub use crate::validation::{
        ValidationIssue, ValidationProfile, ValidationReport, ValidationSeverity,
    };
}
