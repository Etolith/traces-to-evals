pub mod calibration;
pub mod cli;
pub mod clustering;
mod commands;
pub mod evaluation;
pub mod export;
pub mod exporters;
pub mod extractors;
pub mod graders;
pub mod io;
pub mod judge;
pub mod model;
#[cfg(feature = "llm-judge-openai")]
pub mod providers;
pub mod report;
pub mod scoring;
pub mod validation;

pub use clustering::{
    ClusterAssigner, ClusterAssignment, ClusterAssignmentRule, ClusterRuleMatch, EvalCluster,
    FnClusterAssignmentRule, KeywordAssignmentRule, MetadataAssignmentRule,
    RuleBasedClusterAssigner,
};
#[allow(deprecated)]
pub use clustering::{Clusterer, MetadataClusterer};
pub use evaluation::{
    AsyncEvaluator, EvaluationCriteria, EvaluationResult, EvaluationRun, Evaluator, RunScore,
    ScoreScale, WeightedAggregate,
};
pub use model::{EvalCase, Span, SpanKind, Trace};
pub use report::{ClusterScore, EvaluationReport, EvaluatorScore};

pub mod prelude {
    pub use crate::clustering::{
        ClusterAssigner, ClusterAssignment, ClusterAssignmentRule, ClusterRuleMatch, EvalCluster,
        FnClusterAssignmentRule, KeywordAssignmentRule, MetadataAssignmentRule,
        RuleBasedClusterAssigner,
    };
    pub use crate::evaluation::{
        AsyncEvaluator, EvaluationCriteria, EvaluationResult, EvaluationRun, Evaluator, RunScore,
        ScoreScale, WeightedAggregate,
    };
    pub use crate::extractors::{EvalCaseExtractor, OpenInferenceExtractor, SimpleExtractor};
    pub use crate::graders::{
        ContainsGrader, DeterministicGrader, ExactMatchGrader, NonEmptyOutputGrader,
    };
    pub use crate::model::{EvalCase, Span, SpanKind, Trace};
    pub use crate::report::{ClusterScore, EvaluationReport, EvaluatorScore};
}
