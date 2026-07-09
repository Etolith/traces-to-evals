use std::collections::BTreeMap;

use serde_json::Value;

use traces_to_evals::calibration::{CalibrationModel, HumanRating};
use traces_to_evals::io::jsonl::{JsonlReader, JsonlWriter};
use traces_to_evals::prelude::*;

#[derive(Debug, Default)]
struct DomainEvaluator;

impl Evaluator for DomainEvaluator {
    fn evaluator_name(&self) -> String {
        "domain".to_string()
    }

    fn evaluate_case(&self, case: &EvalCase) -> Result<EvaluationResult> {
        Ok(EvaluationResult::new(
            case,
            Evaluator::evaluator_name(self),
            0.75,
            ScoreScale::Unit,
            true,
            "domain-specific evaluator accepted the answer",
        )
        .with_cluster_id("arithmetic")
        .with_criteria(EvaluationCriteria {
            relevance: true,
            correctness: true,
            completeness: true,
            safety: true,
        }))
    }
}

#[test]
fn public_api_exposes_typed_errors_for_common_failures() {
    let case = EvalCase::new("case-1", "trace-1", "input");
    let error = NonEmptyOutputGrader.grade(&case).unwrap_err();

    assert!(matches!(
        error,
        TraceEvalError::MissingActualOutput { case_id } if case_id == "case-1"
    ));

    let error = CalibrationModel::fit(&[], &[], 3).unwrap_err();
    assert!(matches!(error, TraceEvalError::CalibrationOverlap));
}

#[test]
fn public_api_composes_trace_extraction_evaluation_calibration_and_aggregation() -> Result<()> {
    let traces = vec![
        Trace::new("trace-1")
            .with_span(Span::llm("span-1", "prompt").with_input("What is 2 + 2?"))
            .with_span(Span::llm("span-2", "completion").with_output("4")),
    ];

    let mut run = EvaluationRun::from_traces(&SimpleExtractor, &traces)?;
    run.cases[0].expected_output = Some("4".to_string());
    run.cases[0].metadata.insert(
        "cluster_id".to_string(),
        Value::String("arithmetic".to_string()),
    );
    let case_id = run.cases[0].id.clone();
    let trace_id = run.cases[0].trace_id.clone();

    let assigner = RuleBasedClusterAssigner::empty(vec![EvalCluster {
        id: "arithmetic".to_string(),
        label: "Arithmetic".to_string(),
        description: None,
        weight: 1.5,
        metadata: BTreeMap::new(),
    }])
    .with_rule(FnClusterAssignmentRule::new(
        "custom_math_input",
        |case, _clusters| {
            if case.input.contains("2 + 2") {
                Some(ClusterRuleMatch::new("arithmetic", 0.95))
            } else {
                None
            }
        },
    ))
    .with_rule(MetadataAssignmentRule::new());
    let assignments = assigner.assign_cases(&run.cases)?;
    assert_eq!(assignments[0].cluster_id, "arithmetic");
    assert_eq!(assignments[0].method, "custom_math_input");

    let run = run
        .evaluate_with(&NonEmptyOutputGrader)?
        .evaluate_with(&ExactMatchGrader)?
        .evaluate_with(&DomainEvaluator)?;

    let ratings = vec![HumanRating {
        case_id,
        trace_id,
        score: 4,
        passed: None,
        notes: None,
    }];
    let calibration = CalibrationModel::fit(&ratings, run.results(), 3)?;
    let run = calibration.apply_run(run);

    let score = run.aggregate_with(
        &WeightedAggregate::new()
            .with_evaluator_weight("exact_match", 2.0)
            .with_cluster_weight("arithmetic", 1.5),
    );

    assert_eq!(run.cases.len(), 1);
    assert_eq!(run.results().len(), 3);
    assert_eq!(score.result_count, 3);
    assert_eq!(score.passed_count, 3);
    assert!(score.weighted_score > 0.0);

    let report = EvaluationReport::from_results_with_aggregate(
        run.results(),
        &WeightedAggregate::new().with_cluster_weight("arithmetic", 1.5),
    );
    assert_eq!(report.total_cases, 1);
    assert_eq!(report.evaluator_scores.len(), 3);

    let mut buffer = Vec::new();
    JsonlWriter::new(&mut buffer).write_iter(run.results().iter())?;
    let round_tripped: Vec<EvaluationResult> =
        JsonlReader::new(buffer.as_slice(), "memory").read_all()?;

    assert_eq!(round_tripped.len(), run.results().len());
    Ok(())
}
