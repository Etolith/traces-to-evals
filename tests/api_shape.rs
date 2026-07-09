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

#[test]
fn public_api_composes_cluster_discovery_primitives() -> Result<()> {
    let project = ProjectName::new("acme-evals")?;
    let case = EvalCase::new("case-1", "trace-1", "reset my password")
        .with_expected_output("Use the reset link")
        .with_actual_output("irrelevant output for projection");

    let projector = DefaultClusterTextProjector::new().with_project_name(project.clone());
    let projected = projector.project_case(&case);
    assert!(!projected.text.contains("actual_output"));

    let embedding = CaseEmbedding::new_with_project(
        &project,
        &projected,
        "test-provider",
        "test-model",
        vec![1.0, 0.0],
        projector.projection_version(),
    );
    embedding.validate()?;
    assert_eq!(embedding.schema_version, "acme-evals.case_embedding.v1");

    let cluster = DiscoveredCluster::new("cluster-0001", 1, vec![case.id.clone()])
        .with_centroid(vec![1.0, 0.0]);
    let quality = ClusterQualityReport {
        cluster_count: 1,
        assigned_case_count: 1,
        mean_distance: Some(0.0),
        silhouette_score: None,
        clusters: vec![cluster.quality.clone()],
    };
    let model = ClusterModel::new_with_project(
        &project,
        "model-1",
        "2026-01-01T00:00:00Z",
        ClusterModelSource {
            case_count: 1,
            embedding_provider: Some("test-provider".to_string()),
            embedding_model: Some("test-model".to_string()),
            embedding_dimensions: Some(2),
            projection_version: Some(projector.projection_version()),
            algorithm: "manual".to_string(),
            distance_metric: "cosine".to_string(),
            random_seed: 42,
        },
        vec![cluster],
        Vec::new(),
        quality,
    );
    model.validate()?;
    assert_eq!(model.schema_version, "acme-evals.cluster_model.v1");

    let assignment =
        ClusterModelAssigner::new(model).assign_case_embedding(&case, &embedding.vector)?;
    assert_eq!(assignment.cluster_id, "cluster-0001");
    assert_eq!(assignment.method, "embedding_nearest_centroid");
    assert_eq!(assignment.distance, Some(0.0));
    assert!(!assignment.novelty);

    Ok(())
}
