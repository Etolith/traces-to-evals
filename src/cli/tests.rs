use clap::Parser;

use super::*;

#[test]
fn parses_judge_grader_args() {
    let cli = Cli::parse_from([
        "traceeval",
        "grade",
        "--cases",
        "cases.jsonl",
        "--judge",
        "openai-dive",
        "--model",
        "gpt-4o",
        "--out",
        "out.jsonl",
    ]);

    let Command::Grade(args) = cli.command else {
        panic!("expected grade command");
    };

    assert_eq!(args.cases, PathBuf::from("cases.jsonl"));
    assert_eq!(args.judge, Some(JudgeProviderName::OpenaiDive));
    assert_eq!(args.model.as_deref(), Some("gpt-4o"));
    assert_eq!(args.out, PathBuf::from("out.jsonl"));
}

#[test]
fn rejects_missing_value_for_cases() {
    let result = Cli::try_parse_from([
        "traceeval",
        "grade",
        "--cases",
        "--judge",
        "openai-dive",
        "--model",
        "gpt-4o",
        "--out",
        "out.jsonl",
    ]);

    assert!(result.is_err());
}

#[test]
fn parses_detect_outputs_and_limits() {
    let cli = Cli::parse_from([
        "traceeval",
        "detect",
        "--traces",
        "traces.jsonl",
        "--out",
        "findings.jsonl",
        "--adapter-config",
        "adapter.json",
        "--normalized-out",
        "behavior.jsonl",
        "--candidates-out",
        "candidates.jsonl",
        "--evidence-packet-out",
        "evidence-packet.json",
        "--projections-out",
        "projections.jsonl",
        "--projection-cases-out",
        "projection-cases.jsonl",
        "--signature-groups-out",
        "signature-groups.jsonl",
        "--projection-metadata-key",
        "intent",
        "--max-tool-calls",
        "12",
    ]);

    let Command::Detect(args) = cli.command else {
        panic!("expected detect command");
    };

    assert_eq!(args.format, BehaviorTraceFormat::OpenInference);
    assert_eq!(args.max_tool_calls, 12);
    assert_eq!(args.adapter_config, Some(PathBuf::from("adapter.json")));
    assert_eq!(args.normalized_out, Some(PathBuf::from("behavior.jsonl")));
    assert_eq!(args.candidates_out, Some(PathBuf::from("candidates.jsonl")));
    assert_eq!(
        args.evidence_packet_out,
        Some(PathBuf::from("evidence-packet.json"))
    );
    assert_eq!(
        args.projections_out,
        Some(PathBuf::from("projections.jsonl"))
    );
    assert_eq!(
        args.projection_cases_out,
        Some(PathBuf::from("projection-cases.jsonl"))
    );
    assert_eq!(
        args.signature_groups_out,
        Some(PathBuf::from("signature-groups.jsonl"))
    );
    assert_eq!(args.projection_metadata_keys, ["intent"]);
}

#[test]
fn parses_opt_in_semantic_detection_args() {
    let cli = Cli::parse_from([
        "traceeval",
        "detect",
        "--traces",
        "traces.jsonl",
        "--out",
        "findings.jsonl",
        "--semantic-judge",
        "openai-dive",
        "--semantic-model",
        "gpt-test",
        "--semantic-results-out",
        "semantic-results.jsonl",
        "--semantic-projections-out",
        "semantic-projections.jsonl",
        "--semantic-content",
        "pre-redacted-summaries",
        "--semantic-rubric-version",
        "acme.support.v2",
        "--semantic-min-confidence",
        "0.9",
    ]);

    let Command::Detect(args) = cli.command else {
        panic!("expected detect command");
    };
    assert_eq!(args.semantic_judge, Some(JudgeProviderName::OpenaiDive));
    assert_eq!(args.semantic_model.as_deref(), Some("gpt-test"));
    assert_eq!(
        args.semantic_content,
        SemanticContentPolicyName::PreRedactedSummaries
    );
    assert_eq!(args.semantic_rubric_version, "acme.support.v2");
    assert_eq!(args.semantic_min_confidence, 0.9);
    assert_eq!(
        args.semantic_results_out,
        Some(PathBuf::from("semantic-results.jsonl"))
    );
}

#[test]
fn rejects_semantic_judge_without_model() {
    let result = Cli::try_parse_from([
        "traceeval",
        "detect",
        "--traces",
        "traces.jsonl",
        "--out",
        "findings.jsonl",
        "--semantic-judge",
        "openai-dive",
    ]);

    assert!(result.is_err());
}

#[test]
fn parses_combined_remediation_verification_inputs() {
    let cli = Cli::parse_from([
        "traceeval",
        "verify-remediation",
        "--request",
        "request.json",
        "--baseline-findings",
        "baseline-findings.jsonl",
        "--candidate-findings",
        "candidate-findings.jsonl",
        "--baseline-results",
        "baseline-results.jsonl",
        "--candidate-results",
        "candidate-results.jsonl",
        "--out",
        "verification.json",
    ]);

    let Command::VerifyRemediation(args) = cli.command else {
        panic!("expected verify-remediation command");
    };
    assert_eq!(args.request, PathBuf::from("request.json"));
    assert_eq!(
        args.candidate_results,
        PathBuf::from("candidate-results.jsonl")
    );
    assert_eq!(args.out, PathBuf::from("verification.json"));
}

#[test]
fn parses_recurrence_comparison_inputs() {
    let cli = Cli::parse_from([
        "traceeval",
        "compare-recurrence",
        "--request",
        "recurrence-request.json",
        "--baseline-findings",
        "baseline-findings.jsonl",
        "--candidate-findings",
        "candidate-findings.jsonl",
        "--out",
        "comparison.json",
    ]);

    let Command::CompareRecurrence(args) = cli.command else {
        panic!("expected compare-recurrence command");
    };
    assert_eq!(args.request, PathBuf::from("recurrence-request.json"));
    assert_eq!(
        args.baseline_findings,
        PathBuf::from("baseline-findings.jsonl")
    );
    assert_eq!(args.out, PathBuf::from("comparison.json"));
}

#[test]
fn parses_cluster_assign_with_model_source() {
    let cli = Cli::parse_from([
        "traceeval",
        "cluster",
        "assign",
        "--cases",
        "cases.jsonl",
        "--model",
        "cluster_model.json",
        "--embeddings",
        "embeddings.jsonl",
        "--out",
        "assignments.jsonl",
    ]);

    let Command::Cluster(cluster_args) = cli.command else {
        panic!("expected cluster command");
    };
    let ClusterCommand::Assign(args) = cluster_args.command else {
        panic!("expected cluster assign command");
    };

    assert_eq!(args.cases, PathBuf::from("cases.jsonl"));
    assert_eq!(args.model, Some(PathBuf::from("cluster_model.json")));
    assert_eq!(args.embeddings, Some(PathBuf::from("embeddings.jsonl")));
    assert_eq!(args.out, PathBuf::from("assignments.jsonl"));
}

#[test]
fn parses_repeatable_cluster_assignment_metadata_keys() {
    let cli = Cli::parse_from([
        "traceeval",
        "cluster",
        "assign",
        "--cases",
        "cases.jsonl",
        "--clusters",
        "clusters.jsonl",
        "--metadata-key",
        "task_type",
        "--metadata-key",
        "product_area",
        "--out",
        "assignments.jsonl",
    ]);

    let Command::Cluster(cluster_args) = cli.command else {
        panic!("expected cluster command");
    };
    let ClusterCommand::Assign(args) = cluster_args.command else {
        panic!("expected cluster assign command");
    };

    assert_eq!(args.metadata_keys, ["task_type", "product_area"]);
}

#[test]
fn rejects_cluster_assign_without_source() {
    let result = Cli::try_parse_from([
        "traceeval",
        "cluster",
        "assign",
        "--cases",
        "cases.jsonl",
        "--out",
        "assignments.jsonl",
    ]);

    assert!(result.is_err());
}

#[test]
fn parses_cluster_embed_project_and_dimensions_args() {
    let cli = Cli::parse_from([
        "traceeval",
        "cluster",
        "embed",
        "--cases",
        "cases.jsonl",
        "--provider",
        "openai",
        "--model",
        "text-embedding-3-small",
        "--dimensions",
        "512",
        "--project-name",
        "acme-evals",
        "--out",
        "embeddings.jsonl",
    ]);

    let Command::Cluster(cluster_args) = cli.command else {
        panic!("expected cluster command");
    };
    let ClusterCommand::Embed(args) = cluster_args.command else {
        panic!("expected cluster embed command");
    };

    assert_eq!(args.cases, PathBuf::from("cases.jsonl"));
    assert_eq!(args.provider, ClusterEmbeddingProviderName::Openai);
    assert_eq!(args.model, "text-embedding-3-small");
    assert_eq!(args.dimensions, Some(512));
    assert_eq!(args.project_name.as_deref(), Some("acme-evals"));
    assert_eq!(args.out, PathBuf::from("embeddings.jsonl"));
}
