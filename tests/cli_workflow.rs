use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use tempfile::tempdir;

fn repo_path(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn traceeval() -> Command {
    Command::new(env!("CARGO_BIN_EXE_traceeval"))
}

fn assert_success(output: Output) {
    assert!(
        output.status.success(),
        "command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_detects_agent_findings_and_emits_unreviewed_candidates() {
    let dir = tempdir().unwrap();
    let normalized = dir.path().join("normalized.jsonl");
    let findings = dir.path().join("findings.jsonl");
    let candidates = dir.path().join("candidates.jsonl");

    assert_success(
        traceeval()
            .args(["detect", "--traces"])
            .arg(repo_path("fixtures/behavior/traces.jsonl"))
            .args(["--normalized-out"])
            .arg(&normalized)
            .args(["--candidates-out"])
            .arg(&candidates)
            .args(["--out"])
            .arg(&findings)
            .output()
            .unwrap(),
    );

    let normalized_rows = std::fs::read_to_string(normalized).unwrap();
    assert_eq!(normalized_rows.lines().count(), 4);
    assert!(normalized_rows.contains("\"status\":\"timed_out\""));
    assert!(!normalized_rows.contains("request timed out after dispatch"));

    let finding_rows = std::fs::read_to_string(findings).unwrap();
    let finding_values = finding_rows
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    let detector_ids = finding_values
        .iter()
        .map(|finding| finding["detector_id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(finding_values.len(), 5);
    assert!(detector_ids.contains(&"terminal_tool_failure"));
    assert!(detector_ids.contains(&"uncertain_mutation_state"));
    assert!(detector_ids.contains(&"false_success_claim"));
    assert!(detector_ids.contains(&"missing_resolution"));
    assert!(finding_values.iter().all(|finding| {
        finding["finding_id"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    }));

    let candidate_values = std::fs::read_to_string(candidates)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(candidate_values.len(), finding_values.len());
    assert!(
        candidate_values
            .iter()
            .all(|candidate| candidate["status"] == "candidate")
    );
}

#[test]
fn cli_runs_extract_grade_calibrate_cluster_report_workflow() {
    let dir = tempdir().unwrap();
    let cases = dir.path().join("cases.jsonl");
    let validation = dir.path().join("validation.json");
    let results = dir.path().join("results.jsonl");
    let calibration = dir.path().join("calibration.json");
    let assignments = dir.path().join("assignments.jsonl");
    let clustered_results = dir.path().join("clustered_results.jsonl");
    let report = dir.path().join("report.json");

    assert_success(
        traceeval()
            .args(["extract", "--format", "openinference", "--traces"])
            .arg(repo_path("fixtures/openinference/traces.jsonl"))
            .args(["--out"])
            .arg(&cases)
            .output()
            .unwrap(),
    );

    assert_success(
        traceeval()
            .args(["validate", "--cases"])
            .arg(&cases)
            .args(["--out"])
            .arg(&validation)
            .output()
            .unwrap(),
    );

    assert_success(
        traceeval()
            .args(["grade", "--cases"])
            .arg(&cases)
            .args(["--grader", "non-empty-output", "--out"])
            .arg(&results)
            .output()
            .unwrap(),
    );

    assert_success(
        traceeval()
            .args(["calibrate", "--human-ratings"])
            .arg(repo_path("fixtures/eval/human_ratings.jsonl"))
            .args(["--results"])
            .arg(repo_path("fixtures/eval/historical_results.jsonl"))
            .args(["--out"])
            .arg(&calibration)
            .output()
            .unwrap(),
    );

    assert_success(
        traceeval()
            .args(["cluster", "assign", "--cases"])
            .arg(&cases)
            .args(["--clusters"])
            .arg(repo_path("fixtures/eval/clusters.jsonl"))
            .args(["--out"])
            .arg(&assignments)
            .args(["--results"])
            .arg(&results)
            .args(["--results-out"])
            .arg(&clustered_results)
            .output()
            .unwrap(),
    );

    assert_success(
        traceeval()
            .args(["report", "--results"])
            .arg(&clustered_results)
            .args(["--calibration"])
            .arg(&calibration)
            .args(["--clusters"])
            .arg(repo_path("fixtures/eval/clusters.jsonl"))
            .args(["--out"])
            .arg(&report)
            .output()
            .unwrap(),
    );

    let report: Value = serde_json::from_str(&std::fs::read_to_string(report).unwrap()).unwrap();
    assert_eq!(report["total_cases"], 1);
    assert_eq!(report["total_results"], 1);
    assert_eq!(report["run_score"]["result_count"], 1);
    assert_eq!(report["cluster_scores"][0]["cluster_id"], "arithmetic");
}

#[test]
fn cli_cluster_assigns_from_discovered_model_and_embeddings() {
    let dir = tempdir().unwrap();
    let cases = dir.path().join("cases.jsonl");
    let embeddings = dir.path().join("embeddings.jsonl");
    let model = dir.path().join("cluster_model.json");
    let assignments = dir.path().join("assignments.jsonl");

    std::fs::write(
        &cases,
        r#"{"id":"case-new","trace_id":"trace-new","input":"Help with invoice","actual_output":"ok"}
"#,
    )
    .unwrap();
    std::fs::write(
        &embeddings,
        r#"{"schema_version":"traceeval.case_embedding.v1","case_id":"case-new","trace_id":"trace-new","provider":"test","model":"test-embedding","dimensions":2,"vector":[0.99,0.01],"projection_version":"traceeval.cluster_text.v1","text_hash":"abc"}
"#,
    )
    .unwrap();
    std::fs::write(
        &model,
        serde_json::to_string_pretty(&json!({
            "schema_version": "traceeval.cluster_model.v1",
            "model_id": "model-1",
            "created_at": "2026-01-01T00:00:00Z",
            "source": {
                "case_count": 2,
                "embedding_provider": "test",
                "embedding_model": "test-embedding",
                "embedding_dimensions": 2,
                "projection_version": "traceeval.cluster_text.v1",
                "algorithm": "manual",
                "distance_metric": "cosine",
                "random_seed": 42
            },
            "clusters": [
                {
                    "id": "billing",
                    "size": 2,
                    "centroid": [1.0, 0.0],
                    "representative_case_ids": ["case-a"],
                    "radius": null,
                    "mean_distance": null,
                    "quality": {
                        "cluster_id": "billing",
                        "size": 2,
                        "mean_distance": null,
                        "max_distance": null,
                        "silhouette_score": null,
                        "representative_case_ids": ["case-a"]
                    }
                }
            ],
            "assignments": [],
            "quality": {
                "cluster_count": 1,
                "assigned_case_count": 2,
                "mean_distance": null,
                "silhouette_score": null,
                "clusters": [
                    {
                        "cluster_id": "billing",
                        "size": 2,
                        "mean_distance": null,
                        "max_distance": null,
                        "silhouette_score": null,
                        "representative_case_ids": ["case-a"]
                    }
                ]
            }
        }))
        .unwrap(),
    )
    .unwrap();

    assert_success(
        traceeval()
            .args(["cluster", "assign", "--cases"])
            .arg(&cases)
            .args(["--model"])
            .arg(&model)
            .args(["--embeddings"])
            .arg(&embeddings)
            .args(["--out"])
            .arg(&assignments)
            .output()
            .unwrap(),
    );

    let assignment: Value = serde_json::from_str(
        std::fs::read_to_string(assignments)
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();

    assert_eq!(assignment["cluster_id"], "billing");
    assert_eq!(assignment["method"], "embedding_nearest_centroid");
    assert!(assignment["distance"].as_f64().unwrap() < 0.001);
}

#[cfg(feature = "clustering-linfa")]
#[test]
fn cli_cluster_discovers_model_from_embeddings() {
    let dir = tempdir().unwrap();
    let cases = dir.path().join("cases.jsonl");
    let embeddings = dir.path().join("embeddings.jsonl");
    let model = dir.path().join("cluster_model.json");
    let assignments = dir.path().join("assignments.jsonl");
    let clusters = dir.path().join("clusters.jsonl");

    std::fs::write(
        &cases,
        [
            r#"{"id":"case-a","trace_id":"trace-a","input":"billing invoice"}"#,
            r#"{"id":"case-b","trace_id":"trace-b","input":"billing receipt"}"#,
            r#"{"id":"case-c","trace_id":"trace-c","input":"password reset"}"#,
            r#"{"id":"case-d","trace_id":"trace-d","input":"login recovery"}"#,
        ]
        .join("\n")
            + "\n",
    )
    .unwrap();
    std::fs::write(
        &embeddings,
        [
            r#"{"schema_version":"traceeval.case_embedding.v1","case_id":"case-a","trace_id":"trace-a","provider":"test","model":"unit-vectors","dimensions":2,"vector":[1.0,0.0],"projection_version":"traceeval.cluster_text.v1","text_hash":"a"}"#,
            r#"{"schema_version":"traceeval.case_embedding.v1","case_id":"case-b","trace_id":"trace-b","provider":"test","model":"unit-vectors","dimensions":2,"vector":[0.95,0.05],"projection_version":"traceeval.cluster_text.v1","text_hash":"b"}"#,
            r#"{"schema_version":"traceeval.case_embedding.v1","case_id":"case-c","trace_id":"trace-c","provider":"test","model":"unit-vectors","dimensions":2,"vector":[0.0,1.0],"projection_version":"traceeval.cluster_text.v1","text_hash":"c"}"#,
            r#"{"schema_version":"traceeval.case_embedding.v1","case_id":"case-d","trace_id":"trace-d","provider":"test","model":"unit-vectors","dimensions":2,"vector":[0.05,0.95],"projection_version":"traceeval.cluster_text.v1","text_hash":"d"}"#,
        ]
        .join("\n")
            + "\n",
    )
    .unwrap();

    assert_success(
        traceeval()
            .args(["cluster", "discover", "--cases"])
            .arg(&cases)
            .args(["--embeddings"])
            .arg(&embeddings)
            .args([
                "--algorithm",
                "kmeans",
                "--k",
                "2",
                "--representatives",
                "1",
            ])
            .args(["--out-model"])
            .arg(&model)
            .args(["--out-assignments"])
            .arg(&assignments)
            .args(["--out-clusters"])
            .arg(&clusters)
            .output()
            .unwrap(),
    );

    let model_json: Value = serde_json::from_str(&std::fs::read_to_string(model).unwrap()).unwrap();
    assert_eq!(model_json["clusters"].as_array().unwrap().len(), 2);
    assert_eq!(model_json["assignments"].as_array().unwrap().len(), 4);
    assert_eq!(model_json["quality"]["cluster_count"], 2);

    assert_eq!(
        std::fs::read_to_string(assignments)
            .unwrap()
            .lines()
            .count(),
        4
    );
    assert_eq!(
        std::fs::read_to_string(clusters).unwrap().lines().count(),
        2
    );
}

#[test]
fn cli_validate_reports_fixture_case_errors() {
    let dir = tempdir().unwrap();
    let validation = dir.path().join("validation.json");

    let output = traceeval()
        .args(["validate", "--cases"])
        .arg(repo_path("fixtures/eval/cases.jsonl"))
        .args(["--out"])
        .arg(&validation)
        .output()
        .unwrap();

    assert!(!output.status.success());

    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(validation).unwrap()).unwrap();
    let codes = report["errors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|issue| issue["code"].as_str().unwrap())
        .collect::<Vec<_>>();

    assert!(codes.contains(&"missing_actual_output"));
}

#[test]
fn cli_validate_draft_profile_allows_missing_actual_output_as_warning() {
    let dir = tempdir().unwrap();
    let validation = dir.path().join("validation.json");

    assert_success(
        traceeval()
            .args(["validate", "--profile", "draft-cases", "--cases"])
            .arg(repo_path("fixtures/eval/cases.jsonl"))
            .args(["--out"])
            .arg(&validation)
            .output()
            .unwrap(),
    );

    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(validation).unwrap()).unwrap();

    assert_eq!(report["errors"].as_array().unwrap().len(), 0);
    assert!(
        report["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "missing_actual_output"
                && issue["severity"] == "warning")
    );
}

#[test]
fn cli_validate_cluster_model_rejects_invalid_representatives_and_assignments() {
    let dir = tempdir().unwrap();
    let no_representatives = dir.path().join("no_representatives.json");
    let unknown_assignment = dir.path().join("unknown_assignment.json");
    let validation = dir.path().join("validation.json");

    let base_model = serde_json::json!({
        "schema_version": "traceeval.cluster_model.v1",
        "model_id": "model-1",
        "created_at": "2026-01-01T00:00:00Z",
        "source": {
            "case_count": 1,
            "algorithm": "manual",
            "distance_metric": "cosine",
            "random_seed": 42
        },
        "clusters": [{
            "id": "cluster-1",
            "size": 1,
            "representative_case_ids": ["case-1"],
            "quality": {
                "cluster_id": "cluster-1",
                "size": 1,
                "representative_case_ids": ["case-1"]
            }
        }],
        "assignments": [],
        "quality": {
            "cluster_count": 1,
            "assigned_case_count": 0,
            "clusters": []
        }
    });

    let mut missing_rep_model = base_model.clone();
    missing_rep_model["clusters"][0]["representative_case_ids"] = serde_json::json!([]);
    std::fs::write(
        &no_representatives,
        serde_json::to_string_pretty(&missing_rep_model).unwrap(),
    )
    .unwrap();

    let mut unknown_assignment_model = base_model;
    unknown_assignment_model["assignments"] = serde_json::json!([{
        "case_id": "case-1",
        "trace_id": "trace-1",
        "cluster_id": "missing-cluster",
        "confidence": 1.0,
        "method": "test"
    }]);
    std::fs::write(
        &unknown_assignment,
        serde_json::to_string_pretty(&unknown_assignment_model).unwrap(),
    )
    .unwrap();

    let output = traceeval()
        .args(["validate", "--profile", "cluster-model", "--cluster-model"])
        .arg(&no_representatives)
        .args(["--out"])
        .arg(&validation)
        .output()
        .unwrap();
    assert!(!output.status.success());

    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(&validation).unwrap()).unwrap();
    assert!(
        report["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("no representative cases")
    );

    let output = traceeval()
        .args(["validate", "--profile", "cluster-model", "--cluster-model"])
        .arg(&unknown_assignment)
        .args(["--out"])
        .arg(&validation)
        .output()
        .unwrap();
    assert!(!output.status.success());

    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(&validation).unwrap()).unwrap();
    assert!(
        report["errors"][0]["message"]
            .as_str()
            .unwrap()
            .contains("references unknown cluster missing-cluster")
    );
}

#[test]
fn cli_report_includes_failed_case_detail() {
    let dir = tempdir().unwrap();
    let report = dir.path().join("report.json");

    assert_success(
        traceeval()
            .args(["report", "--results"])
            .arg(repo_path("fixtures/eval/historical_results.jsonl"))
            .args(["--clusters"])
            .arg(repo_path("fixtures/eval/clusters.jsonl"))
            .args(["--out"])
            .arg(&report)
            .output()
            .unwrap(),
    );

    let report: Value = serde_json::from_str(&std::fs::read_to_string(report).unwrap()).unwrap();

    assert_eq!(report["failed_cases"].as_array().unwrap().len(), 1);
    assert_eq!(
        report["failed_cases"][0]["case_id"],
        "case-missing-output-1"
    );
    assert_eq!(report["failed_cases"][0]["cluster_label"], "Arithmetic");
    assert_eq!(report["worst_clusters"][0]["cluster_id"], "arithmetic");
    assert_eq!(report["worst_clusters"][0]["label"], "Arithmetic");
    assert_eq!(
        report["cluster_scores"]
            .as_array()
            .unwrap()
            .iter()
            .find(|cluster| cluster["cluster_id"] == "arithmetic")
            .unwrap()["description"],
        "Arithmetic and direct calculation tasks."
    );
}
