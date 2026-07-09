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
    assert_eq!(report["worst_clusters"][0]["cluster_id"], "arithmetic");
}
