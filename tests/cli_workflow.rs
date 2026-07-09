use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;
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
            .args(["cluster", "--cases"])
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
