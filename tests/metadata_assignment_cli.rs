use std::process::Command;

use serde_json::Value;
use tempfile::tempdir;

#[test]
fn cli_requires_explicit_application_metadata_key() {
    let dir = tempdir().unwrap();
    let cases = dir.path().join("cases.jsonl");
    let clusters = dir.path().join("clusters.jsonl");
    let default_assignments = dir.path().join("assignments.default.jsonl");
    let configured_assignments = dir.path().join("assignments.configured.jsonl");

    std::fs::write(
        &cases,
        r#"{"id":"case-1","trace_id":"trace-1","input":"unrelated input","actual_output":"ok","metadata":{"task_type":"card_delivery"}}
"#,
    )
    .unwrap();
    std::fs::write(
        &clusters,
        r#"{"id":"card_delivery","label":"Card delivery","weight":1.0}
"#,
    )
    .unwrap();

    let default_output = Command::new(env!("CARGO_BIN_EXE_traceeval"))
        .args(["cluster", "assign", "--cases"])
        .arg(&cases)
        .args(["--clusters"])
        .arg(&clusters)
        .args(["--out"])
        .arg(&default_assignments)
        .output()
        .unwrap();
    assert!(default_output.status.success());

    let configured_output = Command::new(env!("CARGO_BIN_EXE_traceeval"))
        .args(["cluster", "assign", "--cases"])
        .arg(&cases)
        .args(["--clusters"])
        .arg(&clusters)
        .args(["--metadata-key", "task_type", "--out"])
        .arg(&configured_assignments)
        .output()
        .unwrap();
    assert!(configured_output.status.success());

    let default_assignment = read_first_json(&default_assignments);
    assert_eq!(default_assignment["cluster_id"], "unclustered");
    assert_eq!(default_assignment["method"], "fallback");

    let configured_assignment = read_first_json(&configured_assignments);
    assert_eq!(configured_assignment["cluster_id"], "card_delivery");
    assert_eq!(configured_assignment["method"], "metadata");
}

fn read_first_json(path: &std::path::Path) -> Value {
    let content = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(content.lines().next().unwrap()).unwrap()
}
