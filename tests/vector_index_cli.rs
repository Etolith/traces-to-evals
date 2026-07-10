use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use tempfile::tempdir;

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

fn write_fixture_files(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let cases = dir.join("cases.jsonl");
    let embeddings = dir.join("embeddings.jsonl");
    let model = dir.join("cluster_model.json");

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
                    "quality": {
                        "cluster_id": "billing",
                        "size": 2,
                        "representative_case_ids": ["case-a"]
                    }
                }
            ],
            "assignments": [],
            "quality": {
                "cluster_count": 1,
                "assigned_case_count": 2,
                "clusters": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    (cases, embeddings, model)
}

#[test]
fn cli_builds_bruteforce_vector_index_and_assigns_with_it() {
    let dir = tempdir().unwrap();
    let (cases, embeddings, model) = write_fixture_files(dir.path());
    let index = dir.path().join("centroids.index.json");
    let row_map = dir.path().join("centroids.rows.json");
    let assignments = dir.path().join("assignments.jsonl");

    assert_success(
        traceeval()
            .args(["cluster", "index", "--model"])
            .arg(&model)
            .args(["--backend", "brute-force", "--metric", "cosine"])
            .args(["--out-index"])
            .arg(&index)
            .args(["--out-row-map"])
            .arg(&row_map)
            .output()
            .unwrap(),
    );

    assert_success(
        traceeval()
            .args(["cluster", "assign", "--cases"])
            .arg(&cases)
            .args(["--model"])
            .arg(&model)
            .args(["--embeddings"])
            .arg(&embeddings)
            .args(["--vector-index", "brute-force", "--index-file"])
            .arg(&index)
            .args(["--index-row-map"])
            .arg(&row_map)
            .args(["--out"])
            .arg(&assignments)
            .output()
            .unwrap(),
    );

    let row_map_json: Value =
        serde_json::from_str(&std::fs::read_to_string(row_map).unwrap()).unwrap();
    assert_eq!(
        row_map_json["schema_version"],
        "traceeval.vector_index_row_map.v1"
    );
    assert_eq!(row_map_json["rows"][0]["external_id"], "billing");

    let assignment: Value = serde_json::from_str(
        std::fs::read_to_string(assignments)
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(assignment["cluster_id"], "billing");
    assert_eq!(assignment["method"], "embedding_vector_index");
    assert!(assignment["distance"].as_f64().unwrap() < 0.001);
}

#[cfg(feature = "ann-paimon")]
#[test]
fn cli_builds_paimon_vector_index_and_assigns_with_it() {
    let dir = tempdir().unwrap();
    let (cases, embeddings, model) = write_fixture_files(dir.path());
    let index = dir.path().join("centroids.pvindex");
    let row_map = dir.path().join("centroids.rows.json");
    let assignments = dir.path().join("assignments.jsonl");

    assert_success(
        traceeval()
            .args(["cluster", "index", "--model"])
            .arg(&model)
            .args([
                "--backend",
                "paimon",
                "--index-kind",
                "ivf-flat",
                "--metric",
                "cosine",
                "--nlist",
                "1",
            ])
            .args(["--out-index"])
            .arg(&index)
            .args(["--out-row-map"])
            .arg(&row_map)
            .output()
            .unwrap(),
    );

    assert_success(
        traceeval()
            .args(["cluster", "assign", "--cases"])
            .arg(&cases)
            .args(["--model"])
            .arg(&model)
            .args(["--embeddings"])
            .arg(&embeddings)
            .args(["--vector-index", "paimon", "--index-file"])
            .arg(&index)
            .args(["--index-row-map"])
            .arg(&row_map)
            .args(["--nprobe", "1", "--out"])
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
    assert_eq!(assignment["method"], "embedding_vector_index");
}
