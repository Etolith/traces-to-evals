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

fn write_jsonl_values(path: &Path, values: &[Value]) {
    let mut contents = values
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    contents.push('\n');
    std::fs::write(path, contents).unwrap();
}

#[test]
fn cli_detects_agent_findings_and_emits_unreviewed_candidates() {
    let dir = tempdir().unwrap();
    let normalized = dir.path().join("normalized.jsonl");
    let findings = dir.path().join("findings.jsonl");
    let candidates = dir.path().join("candidates.jsonl");
    let evidence_packet = dir.path().join("evidence-packet.json");
    let projections = dir.path().join("projections.jsonl");
    let projection_cases = dir.path().join("projection-cases.jsonl");
    let signature_groups = dir.path().join("signature-groups.jsonl");
    let fixed_findings = dir.path().join("fixed-findings.jsonl");
    let verification = dir.path().join("verification.json");
    let remediation_request = dir.path().join("remediation-request.json");
    let baseline_results = dir.path().join("baseline-results.jsonl");
    let candidate_results = dir.path().join("candidate-results.jsonl");
    let remediation_verification = dir.path().join("remediation-verification.json");
    let recurrence_request = dir.path().join("recurrence-request.json");
    let recurrence_comparison = dir.path().join("recurrence-comparison.json");

    assert_success(
        traceeval()
            .args(["detect", "--traces"])
            .arg(repo_path("fixtures/behavior/traces.jsonl"))
            .args(["--adapter-config"])
            .arg(repo_path("fixtures/behavior/adapter.json"))
            .args(["--normalized-out"])
            .arg(&normalized)
            .args(["--candidates-out"])
            .arg(&candidates)
            .args(["--evidence-packet-out"])
            .arg(&evidence_packet)
            .args(["--projections-out"])
            .arg(&projections)
            .args(["--projection-cases-out"])
            .arg(&projection_cases)
            .args(["--projection-metadata-key", "fixture"])
            .args(["--signature-groups-out"])
            .arg(&signature_groups)
            .args(["--out"])
            .arg(&findings)
            .output()
            .unwrap(),
    );

    let normalized_rows = std::fs::read_to_string(normalized).unwrap();
    assert_eq!(normalized_rows.lines().count(), 4);
    assert!(normalized_rows.contains("\"status\":\"timed_out\""));
    assert!(normalized_rows.contains("\"traceeval.behavior_adapter.version\":\"1\""));
    assert!(!normalized_rows.contains("request timed out after dispatch"));

    let finding_rows = std::fs::read_to_string(&findings).unwrap();
    let finding_values = finding_rows
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    let detector_ids = finding_values
        .iter()
        .map(|finding| finding["detector_id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(finding_values.len(), 4);
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
    std::fs::write(&fixed_findings, "").unwrap();
    assert_success(
        traceeval()
            .args(["verify-findings", "--case-id", "incident-case"])
            .args(["--baseline"])
            .arg(&findings)
            .args(["--candidate"])
            .arg(&fixed_findings)
            .args([
                "--target-signature",
                finding_values[0]["failure_signature"].as_str().unwrap(),
            ])
            .args(["--out"])
            .arg(&verification)
            .output()
            .unwrap(),
    );
    let verification: Value =
        serde_json::from_str(&std::fs::read_to_string(verification).unwrap()).unwrap();
    assert_eq!(verification["finding_gate_passed"], true);

    let evaluation_result = |case_id: &str, evaluator: &str, score: f32, passed: bool| {
        json!({
            "case_id": case_id,
            "trace_id": format!("trace-{case_id}"),
            "evaluator_name": evaluator,
            "raw_score": score,
            "normalized_score": score,
            "score_scale": "unit",
            "passed": passed,
            "evaluation": "fixture",
            "metadata": {"evaluator_spec_hash": "sha256:evaluator-v1"}
        })
    };
    write_jsonl_values(
        &baseline_results,
        &[
            evaluation_result("incident-case", "incident-grader", 0.0, false),
            evaluation_result("suite-case", "suite-grader", 1.0, true),
        ],
    );
    write_jsonl_values(
        &candidate_results,
        &[
            evaluation_result("incident-case", "incident-grader", 1.0, true),
            evaluation_result("suite-case", "suite-grader", 1.0, true),
        ],
    );
    std::fs::write(
        &remediation_request,
        serde_json::to_string_pretty(&json!({
            "schema_version": "traceeval.remediation_verification_request.v1",
            "case_id": "incident-case",
            "target_failure_signatures": [
                finding_values[0]["failure_signature"].as_str().unwrap()
            ],
            "incident_case_id": "incident-case",
            "suite_case_ids": ["suite-case"],
            "severe_threshold": "high",
            "policy_gate": {
                "status": "passed",
                "evidence": [{"kind": "policy_report", "identity": "sha256:policy"}]
            },
            "approval_gate": {
                "status": "passed",
                "evidence": [{"kind": "approval_record", "identity": "approval:fixture"}]
            },
            "baseline_budget": {
                "tool_call_count": 3,
                "latency_ms": 100,
                "cost_microunits": 20
            },
            "candidate_budget": {
                "tool_call_count": 3,
                "latency_ms": 100,
                "cost_microunits": 20
            },
            "policy": {
                "max_new_suite_failures": 0,
                "max_suite_score_drop": 0.0,
                "max_tool_call_increase": 0,
                "max_latency_increase_ms": 0,
                "max_cost_increase_microunits": 0
            }
        }))
        .unwrap(),
    )
    .unwrap();
    assert_success(
        traceeval()
            .args(["verify-remediation", "--request"])
            .arg(&remediation_request)
            .args(["--baseline-findings"])
            .arg(&findings)
            .args(["--candidate-findings"])
            .arg(&fixed_findings)
            .args(["--baseline-results"])
            .arg(&baseline_results)
            .args(["--candidate-results"])
            .arg(&candidate_results)
            .args(["--out"])
            .arg(&remediation_verification)
            .output()
            .unwrap(),
    );
    let remediation_report: Value =
        serde_json::from_str(&std::fs::read_to_string(&remediation_verification).unwrap()).unwrap();
    assert_eq!(remediation_report["passed"], true);
    assert_eq!(
        remediation_report["incident_regression_gate"]["paired_result_count"],
        1
    );
    assert_eq!(
        remediation_report["suite_regression_gate"]["paired_result_count"],
        1
    );
    assert_eq!(
        remediation_report["input_artifacts"]["baseline_findings"]["record_count"],
        4
    );
    assert_eq!(
        remediation_report["input_artifacts"]["candidate_findings"]["record_count"],
        0
    );
    assert!(
        remediation_report["input_artifacts"]["baseline_results"]["content_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );

    let mut rejected_request: Value =
        serde_json::from_str(&std::fs::read_to_string(&remediation_request).unwrap()).unwrap();
    rejected_request["candidate_budget"]["tool_call_count"] = json!(4);
    std::fs::write(
        &remediation_request,
        serde_json::to_string_pretty(&rejected_request).unwrap(),
    )
    .unwrap();
    let rejected_output = traceeval()
        .args(["verify-remediation", "--request"])
        .arg(&remediation_request)
        .args(["--baseline-findings"])
        .arg(&findings)
        .args(["--candidate-findings"])
        .arg(&fixed_findings)
        .args(["--baseline-results"])
        .arg(&baseline_results)
        .args(["--candidate-results"])
        .arg(&candidate_results)
        .args(["--out"])
        .arg(&remediation_verification)
        .output()
        .unwrap();
    assert!(!rejected_output.status.success());
    let rejected_report: Value =
        serde_json::from_str(&std::fs::read_to_string(&remediation_verification).unwrap()).unwrap();
    assert_eq!(rejected_report["passed"], false);
    assert_eq!(rejected_report["budget_gate"]["passed"], false);

    std::fs::write(
        &recurrence_request,
        serde_json::to_string_pretty(&json!({
            "schema_version": "traceeval.finding_recurrence_request.v1",
            "target_failure_signatures": [
                finding_values[0]["failure_signature"].as_str().unwrap()
            ],
            "baseline_window": {
                "window_id": "baseline-window",
                "observed_trace_count": 4,
                "population_basis": "exact"
            },
            "candidate_window": {
                "window_id": "candidate-window",
                "observed_trace_count": 4,
                "population_basis": "exact"
            },
            "severe_threshold": "high"
        }))
        .unwrap(),
    )
    .unwrap();
    assert_success(
        traceeval()
            .args(["compare-recurrence", "--request"])
            .arg(&recurrence_request)
            .args(["--baseline-findings"])
            .arg(&findings)
            .args(["--candidate-findings"])
            .arg(&fixed_findings)
            .args(["--out"])
            .arg(&recurrence_comparison)
            .output()
            .unwrap(),
    );
    let recurrence_comparison: Value =
        serde_json::from_str(&std::fs::read_to_string(recurrence_comparison).unwrap()).unwrap();
    assert_eq!(recurrence_comparison["evidence_complete"], true);
    assert_eq!(recurrence_comparison["candidate_recurrence_rate"], 0.0);
    assert_eq!(recurrence_comparison["comparator_version"], "2");

    let candidate_values = std::fs::read_to_string(candidates)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(candidate_values.len(), finding_values.len());
    let packet: Value =
        serde_json::from_str(&std::fs::read_to_string(evidence_packet).unwrap()).unwrap();
    assert!(
        packet["content_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );
    assert!(candidate_values.iter().all(|candidate| {
        candidate["evidence_packet_id"] == packet["packet_id"]
            && candidate["definition_hash"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
    }));
    assert!(
        candidate_values
            .iter()
            .all(|candidate| candidate.get("proposed_input").is_none())
    );
    assert!(
        candidate_values
            .iter()
            .all(|candidate| candidate["status"] == "candidate")
    );

    let projection_values = std::fs::read_to_string(projections)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(projection_values.len(), finding_values.len());
    assert!(projection_values.iter().all(|projection| {
        projection["text"].as_str().unwrap().contains("fixture:")
            && projection["text_hash"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
    }));
    let projection_case_values = std::fs::read_to_string(projection_cases)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(projection_case_values.len(), finding_values.len());
    assert!(
        projection_case_values
            .iter()
            .all(|case| { case["metadata"]["artifact_kind"] == "behavior_finding_projection" })
    );

    let group_values = std::fs::read_to_string(signature_groups)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(group_values.len(), finding_values.len());
    assert!(
        group_values
            .iter()
            .all(|group| group["occurrence_count"] == 1)
    );
}
