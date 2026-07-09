use std::collections::BTreeMap;

use serde_json::Value;

use super::*;
use crate::model::EvalCase;

#[test]
fn assigns_cluster_from_metadata() {
    let mut case = EvalCase::new("case-1", "trace-1", "input");
    case.metadata.insert(
        "cluster_id".to_string(),
        Value::String("arithmetic".to_string()),
    );

    let assignment = RuleBasedClusterAssigner::new(vec![cluster("arithmetic", "Arithmetic")])
        .assign_case(&case)
        .unwrap();

    assert_eq!(assignment.cluster_id, "arithmetic");
    assert_eq!(assignment.method, "metadata");
}

#[test]
fn applies_assignments_to_results() {
    let case = EvalCase::new("case-1", "trace-1", "input");
    let result = EvaluationResult::binary(&case, "non_empty", true, "ok");
    let assignment = ClusterAssignment::new(&case, "arithmetic", 1.0, "metadata");

    let results = apply_assignments_to_results(vec![result], &[assignment]);

    assert_eq!(results[0].cluster_id.as_deref(), Some("arithmetic"));
}

#[test]
fn supports_custom_assignment_rules() {
    let case = EvalCase::new("case-1", "trace-1", "refund this order");
    let assigner =
        RuleBasedClusterAssigner::empty(vec![cluster("support", "Customer Support")]).with_rule(
            FnClusterAssignmentRule::new("custom_refund_rule", |case, clusters| {
                if case.input.contains("refund")
                    && clusters.iter().any(|cluster| cluster.id == "support")
                {
                    Some(ClusterRuleMatch::new("support", 0.95))
                } else {
                    None
                }
            }),
        );

    let assignment = assigner.assign_case(&case).unwrap();

    assert_eq!(assignment.cluster_id, "support");
    assert_eq!(assignment.confidence, 0.95);
    assert_eq!(assignment.method, "custom_refund_rule");
}

fn cluster(id: &str, label: &str) -> EvalCluster {
    EvalCluster {
        id: id.to_string(),
        label: label.to_string(),
        description: None,
        weight: 1.0,
        metadata: BTreeMap::new(),
    }
}
