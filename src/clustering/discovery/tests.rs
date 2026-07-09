use serde_json::Value;

use super::*;
use crate::clustering::assignment::{ClusterAssignment, UNCLUSTERED};
use crate::clustering::quality::ClusterQualityReport;
use crate::model::EvalCase;
use crate::project::ProjectName;

fn model() -> ClusterModel {
    let clusters = vec![
        DiscoveredCluster::new("cluster-0001", 2, vec!["case-a".to_string()])
            .with_centroid(vec![1.0, 0.0]),
        DiscoveredCluster::new("cluster-0002", 2, vec!["case-b".to_string()])
            .with_centroid(vec![0.0, 1.0]),
    ];
    let quality = ClusterQualityReport {
        cluster_count: 2,
        assigned_case_count: 4,
        mean_distance: None,
        silhouette_score: None,
        clusters: clusters
            .iter()
            .map(|cluster| cluster.quality.clone())
            .collect(),
    };

    ClusterModel::new(
        "model-1",
        "2026-01-01T00:00:00Z",
        ClusterModelSource {
            case_count: 4,
            embedding_provider: Some("test".to_string()),
            embedding_model: Some("test".to_string()),
            embedding_dimensions: Some(2),
            projection_version: Some("projection".to_string()),
            algorithm: "manual".to_string(),
            distance_metric: "cosine".to_string(),
            random_seed: 42,
        },
        clusters,
        Vec::new(),
        quality,
    )
}

#[test]
fn nearest_centroid_assignment_sets_distance_confidence_and_method() {
    let assigner = ClusterModelAssigner::new(model());
    let case = EvalCase::new("case-new", "trace-new", "input");

    let assignment = assigner.assign_case_embedding(&case, &[0.9, 0.1]).unwrap();

    assert_eq!(assignment.cluster_id, "cluster-0001");
    assert_eq!(assignment.method, "embedding_nearest_centroid");
    assert!(assignment.distance.unwrap() < 0.1);
    assert!(assignment.confidence > 0.95);
    assert!(!assignment.novelty);
}

#[test]
fn nearest_centroid_assignment_marks_novelty() {
    let assigner = ClusterModelAssigner::new(model()).with_novelty_distance_threshold(0.01);
    let case = EvalCase::new("case-new", "trace-new", "input");

    let assignment = assigner.assign_case_embedding(&case, &[0.7, 0.7]).unwrap();

    assert_eq!(assignment.cluster_id, UNCLUSTERED);
    assert!(assignment.novelty);
    assert_eq!(
        assignment.metadata.get("nearest_cluster_id"),
        Some(&Value::String("cluster-0001".to_string()))
    );
}

#[test]
fn cluster_model_validation_rejects_unknown_assignment_cluster() {
    let mut model = model();
    let case = EvalCase::new("case-new", "trace-new", "input");
    model.assignments.push(ClusterAssignment::new(
        &case,
        "missing-cluster",
        1.0,
        "test",
    ));

    assert!(model.validate().is_err());
}

#[test]
fn cluster_model_supports_custom_project_schema_namespace() {
    let project = ProjectName::new("acme-evals").unwrap();
    let base_model = model();
    let model = ClusterModel::new_with_project(
        &project,
        "model-1",
        "2026-01-01T00:00:00Z",
        base_model.source,
        base_model.clusters,
        Vec::new(),
        base_model.quality,
    );

    assert_eq!(model.schema_version, "acme-evals.cluster_model.v1");
    assert!(model.validate().is_ok());
}

#[cfg(feature = "clustering-linfa")]
#[test]
fn kmeans_discovery_fits_model_with_assignments_and_quality() {
    use crate::clustering::{CaseEmbedding, ClusterTextProjector, DefaultClusterTextProjector};

    let cases = vec![
        EvalCase::new("case-a", "trace-a", "billing invoice"),
        EvalCase::new("case-b", "trace-b", "billing receipt"),
        EvalCase::new("case-c", "trace-c", "password reset"),
        EvalCase::new("case-d", "trace-d", "login recovery"),
    ];
    let projector = DefaultClusterTextProjector::new();
    let embeddings = cases
        .iter()
        .zip([
            vec![1.0, 0.0],
            vec![0.95, 0.05],
            vec![0.0, 1.0],
            vec![0.05, 0.95],
        ])
        .map(|(case, vector)| {
            let projected = projector.project_case(case);
            CaseEmbedding::new(
                &projected,
                "test",
                "unit-vectors",
                vector,
                projector.projection_version(),
            )
        })
        .collect::<Vec<_>>();
    let options = ClusterDiscoveryOptions {
        model_id: Some("test-model".to_string()),
        algorithm: ClusterAlgorithm::KMeans {
            k: 2,
            max_iterations: 100,
            tolerance: 0.0001,
        },
        representative_count: 1,
        ..ClusterDiscoveryOptions::default()
    };
    let discovery = KMeansClusterDiscovery {
        k: 2,
        max_iterations: 100,
        tolerance: 0.0001,
        random_seed: 42,
    };

    let model = discovery
        .fit(ClusterDiscoveryInput {
            cases: &cases,
            embeddings: Some(&embeddings),
            human_ratings: None,
            previous_results: None,
            options: &options,
        })
        .unwrap();

    model.validate().unwrap();
    assert_eq!(model.model_id, "test-model");
    assert_eq!(model.clusters.len(), 2);
    assert_eq!(model.assignments.len(), 4);
    assert_eq!(model.quality.cluster_count, 2);
    assert_eq!(model.quality.assigned_case_count, 4);
    assert!(model.quality.mean_distance.is_some());
    assert!(model.quality.silhouette_score.is_some());
    assert!(
        model
            .clusters
            .iter()
            .all(|cluster| cluster.centroid.is_some()
                && cluster.representative_case_ids.len() == 1
                && cluster.mean_distance.is_some()
                && cluster.radius.is_some())
    );
    assert!(
        model
            .assignments
            .iter()
            .all(|assignment| assignment.method == "kmeans" && assignment.distance.is_some())
    );
}
