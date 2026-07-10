use super::*;
use crate::clustering::{
    ClusterModel, ClusterModelSource, ClusterQualityReport, DiscoveredCluster,
    EmbeddingClusterAssigner,
};
use crate::model::EvalCase;

fn records() -> Vec<OwnedVectorRecord> {
    vec![
        OwnedVectorRecord {
            row_id: 0,
            external_id: "a".to_string(),
            vector: vec![1.0, 0.0],
        },
        OwnedVectorRecord {
            row_id: 1,
            external_id: "b".to_string(),
            vector: vec![0.0, 1.0],
        },
    ]
}

#[test]
fn brute_force_returns_nearest_row() {
    let mut index = BruteForceVectorIndex::new(VectorMetric::Cosine, records()).unwrap();
    let hits = index
        .search(
            &[0.9, 0.1],
            &VectorSearchOptions::new(1, VectorMetric::Cosine),
        )
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].row_id, 0);
    assert!(hits[0].distance < 0.01);
}

#[test]
fn brute_force_honors_allowed_row_ids() {
    let mut index = BruteForceVectorIndex::new(VectorMetric::Cosine, records()).unwrap();
    let hits = index
        .search(
            &[0.9, 0.1],
            &VectorSearchOptions::new(1, VectorMetric::Cosine).with_allowed_row_ids(vec![1]),
        )
        .unwrap();

    assert_eq!(hits[0].row_id, 1);
}

#[test]
fn row_map_rejects_unknown_row_id_lookup() {
    let row_map = VectorIndexRowMap::new(
        "index-1",
        vec![VectorIndexRow {
            row_id: 7,
            external_id: "cluster-1".to_string(),
        }],
    );

    let error = row_map.external_id_for(8).unwrap_err();
    assert!(matches!(error, TraceEvalError::VectorIndex { backend, .. } if backend == "row-map"));
}

#[test]
fn vector_index_assigner_matches_nearest_centroid() {
    let case = EvalCase::new("case-1", "trace-1", "billing help");
    let cluster =
        DiscoveredCluster::new("billing", 1, vec![case.id.clone()]).with_centroid(vec![1.0, 0.0]);
    let model = ClusterModel::new(
        "model-1",
        "2026-01-01T00:00:00Z",
        ClusterModelSource {
            case_count: 1,
            embedding_provider: None,
            embedding_model: None,
            embedding_dimensions: Some(2),
            projection_version: None,
            algorithm: "manual".to_string(),
            distance_metric: "cosine".to_string(),
            random_seed: 42,
        },
        vec![cluster],
        Vec::new(),
        ClusterQualityReport {
            cluster_count: 1,
            assigned_case_count: 1,
            mean_distance: None,
            silhouette_score: None,
            clusters: Vec::new(),
        },
    );
    let records = cluster_centroid_records(&model);
    let borrowed = borrowed_records(&records);
    let row_map = VectorIndexRowMap::from_records("index-1", &borrowed);
    let index = BruteForceVectorIndexBuilder::new(VectorMetric::Cosine)
        .build(&borrowed)
        .unwrap();
    let mut assigner = VectorIndexClusterAssigner::new(model, index, row_map).unwrap();

    let assignment = assigner
        .assign_case_embedding(&case, &[0.95, 0.05])
        .unwrap();

    assert_eq!(assignment.cluster_id, "billing");
    assert_eq!(assignment.method, "embedding_vector_index");
    assert!(assignment.distance.unwrap() < 0.01);
}

#[cfg(feature = "ann-paimon")]
#[test]
fn paimon_backend_builds_and_searches_temp_index() {
    let dir = tempfile::tempdir().unwrap();
    let records = records();
    let borrowed = borrowed_records(&records);
    let builder = PaimonVectorIndexBuilder {
        config: PaimonVectorIndexConfig {
            metric: VectorMetric::Cosine,
            dimensions: 2,
            kind: PaimonVectorIndexKind::IvfFlat { nlist: 1 },
            optimize_on_open: true,
        },
        output_path: dir.path().join("vectors.pvindex"),
    };
    let mut index = builder.build(&borrowed).unwrap();

    let hits = index
        .search(
            &[0.95, 0.05],
            &VectorSearchOptions::new(1, VectorMetric::Cosine),
        )
        .unwrap();

    assert_eq!(hits[0].row_id, 0);
}
