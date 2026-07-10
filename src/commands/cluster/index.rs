use std::path::Path;

use anyhow::{Result, anyhow};

#[cfg(feature = "ann-paimon")]
use crate::cli::PaimonIndexKindName;
use crate::cli::{ClusterAssignArgs, ClusterIndexArgs, VectorIndexBackendName, VectorMetricName};
use crate::clustering::{
    BruteForceVectorIndex, BruteForceVectorIndexBuilder, CaseEmbedding, ClusterAssignment,
    ClusterModel, DistanceMetric, EmbeddingClusterAssigner, VectorIndex, VectorIndexBuilder,
    VectorIndexClusterAssigner, VectorIndexRowMap, VectorMetric, VectorRecord, VectorSearchOptions,
    borrowed_records, cluster_centroid_records,
};
#[cfg(feature = "ann-paimon")]
use crate::clustering::{
    PaimonHnswOptions, PaimonVectorIndex, PaimonVectorIndexBuilder, PaimonVectorIndexConfig,
    PaimonVectorIndexKind,
};
use crate::io::json::JsonFile;
use crate::model::EvalCase;

use super::project_name_arg;

pub(super) fn build_index(args: ClusterIndexArgs) -> Result<()> {
    let project_name = project_name_arg(args.project_name.clone())?;
    let model: ClusterModel = JsonFile::new(&args.model).read()?;
    let records = cluster_centroid_records(&model);
    let borrowed = borrowed_records(&records);
    let row_map =
        VectorIndexRowMap::from_records_with_project(&project_name, &model.model_id, &borrowed);

    match args.backend {
        VectorIndexBackendName::BruteForce => {
            let index =
                BruteForceVectorIndexBuilder::new(vector_metric(args.metric)).build(&borrowed)?;
            JsonFile::new(&args.out_index).write_pretty(&index)?;
        }
        VectorIndexBackendName::Paimon => build_paimon_index(&args, &borrowed)?,
    }

    JsonFile::new(&args.out_row_map).write_pretty(&row_map)?;
    Ok(())
}

pub(super) fn assign_with_vector_index(
    args: &ClusterAssignArgs,
    backend: VectorIndexBackendName,
    model: ClusterModel,
    cases: &[EvalCase],
    embeddings: &[CaseEmbedding],
) -> Result<Vec<ClusterAssignment>> {
    let row_map_path = args
        .index_row_map
        .as_ref()
        .ok_or_else(|| anyhow!("--vector-index requires --index-row-map"))?;
    let index_path = args
        .index_file
        .as_ref()
        .ok_or_else(|| anyhow!("--vector-index requires --index-file"))?;
    let row_map: VectorIndexRowMap = JsonFile::new(row_map_path).read()?;

    match backend {
        VectorIndexBackendName::BruteForce => {
            let index: BruteForceVectorIndex = JsonFile::new(index_path).read()?;
            assign_with_loaded_index(args, model, index, row_map, cases, embeddings)
        }
        VectorIndexBackendName::Paimon => {
            let index = open_paimon_index(index_path)?;
            assign_with_loaded_index(args, model, index, row_map, cases, embeddings)
        }
    }
}

fn assign_with_loaded_index<I>(
    args: &ClusterAssignArgs,
    model: ClusterModel,
    mut index: I,
    row_map: VectorIndexRowMap,
    cases: &[EvalCase],
    embeddings: &[CaseEmbedding],
) -> Result<Vec<ClusterAssignment>>
where
    I: VectorIndex,
{
    index.optimize_for_search()?;
    let distance_metric = DistanceMetric::try_from(index.metric())?;
    let mut search_options = VectorSearchOptions::new(1, index.metric());
    if let Some(nprobe) = args.nprobe {
        search_options = search_options.with_nprobe(nprobe);
    }
    if let Some(ef_search) = args.ef_search {
        search_options = search_options.with_ef_search(ef_search);
    }

    let mut assigner = VectorIndexClusterAssigner::new_with_distance_metric(
        model,
        index,
        row_map,
        distance_metric,
    )?
    .with_search_options(search_options)?;
    if let Some(threshold) = args.novelty_distance_threshold {
        assigner = assigner.with_novelty_distance_threshold(threshold);
    }
    Ok(assigner.assign_case_embeddings(cases, embeddings)?)
}

fn vector_metric(metric: VectorMetricName) -> VectorMetric {
    match metric {
        VectorMetricName::Cosine => VectorMetric::Cosine,
        VectorMetricName::Euclidean => VectorMetric::Euclidean,
        VectorMetricName::InnerProduct => VectorMetric::InnerProduct,
    }
}

#[cfg(feature = "ann-paimon")]
fn build_paimon_index(args: &ClusterIndexArgs, records: &[VectorRecord<'_>]) -> Result<()> {
    let config = PaimonVectorIndexConfig {
        metric: vector_metric(args.metric),
        dimensions: records
            .first()
            .map(|record| record.vector.len())
            .ok_or_else(|| anyhow!("cannot build vector index from model with no centroids"))?,
        kind: paimon_index_kind(args)?,
        optimize_on_open: true,
    };
    let builder = PaimonVectorIndexBuilder {
        config,
        output_path: args.out_index.clone(),
    };
    builder.build(records)?;
    Ok(())
}

#[cfg(not(feature = "ann-paimon"))]
fn build_paimon_index(_args: &ClusterIndexArgs, _records: &[VectorRecord<'_>]) -> Result<()> {
    Err(anyhow!(
        "cluster index --backend paimon requires rebuilding with --features ann-paimon"
    ))
}

#[cfg(feature = "ann-paimon")]
fn open_paimon_index(path: impl AsRef<Path>) -> Result<PaimonVectorIndex> {
    Ok(PaimonVectorIndex::open(path)?)
}

#[cfg(not(feature = "ann-paimon"))]
fn open_paimon_index(_path: impl AsRef<Path>) -> Result<BruteForceVectorIndex> {
    Err(anyhow!(
        "cluster assign --vector-index paimon requires rebuilding with --features ann-paimon"
    ))
}

#[cfg(feature = "ann-paimon")]
fn paimon_index_kind(args: &ClusterIndexArgs) -> Result<PaimonVectorIndexKind> {
    let hnsw = || PaimonHnswOptions {
        m: args.hnsw_m.unwrap_or(20),
        ef_construction: args.hnsw_ef_construction.unwrap_or(150),
        max_level: args.hnsw_max_level.unwrap_or(7),
    };

    Ok(match args.index_kind {
        PaimonIndexKindName::IvfFlat => PaimonVectorIndexKind::IvfFlat { nlist: args.nlist },
        PaimonIndexKindName::IvfPq => PaimonVectorIndexKind::IvfPq {
            nlist: args.nlist,
            m: args
                .pq_m
                .ok_or_else(|| anyhow!("--index-kind ivf-pq requires --pq-m"))?,
            use_opq: args.use_opq,
        },
        PaimonIndexKindName::IvfHnswFlat => PaimonVectorIndexKind::IvfHnswFlat {
            nlist: args.nlist,
            hnsw: hnsw(),
        },
        PaimonIndexKindName::IvfHnswSq => PaimonVectorIndexKind::IvfHnswSq {
            nlist: args.nlist,
            hnsw: hnsw(),
        },
    })
}
