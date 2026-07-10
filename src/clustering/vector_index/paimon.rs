use std::fs::File;
use std::path::{Path, PathBuf};

use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::hnsw::HnswBuildParams;
use paimon_vindex_core::index::{
    VectorIndexConfig, VectorIndexReader, VectorIndexTrainer, VectorIndexWriter, VectorSearchParams,
};
use paimon_vindex_core::io::PosWriter;
use roaring::RoaringTreemap;

use crate::clustering::vector_index::{
    VectorIndex, VectorIndexBuilder, VectorMetric, VectorRecord, VectorSearchHit,
    VectorSearchOptions, validate_search_options, vector_index_error,
};
use crate::{Result, TraceEvalError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaimonVectorIndexBuilder {
    pub config: PaimonVectorIndexConfig,
    pub output_path: PathBuf,
}

impl VectorIndexBuilder for PaimonVectorIndexBuilder {
    type Index = PaimonVectorIndex;

    fn build(&self, records: &[VectorRecord<'_>]) -> Result<Self::Index> {
        let (row_ids, vectors, vector_count) =
            flatten_records(records, self.config.dimensions, "paimon")?;
        let config = self.config.to_paimon_config();
        let training = VectorIndexTrainer::train(config, &vectors, vector_count)
            .map_err(|error| paimon_error(error.to_string()))?;
        let mut writer = VectorIndexWriter::new(training);
        writer
            .add_vectors(&row_ids, &vectors, vector_count)
            .map_err(|error| paimon_error(error.to_string()))?;

        let mut file = File::create(&self.output_path)?;
        let mut out = PosWriter::new(&mut file);
        writer
            .write(&mut out)
            .map_err(|error| paimon_error(error.to_string()))?;

        let mut index = PaimonVectorIndex::open(&self.output_path)?;
        if self.config.optimize_on_open {
            index.optimize_for_search()?;
        }
        Ok(index)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaimonVectorIndexConfig {
    pub metric: VectorMetric,
    pub dimensions: usize,
    pub kind: PaimonVectorIndexKind,
    pub optimize_on_open: bool,
}

impl PaimonVectorIndexConfig {
    fn to_paimon_config(&self) -> VectorIndexConfig {
        let metric = paimon_metric(self.metric);
        match &self.kind {
            PaimonVectorIndexKind::IvfFlat { nlist } => VectorIndexConfig::IvfFlat {
                dimension: self.dimensions,
                nlist: *nlist,
                metric,
            },
            PaimonVectorIndexKind::IvfPq { nlist, m, use_opq } => VectorIndexConfig::IvfPq {
                dimension: self.dimensions,
                nlist: *nlist,
                m: *m,
                metric,
                use_opq: *use_opq,
            },
            PaimonVectorIndexKind::IvfHnswFlat { nlist, hnsw } => VectorIndexConfig::IvfHnswFlat {
                dimension: self.dimensions,
                nlist: *nlist,
                metric,
                hnsw: hnsw.to_paimon(),
            },
            PaimonVectorIndexKind::IvfHnswSq { nlist, hnsw } => VectorIndexConfig::IvfHnswSq {
                dimension: self.dimensions,
                nlist: *nlist,
                metric,
                hnsw: hnsw.to_paimon(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaimonVectorIndexKind {
    IvfFlat {
        nlist: usize,
    },
    IvfPq {
        nlist: usize,
        m: usize,
        use_opq: bool,
    },
    IvfHnswFlat {
        nlist: usize,
        hnsw: PaimonHnswOptions,
    },
    IvfHnswSq {
        nlist: usize,
        hnsw: PaimonHnswOptions,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaimonHnswOptions {
    pub m: usize,
    pub ef_construction: usize,
    pub max_level: usize,
}

impl Default for PaimonHnswOptions {
    fn default() -> Self {
        Self {
            m: 20,
            ef_construction: 150,
            max_level: 7,
        }
    }
}

impl PaimonHnswOptions {
    fn to_paimon(self) -> HnswBuildParams {
        HnswBuildParams {
            m: self.m,
            ef_construction: self.ef_construction,
            max_level: self.max_level,
        }
    }
}

pub struct PaimonVectorIndex {
    reader: VectorIndexReader<File>,
    metric: VectorMetric,
    dimensions: usize,
}

impl PaimonVectorIndex {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader =
            VectorIndexReader::open(file).map_err(|error| paimon_error(error.to_string()))?;
        let metadata = reader.metadata();
        Ok(Self {
            reader,
            metric: vector_metric(metadata.metric)?,
            dimensions: metadata.dimension,
        })
    }
}

impl VectorIndex for PaimonVectorIndex {
    fn metric(&self) -> VectorMetric {
        self.metric
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn optimize_for_search(&mut self) -> Result<()> {
        self.reader
            .optimize_for_search()
            .map_err(|error| paimon_error(error.to_string()))
    }

    fn search(
        &mut self,
        query: &[f32],
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchHit>> {
        validate_search_options("paimon", self.metric, self.dimensions, query, options)?;
        let params = paimon_search_params(options);
        let (ids, distances) = if let Some(row_ids) = &options.allowed_row_ids {
            let filter = roaring_filter(row_ids)?;
            self.reader
                .search_with_roaring_filter(query, params, &filter)
                .map_err(|error| paimon_error(error.to_string()))?
        } else {
            self.reader
                .search(query, params)
                .map_err(|error| paimon_error(error.to_string()))?
        };
        hits_from_paimon(ids, distances)
    }

    fn search_batch(
        &mut self,
        queries: &[Vec<f32>],
        options: &VectorSearchOptions,
    ) -> Result<Vec<Vec<VectorSearchHit>>> {
        for query in queries {
            validate_search_options("paimon", self.metric, self.dimensions, query, options)?;
        }
        let query_count = queries.len();
        if query_count == 0 {
            return Ok(Vec::new());
        }

        let flattened = queries
            .iter()
            .flat_map(|query| query.iter().copied())
            .collect::<Vec<_>>();
        let params = paimon_search_params(options);
        let (ids, distances) = if let Some(row_ids) = &options.allowed_row_ids {
            let filter = roaring_filter(row_ids)?;
            self.reader
                .search_batch_with_roaring_filter(&flattened, query_count, params, &filter)
                .map_err(|error| paimon_error(error.to_string()))?
        } else {
            self.reader
                .search_batch(&flattened, query_count, params)
                .map_err(|error| paimon_error(error.to_string()))?
        };

        let mut batches = Vec::with_capacity(query_count);
        for (id_chunk, distance_chunk) in ids
            .chunks(options.top_k)
            .zip(distances.chunks(options.top_k))
        {
            batches.push(hits_from_paimon(
                id_chunk.to_vec(),
                distance_chunk.to_vec(),
            )?);
        }
        Ok(batches)
    }
}

fn paimon_metric(metric: VectorMetric) -> MetricType {
    match metric {
        VectorMetric::Cosine => MetricType::Cosine,
        VectorMetric::Euclidean => MetricType::L2,
        VectorMetric::InnerProduct => MetricType::InnerProduct,
    }
}

fn vector_metric(metric: MetricType) -> Result<VectorMetric> {
    match metric {
        MetricType::Cosine => Ok(VectorMetric::Cosine),
        MetricType::L2 => Ok(VectorMetric::Euclidean),
        MetricType::InnerProduct => Ok(VectorMetric::InnerProduct),
    }
}

fn paimon_search_params(options: &VectorSearchOptions) -> VectorSearchParams {
    match options.ef_search {
        Some(ef_search) => VectorSearchParams::with_ef_search(
            options.top_k,
            options.nprobe.unwrap_or(1),
            ef_search,
        ),
        None => VectorSearchParams::new(options.top_k, options.nprobe.unwrap_or(1)),
    }
}

fn roaring_filter(row_ids: &[u64]) -> Result<Vec<u8>> {
    let mut allowed = RoaringTreemap::new();
    for row_id in row_ids {
        allowed.insert(*row_id);
    }
    let mut bytes = Vec::new();
    allowed
        .serialize_into(&mut bytes)
        .map_err(|error| paimon_error(error.to_string()))?;
    Ok(bytes)
}

fn flatten_records(
    records: &[VectorRecord<'_>],
    dimensions: usize,
    backend: &str,
) -> Result<(Vec<i64>, Vec<f32>, usize)> {
    if records.is_empty() {
        return Err(vector_index_error(backend, "index has no records"));
    }
    if dimensions == 0 {
        return Err(vector_index_error(
            backend,
            "vectors must have at least one dimension",
        ));
    }

    let mut row_ids = Vec::with_capacity(records.len());
    let mut vectors = Vec::with_capacity(records.len() * dimensions);
    for record in records {
        if record.vector.len() != dimensions {
            return Err(vector_index_error(
                backend,
                format!(
                    "record {} dimensions {} do not match expected {}",
                    record.external_id,
                    record.vector.len(),
                    dimensions
                ),
            ));
        }
        if record.vector.iter().any(|value| !value.is_finite()) {
            return Err(vector_index_error(
                backend,
                format!("record {} contains non-finite value", record.external_id),
            ));
        }
        let row_id = i64::try_from(record.row_id).map_err(|_| {
            vector_index_error(
                backend,
                format!("row_id {} exceeds Paimon i64 range", record.row_id),
            )
        })?;
        row_ids.push(row_id);
        vectors.extend_from_slice(record.vector);
    }

    Ok((row_ids, vectors, records.len()))
}

fn hits_from_paimon(ids: Vec<i64>, distances: Vec<f32>) -> Result<Vec<VectorSearchHit>> {
    ids.into_iter()
        .zip(distances)
        .filter(|(row_id, _)| *row_id >= 0)
        .map(|(row_id, distance)| {
            Ok(VectorSearchHit {
                row_id: u64::try_from(row_id).map_err(|_| {
                    vector_index_error("paimon", format!("negative row_id {}", row_id))
                })?,
                distance,
            })
        })
        .collect()
}

fn paimon_error(message: impl Into<String>) -> TraceEvalError {
    TraceEvalError::VectorIndex {
        backend: "paimon".to_string(),
        message: message.into(),
    }
}
