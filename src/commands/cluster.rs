use std::path::PathBuf;

use anyhow::{Result, anyhow};

use crate::cli::{
    ClusterAlgorithmName, ClusterArgs, ClusterAssignArgs, ClusterCommand, ClusterDiscoverArgs,
    ClusterEmbedArgs, ClusterEmbeddingProviderName, ClusterLabelArgs, ClusterLabelProviderName,
};
use crate::clustering::{
    CaseEmbedding, ClusterAssigner, ClusterAssignment, ClusterModel, ClusterModelAssigner,
    EmbeddingClusterAssigner, EvalCluster, RuleBasedClusterAssigner, apply_assignments_to_results,
};
use crate::evaluation::EvaluationResult;
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlFile;
use crate::model::EvalCase;

pub fn run(args: ClusterArgs) -> Result<()> {
    match args.command {
        ClusterCommand::Assign(args) => assign(args),
        ClusterCommand::Embed(args) => embed(args),
        ClusterCommand::Discover(args) => discover(args),
        ClusterCommand::Label(args) => label(args),
    }
}

fn assign(args: ClusterAssignArgs) -> Result<()> {
    let cases: Vec<EvalCase> = JsonlFile::new(&args.cases).read_all()?;
    let assignments = match (args.clusters.as_ref(), args.model.as_ref()) {
        (Some(clusters_path), None) => {
            let clusters: Vec<EvalCluster> = JsonlFile::new(clusters_path).read_all()?;
            let assigner = RuleBasedClusterAssigner::new(clusters);
            assigner.assign_cases(&cases)?
        }
        (None, Some(model_path)) => {
            let model: ClusterModel = JsonFile::new(model_path).read()?;
            let embeddings_path = args
                .embeddings
                .as_ref()
                .ok_or_else(|| anyhow!("--model assignment requires --embeddings"))?;
            let embeddings: Vec<CaseEmbedding> = JsonlFile::new(embeddings_path).read_all()?;
            let mut assigner = ClusterModelAssigner::new(model);
            if let Some(threshold) = args.novelty_distance_threshold {
                assigner = assigner.with_novelty_distance_threshold(threshold);
            }
            assigner.assign_case_embeddings(&cases, &embeddings)?
        }
        _ => {
            return Err(anyhow!(
                "cluster assign requires exactly one of --clusters or --model"
            ));
        }
    };

    JsonlFile::new(&args.out).write_all(&assignments)?;
    annotate_results(args.results, args.results_out, &assignments)
}

fn annotate_results(
    results: Option<PathBuf>,
    results_out: Option<PathBuf>,
    assignments: &[ClusterAssignment],
) -> Result<()> {
    match (results, results_out) {
        (Some(results_path), Some(out_path)) => {
            let results: Vec<EvaluationResult> = JsonlFile::new(results_path).read_all()?;
            let results = apply_assignments_to_results(results, assignments);
            JsonlFile::new(out_path).write_all(&results)?;
            Ok(())
        }
        (None, Some(_)) => Err(anyhow!("--results-out requires --results")),
        _ => Ok(()),
    }
}

fn embed(args: ClusterEmbedArgs) -> Result<()> {
    match args.provider {
        ClusterEmbeddingProviderName::Openai => Err(anyhow!(
            "cluster embed --provider openai is specified but not implemented yet; planned feature: embeddings-openai"
        )),
    }
}

fn discover(args: ClusterDiscoverArgs) -> Result<()> {
    match args.algorithm {
        ClusterAlgorithmName::Kmeans if args.k.is_none() => {
            Err(anyhow!("cluster discover --algorithm kmeans requires --k"))
        }
        ClusterAlgorithmName::Kmeans => Err(anyhow!(
            "cluster discover --algorithm kmeans is specified but not implemented yet; planned feature: clustering-linfa"
        )),
        ClusterAlgorithmName::Dbscan => Err(anyhow!(
            "cluster discover --algorithm dbscan is specified but not implemented yet; planned feature: clustering-linfa"
        )),
    }
}

fn label(args: ClusterLabelArgs) -> Result<()> {
    match args.provider {
        ClusterLabelProviderName::Openai => Err(anyhow!(
            "cluster label --provider openai is specified but not implemented yet; planned feature: cluster-label-openai"
        )),
    }
}
