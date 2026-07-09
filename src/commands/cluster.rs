#[cfg(any(feature = "embeddings-openai", feature = "cluster-label-openai"))]
use std::path::Path;
use std::path::PathBuf;

use anyhow::{Result, anyhow};

use crate::cli::{
    ClusterAlgorithmName, ClusterArgs, ClusterAssignArgs, ClusterCommand, ClusterDiscoverArgs,
    ClusterEmbedArgs, ClusterEmbeddingProviderName, ClusterLabelArgs, ClusterLabelProviderName,
};
use crate::clustering::{
    CaseEmbedding, ClusterAlgorithm, ClusterAssigner, ClusterAssignment, ClusterDiscovery,
    ClusterDiscoveryInput, ClusterDiscoveryOptions, ClusterModel, ClusterModelAssigner,
    DistanceMetric, EmbeddingClusterAssigner, EvalCluster, KMeansClusterDiscovery,
    RuleBasedClusterAssigner, apply_assignments_to_results,
};
#[cfg(feature = "cluster-label-openai")]
use crate::clustering::{ClusterLabeler, OpenAiClusterLabeler};
#[cfg(feature = "embeddings-openai")]
use crate::clustering::{DefaultClusterTextProjector, EmbeddingProvider, OpenAiEmbeddingProvider};
use crate::evaluation::EvaluationResult;
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlFile;
use crate::model::EvalCase;
use crate::project::ProjectName;

#[cfg(any(feature = "embeddings-openai", feature = "cluster-label-openai"))]
pub async fn run(args: ClusterArgs) -> Result<()> {
    match args.command {
        ClusterCommand::Assign(args) => assign(args),
        #[cfg(feature = "embeddings-openai")]
        ClusterCommand::Embed(args) => embed(args).await,
        #[cfg(not(feature = "embeddings-openai"))]
        ClusterCommand::Embed(args) => embed(args),
        ClusterCommand::Discover(args) => discover(args),
        #[cfg(feature = "cluster-label-openai")]
        ClusterCommand::Label(args) => label(args).await,
        #[cfg(not(feature = "cluster-label-openai"))]
        ClusterCommand::Label(args) => label(args),
    }
}

#[cfg(not(any(feature = "embeddings-openai", feature = "cluster-label-openai")))]
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

fn project_name_arg(project_name: Option<String>) -> Result<ProjectName> {
    Ok(project_name.map_or_else(|| Ok(ProjectName::default()), ProjectName::new)?)
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

#[cfg(feature = "embeddings-openai")]
async fn embed(args: ClusterEmbedArgs) -> Result<()> {
    let project_name = project_name_arg(args.project_name)?;
    let cases: Vec<EvalCase> = JsonlFile::new(&args.cases).read_all()?;
    let projector = DefaultClusterTextProjector::new().with_project_name(project_name.clone());

    match args.provider {
        ClusterEmbeddingProviderName::Openai => {
            let mut provider = OpenAiEmbeddingProvider::from_env(args.model);
            if let Some(dimensions) = args.dimensions {
                provider = provider.with_dimensions(dimensions);
            }
            write_embeddings(&cases, &args.out, &project_name, &projector, &provider).await?;
            Ok(())
        }
    }
}

#[cfg(feature = "embeddings-openai")]
async fn write_embeddings<P>(
    cases: &[EvalCase],
    out: impl AsRef<Path>,
    project_name: &ProjectName,
    projector: &DefaultClusterTextProjector,
    provider: &P,
) -> Result<()>
where
    P: EmbeddingProvider,
{
    let embeddings = provider
        .embed_cases_with_project(project_name, projector, cases)
        .await?;
    JsonlFile::new(out).write_all(&embeddings)?;
    Ok(())
}

#[cfg(not(feature = "embeddings-openai"))]
fn embed(args: ClusterEmbedArgs) -> Result<()> {
    match args.provider {
        ClusterEmbeddingProviderName::Openai => Err(anyhow!(
            "cluster embed --provider openai requires rebuilding with --features embeddings-openai"
        )),
    }
}

fn discover(args: ClusterDiscoverArgs) -> Result<()> {
    match args.algorithm {
        ClusterAlgorithmName::Kmeans if args.k.is_none() => {
            Err(anyhow!("cluster discover --algorithm kmeans requires --k"))
        }
        ClusterAlgorithmName::Kmeans => discover_kmeans(args),
        ClusterAlgorithmName::Dbscan => Err(anyhow!(
            "cluster discover --algorithm dbscan is specified but not implemented yet; planned feature: clustering-linfa"
        )),
    }
}

fn discover_kmeans(args: ClusterDiscoverArgs) -> Result<()> {
    let cases: Vec<EvalCase> = JsonlFile::new(&args.cases).read_all()?;
    let embeddings: Vec<CaseEmbedding> = JsonlFile::new(&args.embeddings).read_all()?;
    let k = args
        .k
        .ok_or_else(|| anyhow!("cluster discover --algorithm kmeans requires --k"))?;
    let options = ClusterDiscoveryOptions {
        project_name: project_name_arg(args.project_name)?,
        algorithm: ClusterAlgorithm::KMeans {
            k,
            max_iterations: 100,
            tolerance: 0.0001,
        },
        distance_metric: DistanceMetric::Cosine,
        representative_count: args.representatives,
        random_seed: 42,
        ..ClusterDiscoveryOptions::default()
    };
    let discovery = KMeansClusterDiscovery {
        k,
        max_iterations: 100,
        tolerance: 0.0001,
        random_seed: options.random_seed,
    };
    let model = discovery.fit(ClusterDiscoveryInput {
        cases: &cases,
        embeddings: Some(&embeddings),
        human_ratings: None,
        previous_results: None,
        options: &options,
    })?;
    let assignments = model.assignments.clone();
    let clusters = model.to_eval_clusters();

    JsonFile::new(args.out_model).write_pretty(&model)?;
    JsonlFile::new(args.out_assignments).write_all(&assignments)?;
    JsonlFile::new(args.out_clusters).write_all(&clusters)?;
    Ok(())
}

#[cfg(feature = "cluster-label-openai")]
async fn label(args: ClusterLabelArgs) -> Result<()> {
    match args.provider {
        ClusterLabelProviderName::Openai => {
            let model: ClusterModel = JsonFile::new(&args.model).read()?;
            let cases: Vec<EvalCase> = JsonlFile::new(&args.cases).read_all()?;
            let labeler = OpenAiClusterLabeler::from_env(args.llm_model);
            write_labeled_model(model, &cases, &args.out_model, &args.out_clusters, &labeler).await
        }
    }
}

#[cfg(feature = "cluster-label-openai")]
async fn write_labeled_model<L>(
    model: ClusterModel,
    cases: &[EvalCase],
    out_model: impl AsRef<Path>,
    out_clusters: impl AsRef<Path>,
    labeler: &L,
) -> Result<()>
where
    L: ClusterLabeler,
{
    let labeled_model = labeler.label_model(model, cases).await?;
    let clusters = labeled_model.to_eval_clusters();

    JsonFile::new(out_model).write_pretty(&labeled_model)?;
    JsonlFile::new(out_clusters).write_all(&clusters)?;
    Ok(())
}

#[cfg(not(feature = "cluster-label-openai"))]
fn label(args: ClusterLabelArgs) -> Result<()> {
    match args.provider {
        ClusterLabelProviderName::Openai => Err(anyhow!(
            "cluster label --provider openai requires rebuilding with --features cluster-label-openai"
        )),
    }
}

#[cfg(all(test, feature = "embeddings-openai"))]
mod tests {
    use tempfile::tempdir;

    use super::*;

    struct StaticEmbeddingProvider;

    #[async_trait::async_trait]
    impl EmbeddingProvider for StaticEmbeddingProvider {
        fn provider_name(&self) -> String {
            "test".to_string()
        }

        fn model_name(&self) -> String {
            "static".to_string()
        }

        async fn embed_texts(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .enumerate()
                .map(|(index, _)| vec![index as f32, 1.0])
                .collect())
        }
    }

    #[tokio::test]
    async fn writes_embedding_jsonl_with_project_namespace() {
        let dir = tempdir().unwrap();
        let out = dir.path().join("embeddings.jsonl");
        let cases = vec![
            EvalCase::new("case-1", "trace-1", "input one"),
            EvalCase::new("case-2", "trace-2", "input two"),
        ];
        let project_name = ProjectName::new("acme-evals").unwrap();
        let projector = DefaultClusterTextProjector::new().with_project_name(project_name.clone());

        write_embeddings(
            &cases,
            &out,
            &project_name,
            &projector,
            &StaticEmbeddingProvider,
        )
        .await
        .unwrap();

        let rows: Vec<CaseEmbedding> = JsonlFile::new(out).read_all().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].schema_version, "acme-evals.case_embedding.v1");
        assert_eq!(rows[0].provider, "test");
        assert_eq!(rows[0].model, "static");
        assert_eq!(rows[0].projection_version, "acme-evals.cluster_text.v1");
    }
}

#[cfg(all(test, feature = "cluster-label-openai"))]
mod label_tests {
    use tempfile::tempdir;

    use crate::clustering::{
        ClusterLabel, ClusterQuality, ClusterQualityReport, DiscoveredCluster,
    };

    use super::*;

    struct StaticLabeler;

    #[async_trait::async_trait]
    impl ClusterLabeler for StaticLabeler {
        fn labeler_name(&self) -> String {
            "test-labeler".to_string()
        }

        async fn label_cluster(
            &self,
            cluster: &DiscoveredCluster,
            examples: &[EvalCase],
        ) -> crate::Result<ClusterLabel> {
            assert_eq!(cluster.id, "cluster-1");
            assert_eq!(examples.len(), 1);
            Ok(
                ClusterLabel::new("Retrieval misses", "Missing source context")
                    .with_confidence(0.9),
            )
        }
    }

    #[tokio::test]
    async fn writes_labeled_model_and_report_clusters() {
        let dir = tempdir().unwrap();
        let out_model = dir.path().join("labeled_model.json");
        let out_clusters = dir.path().join("clusters.jsonl");
        let cases = vec![EvalCase::new("case-1", "trace-1", "question")];
        let model = ClusterModel::new(
            "model-1",
            "2026-01-01T00:00:00Z",
            crate::clustering::ClusterModelSource {
                case_count: 1,
                embedding_provider: None,
                embedding_model: None,
                embedding_dimensions: None,
                projection_version: None,
                algorithm: "test".to_string(),
                distance_metric: "cosine".to_string(),
                random_seed: 42,
            },
            vec![DiscoveredCluster {
                quality: ClusterQuality::new("cluster-1", 1),
                ..DiscoveredCluster::new("cluster-1", 1, vec!["case-1".to_string()])
            }],
            Vec::new(),
            ClusterQualityReport {
                cluster_count: 1,
                assigned_case_count: 1,
                mean_distance: None,
                silhouette_score: None,
                clusters: vec![ClusterQuality::new("cluster-1", 1)],
            },
        );

        write_labeled_model(model, &cases, &out_model, &out_clusters, &StaticLabeler)
            .await
            .unwrap();

        let labeled_model: ClusterModel = JsonFile::new(out_model).read().unwrap();
        let report_clusters: Vec<EvalCluster> = JsonlFile::new(out_clusters).read_all().unwrap();

        assert_eq!(
            labeled_model.clusters[0].label.as_ref().unwrap().label,
            "Retrieval misses"
        );
        assert_eq!(report_clusters[0].label, "Retrieval misses");
        assert_eq!(
            report_clusters[0].description.as_deref(),
            Some("Missing source context")
        );
    }
}
