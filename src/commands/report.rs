use anyhow::Result;

use crate::calibration::CalibrationModel;
use crate::cli::ReportArgs;
use crate::clustering::EvalCluster;
use crate::evaluation::{EvaluationResult, WeightedAggregate};
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlFile;
use crate::report::EvaluationReport;

pub fn run(args: ReportArgs) -> Result<()> {
    let mut results: Vec<EvaluationResult> = JsonlFile::new(&args.results).read_all()?;

    if let Some(path) = args.calibration {
        let calibration: CalibrationModel = JsonFile::new(path).read()?;
        results = results
            .into_iter()
            .map(|result| calibration.apply(result))
            .collect();
    }

    let clusters = match args.clusters {
        Some(path) => JsonlFile::new(path).read_all::<EvalCluster>()?,
        None => Vec::new(),
    };
    let aggregate = clusters
        .iter()
        .fold(WeightedAggregate::new(), |aggregate, cluster| {
            aggregate.with_cluster_weight(cluster.id.clone(), cluster.weight)
        });

    let report =
        EvaluationReport::from_results_with_aggregate_and_clusters(&results, &aggregate, &clusters);

    JsonFile::new(args.out).write_pretty(&report)?;
    Ok(())
}
