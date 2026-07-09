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

    let aggregate = match args.clusters {
        Some(path) => {
            let clusters: Vec<EvalCluster> = JsonlFile::new(path).read_all()?;
            clusters
                .into_iter()
                .fold(WeightedAggregate::new(), |aggregate, cluster| {
                    aggregate.with_cluster_weight(cluster.id, cluster.weight)
                })
        }
        None => WeightedAggregate::default(),
    };

    let report = EvaluationReport::from_results_with_aggregate(&results, &aggregate);

    JsonFile::new(args.out).write_pretty(&report)
}
