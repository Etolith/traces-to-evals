use anyhow::{Result, anyhow};

use crate::cli::ClusterArgs;
use crate::clustering::{
    ClusterAssigner, EvalCluster, RuleBasedClusterAssigner, apply_assignments_to_results,
};
use crate::evaluation::EvaluationResult;
use crate::io::jsonl::JsonlFile;
use crate::model::EvalCase;

pub fn run(args: ClusterArgs) -> Result<()> {
    let cases: Vec<EvalCase> = JsonlFile::new(&args.cases).read_all()?;
    let clusters: Vec<EvalCluster> = JsonlFile::new(&args.clusters).read_all()?;
    let assigner = RuleBasedClusterAssigner::new(clusters);
    let assignments = assigner.assign_cases(&cases)?;

    JsonlFile::new(&args.out).write_all(&assignments)?;

    match (args.results, args.results_out) {
        (Some(results_path), Some(out_path)) => {
            let results: Vec<EvaluationResult> = JsonlFile::new(results_path).read_all()?;
            let results = apply_assignments_to_results(results, &assignments);
            JsonlFile::new(out_path).write_all(&results)?;
            Ok(())
        }
        (None, Some(_)) => Err(anyhow!("--results-out requires --results")),
        _ => Ok(()),
    }
}
