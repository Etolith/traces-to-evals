use anyhow::Result;

use crate::calibration::{CalibrationModel, HumanRating};
use crate::cli::CalibrateArgs;
use crate::evaluation::EvaluationResult;
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlFile;

pub fn run(args: CalibrateArgs) -> Result<()> {
    let human_ratings: Vec<HumanRating> = JsonlFile::new(&args.human_ratings).read_all()?;
    let results: Vec<EvaluationResult> = JsonlFile::new(&args.results).read_all()?;
    let model = CalibrationModel::fit(&human_ratings, &results, args.pass_threshold)?;

    JsonFile::new(args.out).write_pretty(&model)
}
