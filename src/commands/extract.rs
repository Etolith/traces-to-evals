use anyhow::Result;

use crate::cli::{ExtractArgs, ExtractFormat};
use crate::extractors::{OpenInferenceExtractor, SimpleExtractor};
use crate::io::jsonl::JsonlFile;
use crate::model::{EvalCase, Trace};

pub fn run(args: ExtractArgs) -> Result<()> {
    let traces: Vec<Trace> = JsonlFile::new(&args.traces).read_all()?;

    let cases: Vec<EvalCase> = match args.format {
        ExtractFormat::Simple => SimpleExtractor.extract_traces(&traces)?,
        ExtractFormat::OpenInference => OpenInferenceExtractor.extract_traces(&traces)?,
    };

    JsonlFile::new(&args.out).write_all(&cases)
}
