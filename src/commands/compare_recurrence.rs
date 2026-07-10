use anyhow::Result;

use crate::behavior::{BehaviorFinding, FindingRecurrenceComparator, FindingRecurrenceRequest};
use crate::cli::CompareRecurrenceArgs;
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlFile;

pub fn run(args: CompareRecurrenceArgs) -> Result<()> {
    let request: FindingRecurrenceRequest = JsonFile::new(&args.request).read()?;
    let baseline_findings: Vec<BehaviorFinding> =
        JsonlFile::new(&args.baseline_findings).read_all()?;
    let candidate_findings: Vec<BehaviorFinding> =
        JsonlFile::new(&args.candidate_findings).read_all()?;
    let comparison = FindingRecurrenceComparator::new().compare_request(
        request,
        &baseline_findings,
        &candidate_findings,
    )?;
    JsonFile::new(args.out).write_pretty(&comparison)?;
    Ok(())
}
