use anyhow::Result;

use crate::behavior::{BehaviorFinding, FindingSeverity, PairedFindingVerifier};
use crate::cli::{FindingSeverityName, VerifyFindingsArgs};
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlFile;

pub fn run(args: VerifyFindingsArgs) -> Result<()> {
    let baseline: Vec<BehaviorFinding> = JsonlFile::new(&args.baseline).read_all()?;
    let candidate: Vec<BehaviorFinding> = JsonlFile::new(&args.candidate).read_all()?;
    let report = PairedFindingVerifier::default()
        .with_severe_threshold(severity(args.severe_threshold))
        .verify(args.case_id, args.target_signatures, &baseline, &candidate);
    JsonFile::new(args.out).write_pretty(&report)?;
    Ok(())
}

fn severity(value: FindingSeverityName) -> FindingSeverity {
    match value {
        FindingSeverityName::Info => FindingSeverity::Info,
        FindingSeverityName::Low => FindingSeverity::Low,
        FindingSeverityName::Medium => FindingSeverity::Medium,
        FindingSeverityName::High => FindingSeverity::High,
        FindingSeverityName::Critical => FindingSeverity::Critical,
    }
}
