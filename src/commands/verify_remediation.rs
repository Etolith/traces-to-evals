use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

use crate::behavior::{
    BehaviorFinding, RemediationInputArtifacts, RemediationVerificationRequest,
    RemediationVerifier, VerificationArtifactDigest,
};
use crate::cli::VerifyRemediationArgs;
use crate::evaluation::EvaluationResult;
use crate::io::json::JsonFile;
use crate::io::jsonl::JsonlReader;

pub fn run(args: VerifyRemediationArgs) -> Result<()> {
    let mut request: RemediationVerificationRequest = JsonFile::new(&args.request).read()?;
    let (baseline_findings, baseline_findings_digest): (Vec<BehaviorFinding>, _) =
        read_jsonl_artifact(&args.baseline_findings)?;
    let (candidate_findings, candidate_findings_digest): (Vec<BehaviorFinding>, _) =
        read_jsonl_artifact(&args.candidate_findings)?;
    let (baseline_results, baseline_results_digest): (Vec<EvaluationResult>, _) =
        read_jsonl_artifact(&args.baseline_results)?;
    let (candidate_results, candidate_results_digest): (Vec<EvaluationResult>, _) =
        read_jsonl_artifact(&args.candidate_results)?;
    let input_artifacts = RemediationInputArtifacts {
        baseline_findings: baseline_findings_digest,
        candidate_findings: candidate_findings_digest,
        baseline_results: baseline_results_digest,
        candidate_results: candidate_results_digest,
    };
    if request
        .input_artifacts
        .as_ref()
        .is_some_and(|declared| declared != &input_artifacts)
    {
        bail!("declared input_artifacts do not match the supplied files");
    }
    request.input_artifacts = Some(input_artifacts);
    let report = RemediationVerifier::new().verify_request(
        request,
        &baseline_findings,
        &candidate_findings,
        &baseline_results,
        &candidate_results,
    )?;
    JsonFile::new(args.out).write_pretty(&report)?;
    if !report.passed {
        bail!(
            "remediation verification failed: {}",
            report.reasons.join("; ")
        );
    }
    Ok(())
}

fn read_jsonl_artifact<T>(path: &Path) -> Result<(Vec<T>, VerificationArtifactDigest)>
where
    T: DeserializeOwned,
{
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read verification artifact {}", path.display()))?;
    let records =
        JsonlReader::new(Cursor::new(bytes.as_slice()), path.display().to_string()).read_all()?;
    let digest = VerificationArtifactDigest {
        content_hash: format!("sha256:{:x}", Sha256::digest(&bytes)),
        byte_count: bytes.len() as u64,
        record_count: records.len() as u64,
    };
    Ok((records, digest))
}
