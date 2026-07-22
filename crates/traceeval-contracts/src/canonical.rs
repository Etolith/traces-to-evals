use serde::Serialize;
use sha2::{Digest, Sha256};

use super::ContractError;

pub const EVALUATOR_RELEASE_HASH_DOMAIN: &str = "traceeval.evaluator-release.v1";
pub const AGENT_CONTEXT_RELEASE_HASH_DOMAIN: &str = "perseval.agent-context-release.v1";
pub const CONTEXT_PROJECTION_HASH_DOMAIN: &str = "traceeval.context-projection.v1";
pub const TRACE_CONTEXT_BINDING_HASH_DOMAIN: &str = "perseval.trace-context-binding.v1";
pub const AGENT_TAXONOMY_RELEASE_HASH_DOMAIN: &str = "perseval.agent-taxonomy-release.v1";
pub const TAXONOMY_ASSIGNMENT_HASH_DOMAIN: &str = "perseval.taxonomy-assignment.v1";

/// Serializes a contract with the JSON Canonicalization Scheme from RFC 8785.
pub fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, ContractError> {
    Ok(serde_json_canonicalizer::to_vec(value)?)
}

/// Computes a domain-separated SHA-256 identity over RFC 8785 canonical JSON.
pub fn canonical_content_id<T: Serialize>(
    domain: &str,
    value: &T,
) -> Result<String, ContractError> {
    let canonical = canonical_json_bytes(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update(canonical);
    Ok(format!("sha256:{:x}", hasher.finalize()))
}
