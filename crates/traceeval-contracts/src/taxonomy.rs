use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{
    AGENT_TAXONOMY_RELEASE_HASH_DOMAIN, ContractError, TAXONOMY_ASSIGNMENT_HASH_DOMAIN,
    canonical_content_id, require_non_empty, require_sha256,
};

pub const AGENT_TAXONOMY_RELEASE_SCHEMA_VERSION: &str = "traceeval.agent_taxonomy_release.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaxonomyDimensionV1 {
    Task,
    Capability,
    SuccessCriterion,
    NonGoal,
    Policy,
    Risk,
    Escalation,
    FailureMode,
    RootCause,
    Severity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaxonomyNodeStateV1 {
    Active,
    Retired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaxonomyNodeV1 {
    pub node_id: String,
    pub dimension: TaxonomyDimensionV1,
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub aliases: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub parent_ids: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowed_relation_types: BTreeSet<TaxonomyRelationKindV1>,
    pub state: TaxonomyNodeStateV1,
    pub provenance: String,
    pub sensitivity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub portable_base_term: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaxonomyRelationKindV1 {
    Uses,
    Affects,
    Violates,
    Requires,
    EscalatesTo,
    CausedBy,
    RelatedTo,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaxonomyRelationV1 {
    pub source_node_id: String,
    pub kind: TaxonomyRelationKindV1,
    pub target_node_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum TaxonomyLineageOperationV1 {
    Create {
        node_id: String,
    },
    MatchExisting {
        proposed_id: String,
        existing_id: String,
    },
    Merge {
        source_ids: BTreeSet<String>,
        target_id: String,
    },
    Split {
        source_id: String,
        target_ids: BTreeSet<String>,
    },
    Reparent {
        node_id: String,
        previous_parent_ids: BTreeSet<String>,
        new_parent_ids: BTreeSet<String>,
    },
    Rename {
        node_id: String,
        previous_name: String,
        new_name: String,
    },
    Retire {
        node_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentTaxonomyReleaseV1 {
    pub schema_version: String,
    pub taxonomy_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_release_id: Option<String>,
    pub nodes: Vec<TaxonomyNodeV1>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub relations: BTreeSet<TaxonomyRelationV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lineage: Vec<TaxonomyLineageOperationV1>,
}

impl AgentTaxonomyReleaseV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != AGENT_TAXONOMY_RELEASE_SCHEMA_VERSION {
            return Err(taxonomy_error("unsupported taxonomy schema version"));
        }
        require_non_empty(&self.taxonomy_id, "taxonomy_id", taxonomy_error)?;
        if let Some(previous_release_id) = &self.previous_release_id {
            require_sha256(previous_release_id, "previous_release_id", taxonomy_error)?;
        }
        let mut nodes = BTreeMap::new();
        for node in &self.nodes {
            require_non_empty(&node.node_id, "node_id", taxonomy_error)?;
            require_non_empty(&node.name, "node name", taxonomy_error)?;
            require_non_empty(&node.description, "node description", taxonomy_error)?;
            require_non_empty(&node.provenance, "node provenance", taxonomy_error)?;
            require_non_empty(&node.sensitivity, "node sensitivity", taxonomy_error)?;
            if let Some(portable_base_term) = &node.portable_base_term {
                require_non_empty(portable_base_term, "portable_base_term", taxonomy_error)?;
            }
            if nodes.insert(node.node_id.as_str(), node).is_some() {
                return Err(taxonomy_error(format!(
                    "duplicate node_id {}",
                    node.node_id
                )));
            }
        }
        for node in &self.nodes {
            for parent_id in &node.parent_ids {
                let parent = nodes.get(parent_id.as_str()).ok_or_else(|| {
                    taxonomy_error(format!(
                        "node {} references unknown parent {parent_id}",
                        node.node_id
                    ))
                })?;
                if parent.dimension != node.dimension {
                    return Err(taxonomy_error(format!(
                        "node {} crosses dimensions through parent {parent_id}",
                        node.node_id
                    )));
                }
            }
        }
        ensure_acyclic(&nodes)?;
        for relation in &self.relations {
            if relation.source_node_id == relation.target_node_id {
                return Err(taxonomy_error("self-relations are not allowed"));
            }
            if !nodes.contains_key(relation.source_node_id.as_str())
                || !nodes.contains_key(relation.target_node_id.as_str())
            {
                return Err(taxonomy_error(format!(
                    "relation {:?} references an unknown node",
                    relation.kind
                )));
            }
            let source = nodes[relation.source_node_id.as_str()];
            if !source.allowed_relation_types.is_empty()
                && !source.allowed_relation_types.contains(&relation.kind)
            {
                return Err(taxonomy_error(format!(
                    "relation {:?} is not allowed by source node {}",
                    relation.kind, relation.source_node_id
                )));
            }
        }
        validate_lineage(&self.lineage, &nodes)?;
        Ok(())
    }

    pub fn release_id(&self) -> Result<String, ContractError> {
        self.validate()?;
        let mut normalized = self.clone();
        normalized
            .nodes
            .sort_by(|left, right| left.node_id.cmp(&right.node_id));
        canonical_content_id(AGENT_TAXONOMY_RELEASE_HASH_DOMAIN, &normalized)
    }

    /// Validates lineage claims against the exact prior immutable release.
    pub fn validate_transition(
        &self,
        previous: &AgentTaxonomyReleaseV1,
    ) -> Result<(), ContractError> {
        self.validate()?;
        previous.validate()?;
        if self.taxonomy_id != previous.taxonomy_id {
            return Err(taxonomy_error(
                "taxonomy transition must keep the same taxonomy_id",
            ));
        }
        if self.previous_release_id.as_deref() != Some(previous.release_id()?.as_str()) {
            return Err(taxonomy_error(
                "previous_release_id does not match the supplied prior release",
            ));
        }
        let prior: BTreeMap<&str, &TaxonomyNodeV1> = previous
            .nodes
            .iter()
            .map(|node| (node.node_id.as_str(), node))
            .collect();
        let next: BTreeMap<&str, &TaxonomyNodeV1> = self
            .nodes
            .iter()
            .map(|node| (node.node_id.as_str(), node))
            .collect();
        for operation in &self.lineage {
            validate_lineage_transition(operation, &prior, &next)?;
        }
        Ok(())
    }
}

fn validate_lineage_transition(
    operation: &TaxonomyLineageOperationV1,
    prior: &BTreeMap<&str, &TaxonomyNodeV1>,
    next: &BTreeMap<&str, &TaxonomyNodeV1>,
) -> Result<(), ContractError> {
    match operation {
        TaxonomyLineageOperationV1::Create { node_id } => {
            if prior.contains_key(node_id.as_str()) || !next.contains_key(node_id.as_str()) {
                return Err(taxonomy_error(
                    "created node must be absent from the prior release and present in the next",
                ));
            }
        }
        TaxonomyLineageOperationV1::MatchExisting { existing_id, .. } => {
            if !prior.contains_key(existing_id.as_str()) || !next.contains_key(existing_id.as_str())
            {
                return Err(taxonomy_error(
                    "matched existing node must persist across the transition",
                ));
            }
        }
        TaxonomyLineageOperationV1::Merge {
            source_ids,
            target_id,
        } => {
            let target = next
                .get(target_id.as_str())
                .ok_or_else(|| taxonomy_error("merge target must exist in the next release"))?;
            if target.state != TaxonomyNodeStateV1::Active {
                return Err(taxonomy_error("merge target must be active"));
            }
            for source_id in source_ids {
                let prior_source = prior.get(source_id.as_str()).ok_or_else(|| {
                    taxonomy_error("merge source must exist in the prior release")
                })?;
                let next_source = next.get(source_id.as_str()).ok_or_else(|| {
                    taxonomy_error("merge source must remain as a retired lineage node")
                })?;
                if next_source.state != TaxonomyNodeStateV1::Retired
                    || prior_source.dimension != target.dimension
                {
                    return Err(taxonomy_error(
                        "merge sources must retire into a target of the same dimension",
                    ));
                }
            }
        }
        TaxonomyLineageOperationV1::Split {
            source_id,
            target_ids,
        } => {
            let prior_source = prior
                .get(source_id.as_str())
                .ok_or_else(|| taxonomy_error("split source must exist in the prior release"))?;
            let next_source = next.get(source_id.as_str()).ok_or_else(|| {
                taxonomy_error("split source must remain as a retired lineage node")
            })?;
            if next_source.state != TaxonomyNodeStateV1::Retired {
                return Err(taxonomy_error("split source must be retired"));
            }
            for target_id in target_ids {
                let target = next
                    .get(target_id.as_str())
                    .ok_or_else(|| taxonomy_error("split target must exist in the next release"))?;
                if target.state != TaxonomyNodeStateV1::Active
                    || target.dimension != prior_source.dimension
                {
                    return Err(taxonomy_error(
                        "split targets must be active and keep the source dimension",
                    ));
                }
            }
        }
        TaxonomyLineageOperationV1::Reparent {
            node_id,
            previous_parent_ids,
            new_parent_ids,
        } => {
            let prior_node = prior
                .get(node_id.as_str())
                .ok_or_else(|| taxonomy_error("reparented node missing from prior release"))?;
            let next_node = next
                .get(node_id.as_str())
                .ok_or_else(|| taxonomy_error("reparented node missing from next release"))?;
            if &prior_node.parent_ids != previous_parent_ids
                || &next_node.parent_ids != new_parent_ids
            {
                return Err(taxonomy_error(
                    "reparent lineage does not match prior and next parent state",
                ));
            }
        }
        TaxonomyLineageOperationV1::Rename {
            node_id,
            previous_name,
            new_name,
        } => {
            let prior_node = prior
                .get(node_id.as_str())
                .ok_or_else(|| taxonomy_error("renamed node missing from prior release"))?;
            let next_node = next
                .get(node_id.as_str())
                .ok_or_else(|| taxonomy_error("renamed node missing from next release"))?;
            if &prior_node.name != previous_name || &next_node.name != new_name {
                return Err(taxonomy_error(
                    "rename lineage does not match prior and next names",
                ));
            }
        }
        TaxonomyLineageOperationV1::Retire { node_id } => {
            let prior_node = prior
                .get(node_id.as_str())
                .ok_or_else(|| taxonomy_error("retired node missing from prior release"))?;
            let next_node = next
                .get(node_id.as_str())
                .ok_or_else(|| taxonomy_error("retired node missing from next release"))?;
            if prior_node.state != TaxonomyNodeStateV1::Active
                || next_node.state != TaxonomyNodeStateV1::Retired
            {
                return Err(taxonomy_error(
                    "retire lineage requires an active-to-retired transition",
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaxonomyOpenSetStateV1 {
    Known,
    Unknown,
    Other,
    Novel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaxonomyAssignmentSourceV1 {
    LearnedAssessment,
    LearnedDiscovery,
    HumanReview,
    Imported,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaxonomyAssignmentV1 {
    pub subject_revision: String,
    pub taxonomy_release_id: String,
    pub open_set_state: TaxonomyOpenSetStateV1,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub node_ids: BTreeSet<String>,
    pub source: TaxonomyAssignmentSourceV1,
    pub source_identity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_reported_confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub membership_strength: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub evidence_keys: BTreeSet<String>,
}

impl TaxonomyAssignmentV1 {
    pub fn validate_against(&self, taxonomy: &AgentTaxonomyReleaseV1) -> Result<(), ContractError> {
        taxonomy.validate()?;
        require_sha256(
            &self.taxonomy_release_id,
            "taxonomy_release_id",
            taxonomy_error,
        )?;
        if self.taxonomy_release_id != taxonomy.release_id()? {
            return Err(taxonomy_error(
                "assignment taxonomy_release_id does not match taxonomy",
            ));
        }
        require_non_empty(&self.subject_revision, "subject_revision", taxonomy_error)?;
        require_non_empty(&self.source_identity, "source_identity", taxonomy_error)?;
        if self.evidence_keys.is_empty() {
            return Err(taxonomy_error(
                "assignment requires at least one evidence key",
            ));
        }
        match self.open_set_state {
            TaxonomyOpenSetStateV1::Known if self.node_ids.is_empty() => {
                return Err(taxonomy_error(
                    "known assignment requires at least one node",
                ));
            }
            TaxonomyOpenSetStateV1::Unknown
            | TaxonomyOpenSetStateV1::Other
            | TaxonomyOpenSetStateV1::Novel
                if !self.node_ids.is_empty() =>
            {
                return Err(taxonomy_error(
                    "open-set assignment cannot force a known taxonomy node",
                ));
            }
            _ => {}
        }
        let known_ids: BTreeSet<&str> = taxonomy
            .nodes
            .iter()
            .map(|node| node.node_id.as_str())
            .collect();
        for node_id in &self.node_ids {
            if !known_ids.contains(node_id.as_str()) {
                return Err(taxonomy_error(format!(
                    "assignment references unknown node {node_id}"
                )));
            }
        }
        validate_probability(self.model_reported_confidence, "model_reported_confidence")?;
        validate_probability(self.membership_strength, "membership_strength")?;
        Ok(())
    }

    pub fn assignment_id(
        &self,
        taxonomy: &AgentTaxonomyReleaseV1,
    ) -> Result<String, ContractError> {
        self.validate_against(taxonomy)?;
        let evidence_digest = canonical_content_id(
            "perseval.taxonomy-assignment-evidence.v1",
            &self.evidence_keys,
        )?;
        canonical_content_id(
            TAXONOMY_ASSIGNMENT_HASH_DOMAIN,
            &TaxonomyAssignmentIdentity {
                subject_revision: &self.subject_revision,
                agent_taxonomy_release_id: &self.taxonomy_release_id,
                open_set_state: self.open_set_state,
                ordered_node_ids: &self.node_ids,
                assignment_source: self.source,
                source_identity: &self.source_identity,
                evidence_digest: &evidence_digest,
            },
        )
    }
}

#[derive(Serialize)]
struct TaxonomyAssignmentIdentity<'a> {
    subject_revision: &'a str,
    agent_taxonomy_release_id: &'a str,
    open_set_state: TaxonomyOpenSetStateV1,
    ordered_node_ids: &'a BTreeSet<String>,
    assignment_source: TaxonomyAssignmentSourceV1,
    source_identity: &'a str,
    evidence_digest: &'a str,
}

fn ensure_acyclic(nodes: &BTreeMap<&str, &TaxonomyNodeV1>) -> Result<(), ContractError> {
    fn visit<'a>(
        node_id: &'a str,
        nodes: &BTreeMap<&'a str, &'a TaxonomyNodeV1>,
        visiting: &mut BTreeSet<&'a str>,
        visited: &mut BTreeSet<&'a str>,
    ) -> Result<(), ContractError> {
        if visited.contains(node_id) {
            return Ok(());
        }
        if !visiting.insert(node_id) {
            return Err(taxonomy_error(format!(
                "taxonomy parent cycle includes {node_id}"
            )));
        }
        let node = nodes[node_id];
        for parent_id in &node.parent_ids {
            visit(parent_id, nodes, visiting, visited)?;
        }
        visiting.remove(node_id);
        visited.insert(node_id);
        Ok(())
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for node_id in nodes.keys() {
        visit(node_id, nodes, &mut visiting, &mut visited)?;
    }
    Ok(())
}

fn validate_lineage(
    lineage: &[TaxonomyLineageOperationV1],
    nodes: &BTreeMap<&str, &TaxonomyNodeV1>,
) -> Result<(), ContractError> {
    let known = |node_id: &str| nodes.contains_key(node_id);
    for operation in lineage {
        match operation {
            TaxonomyLineageOperationV1::Create { node_id } => {
                require_known(node_id, "created node", &known)?;
            }
            TaxonomyLineageOperationV1::MatchExisting {
                proposed_id,
                existing_id,
            } => {
                require_non_empty(proposed_id, "proposed_id", taxonomy_error)?;
                require_known(existing_id, "existing node", &known)?;
                if proposed_id == existing_id {
                    return Err(taxonomy_error(
                        "match-existing proposed_id must differ from existing_id",
                    ));
                }
            }
            TaxonomyLineageOperationV1::Merge {
                source_ids,
                target_id,
            } => {
                if source_ids.len() < 2 {
                    return Err(taxonomy_error("merge requires at least two source nodes"));
                }
                require_known(target_id, "merge target", &known)?;
                for source_id in source_ids {
                    require_known(source_id, "merge source", &known)?;
                    if source_id == target_id {
                        return Err(taxonomy_error("merge target cannot also be a merge source"));
                    }
                    require_same_dimension(source_id, target_id, nodes, "merge")?;
                }
            }
            TaxonomyLineageOperationV1::Split {
                source_id,
                target_ids,
            } => {
                require_known(source_id, "split source", &known)?;
                if target_ids.len() < 2 {
                    return Err(taxonomy_error("split requires at least two target nodes"));
                }
                for target_id in target_ids {
                    require_known(target_id, "split target", &known)?;
                    if target_id == source_id {
                        return Err(taxonomy_error("split source cannot also be a split target"));
                    }
                    require_same_dimension(source_id, target_id, nodes, "split")?;
                }
            }
            TaxonomyLineageOperationV1::Reparent {
                node_id,
                previous_parent_ids,
                new_parent_ids,
            } => {
                require_known(node_id, "reparented node", &known)?;
                if previous_parent_ids == new_parent_ids {
                    return Err(taxonomy_error("reparent operation must change parents"));
                }
                for parent_id in previous_parent_ids.iter().chain(new_parent_ids.iter()) {
                    require_known(parent_id, "reparent parent", &known)?;
                    require_same_dimension(node_id, parent_id, nodes, "reparent")?;
                }
                if &nodes[node_id.as_str()].parent_ids != new_parent_ids {
                    return Err(taxonomy_error(
                        "reparent new_parent_ids must match the released node state",
                    ));
                }
            }
            TaxonomyLineageOperationV1::Rename {
                node_id,
                previous_name,
                new_name,
            } => {
                require_known(node_id, "renamed node", &known)?;
                require_non_empty(previous_name, "previous_name", taxonomy_error)?;
                require_non_empty(new_name, "new_name", taxonomy_error)?;
                if previous_name == new_name
                    || nodes[node_id.as_str()].name.as_str() != new_name.as_str()
                {
                    return Err(taxonomy_error(
                        "rename must change the name and match the released node state",
                    ));
                }
            }
            TaxonomyLineageOperationV1::Retire { node_id } => {
                require_known(node_id, "retired node", &known)?;
                if nodes[node_id.as_str()].state != TaxonomyNodeStateV1::Retired {
                    return Err(taxonomy_error(
                        "retire operation requires a retired released node",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn require_known(
    node_id: &str,
    role: &str,
    known: &impl Fn(&str) -> bool,
) -> Result<(), ContractError> {
    require_non_empty(node_id, role, taxonomy_error)?;
    if !known(node_id) {
        return Err(taxonomy_error(format!(
            "lineage {role} references unknown node {node_id}"
        )));
    }
    Ok(())
}

fn require_same_dimension(
    left_id: &str,
    right_id: &str,
    nodes: &BTreeMap<&str, &TaxonomyNodeV1>,
    operation: &str,
) -> Result<(), ContractError> {
    if nodes[left_id].dimension != nodes[right_id].dimension {
        return Err(taxonomy_error(format!(
            "{operation} cannot cross taxonomy dimensions"
        )));
    }
    Ok(())
}

fn validate_probability(value: Option<f64>, field: &str) -> Result<(), ContractError> {
    if let Some(value) = value
        && (!value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(taxonomy_error(format!(
            "{field} must be finite and between 0 and 1"
        )));
    }
    Ok(())
}

fn taxonomy_error(message: impl Into<String>) -> ContractError {
    ContractError::InvalidTaxonomy(message.into())
}
