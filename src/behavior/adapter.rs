use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Result, TraceEvalError};

use super::model::{OperationEffect, RetrySafety, ToolRequirement};

pub const BEHAVIOR_ADAPTER_SCHEMA_VERSION: &str = "traceeval.behavior_adapter.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BehaviorAdapterConfig {
    pub schema_version: String,
    pub adapter_id: String,
    pub adapter_version: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tool_mappings: BTreeMap<String, ToolSemanticMapping>,
}

impl Default for BehaviorAdapterConfig {
    fn default() -> Self {
        Self {
            schema_version: BEHAVIOR_ADAPTER_SCHEMA_VERSION.to_string(),
            adapter_id: "generic_openinference".to_string(),
            adapter_version: "1".to_string(),
            tool_mappings: BTreeMap::new(),
        }
    }
}

impl BehaviorAdapterConfig {
    pub fn new(adapter_id: impl Into<String>, adapter_version: impl Into<String>) -> Self {
        Self {
            adapter_id: adapter_id.into(),
            adapter_version: adapter_version.into(),
            ..Self::default()
        }
    }

    pub fn with_tool_mapping(
        mut self,
        tool_name: impl Into<String>,
        mapping: ToolSemanticMapping,
    ) -> Self {
        self.tool_mappings.insert(tool_name.into(), mapping);
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != BEHAVIOR_ADAPTER_SCHEMA_VERSION {
            return Err(TraceEvalError::InvalidBehaviorAdapter {
                adapter_id: self.adapter_id.clone(),
                message: format!(
                    "unsupported schema_version {}; expected {}",
                    self.schema_version, BEHAVIOR_ADAPTER_SCHEMA_VERSION
                ),
            });
        }
        if self.adapter_id.trim().is_empty() {
            return Err(TraceEvalError::InvalidBehaviorAdapter {
                adapter_id: self.adapter_id.clone(),
                message: "adapter_id must not be empty".to_string(),
            });
        }
        if self.adapter_version.trim().is_empty() {
            return Err(TraceEvalError::InvalidBehaviorAdapter {
                adapter_id: self.adapter_id.clone(),
                message: "adapter_version must not be empty".to_string(),
            });
        }
        for (tool_name, mapping) in &self.tool_mappings {
            if tool_name.trim().is_empty() {
                return Err(TraceEvalError::InvalidBehaviorAdapter {
                    adapter_id: self.adapter_id.clone(),
                    message: "tool mapping names must not be empty".to_string(),
                });
            }
            if mapping
                .operation
                .as_deref()
                .is_some_and(|operation| !is_valid_semantic_label(operation))
            {
                return Err(TraceEvalError::InvalidBehaviorAdapter {
                    adapter_id: self.adapter_id.clone(),
                    message: format!(
                        "operation for tool {tool_name} must be a bounded semantic label"
                    ),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn mapping_for(&self, tool_name: &str) -> ToolSemanticMapping {
        self.tool_mappings
            .get(tool_name)
            .cloned()
            .unwrap_or_default()
    }
}

pub(crate) fn is_valid_semantic_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | ':' | '/')
        })
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSemanticMapping {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(default)]
    pub effect: OperationEffect,
    #[serde(default)]
    pub retry_safety: RetrySafety,
    #[serde(default)]
    pub requirement: ToolRequirement,
}

impl ToolSemanticMapping {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: Some(operation.into()),
            ..Self::default()
        }
    }

    pub fn with_effect(mut self, effect: OperationEffect) -> Self {
        self.effect = effect;
        self
    }

    pub fn with_retry_safety(mut self, retry_safety: RetrySafety) -> Self {
        self.retry_safety = retry_safety;
        self
    }

    pub fn with_requirement(mut self, requirement: ToolRequirement) -> Self {
        self.requirement = requirement;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_adapter_schema() {
        let config = BehaviorAdapterConfig {
            schema_version: "traceeval.behavior_adapter.v2".to_string(),
            ..BehaviorAdapterConfig::default()
        };

        assert!(matches!(
            config.validate(),
            Err(TraceEvalError::InvalidBehaviorAdapter { .. })
        ));
    }

    #[test]
    fn rejects_unbounded_operation_labels() {
        let config = BehaviorAdapterConfig::new("test", "1").with_tool_mapping(
            "tool",
            ToolSemanticMapping::new("cancel card for customer 123"),
        );

        assert!(matches!(
            config.validate(),
            Err(TraceEvalError::InvalidBehaviorAdapter { .. })
        ));
    }
}
