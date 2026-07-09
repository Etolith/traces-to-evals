use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

use crate::{Result, TraceEvalError};

pub const DEFAULT_PROJECT_NAME: &str = "traceeval";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ProjectName(String);

impl Default for ProjectName {
    fn default() -> Self {
        Self(DEFAULT_PROJECT_NAME.to_string())
    }
}

impl ProjectName {
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let name = name.trim();
        if name.is_empty() {
            return Err(TraceEvalError::InvalidProjectName {
                name: name.to_string(),
                message: "project name must not be empty".to_string(),
            });
        }
        if !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(TraceEvalError::InvalidProjectName {
                name: name.to_string(),
                message: "project name may only contain ASCII letters, numbers, '.', '_', or '-'"
                    .to_string(),
            });
        }

        Ok(Self(name.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn schema_version(&self, artifact_kind: &str, version: u16) -> String {
        format!("{}.{}.v{}", self.0, artifact_kind, version)
    }

    pub fn case_embedding_schema_version(&self) -> String {
        self.schema_version("case_embedding", 1)
    }

    pub fn cluster_model_schema_version(&self) -> String {
        self.schema_version("cluster_model", 1)
    }

    pub fn cluster_text_projection_version(&self, include_output: bool) -> String {
        let mut version = self.schema_version("cluster_text", 1);
        if include_output {
            version.push_str(".include_output");
        }
        version
    }

    pub fn matches_schema_version(value: &str, artifact_kind: &str, version: u16) -> bool {
        let suffix = format!(".{}.v{}", artifact_kind, version);
        let Some(project_name) = value.strip_suffix(suffix.as_str()) else {
            return false;
        };

        Self::new(project_name).is_ok()
    }
}

impl AsRef<str> for ProjectName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for ProjectName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ProjectName {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::new(name).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_project_scoped_schema_versions() {
        let project = ProjectName::new("acme-evals").unwrap();

        assert_eq!(
            project.case_embedding_schema_version(),
            "acme-evals.case_embedding.v1"
        );
        assert_eq!(
            project.cluster_text_projection_version(true),
            "acme-evals.cluster_text.v1.include_output"
        );
    }

    #[test]
    fn validates_schema_namespace_and_kind() {
        assert!(ProjectName::matches_schema_version(
            "acme.case_embedding.v1",
            "case_embedding",
            1
        ));
        assert!(!ProjectName::matches_schema_version(
            "acme.case_embedding.v2",
            "case_embedding",
            1
        ));
        assert!(!ProjectName::matches_schema_version(
            "bad name.case_embedding.v1",
            "case_embedding",
            1
        ));
    }

    #[test]
    fn serde_deserialize_validates_project_name() {
        let project: ProjectName = serde_json::from_str("\"acme-evals\"").unwrap();
        assert_eq!(project.as_str(), "acme-evals");

        assert!(serde_json::from_str::<ProjectName>("\"bad name\"").is_err());
    }
}
