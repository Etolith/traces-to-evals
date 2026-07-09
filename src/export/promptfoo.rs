use std::path::Path;

use anyhow::Context;
use serde::Serialize;

use crate::Result;
use crate::model::EvalCase;

#[derive(Debug, Default, Clone, Copy)]
pub struct PromptfooExporter;

#[derive(Debug, Serialize)]
struct PromptfooFile<'a> {
    tests: Vec<PromptfooTest<'a>>,
}

#[derive(Debug, Serialize)]
struct PromptfooTest<'a> {
    vars: PromptfooVars<'a>,
    #[serde(rename = "assert")]
    assertions: Vec<PromptfooAssertion<'a>>,
    metadata: PromptfooMetadata<'a>,
}

#[derive(Debug, Serialize)]
struct PromptfooVars<'a> {
    input: &'a str,
}

#[derive(Debug, Serialize)]
struct PromptfooAssertion<'a> {
    #[serde(rename = "type")]
    assertion_type: &'a str,
    value: &'a str,
}

#[derive(Debug, Serialize)]
struct PromptfooMetadata<'a> {
    case_id: &'a str,
    trace_id: &'a str,
    rubric: Option<&'a str>,
}

impl PromptfooExporter {
    pub fn write_json(path: impl AsRef<Path>, cases: &[EvalCase]) -> Result<()> {
        let tests = cases
            .iter()
            .map(|case| PromptfooTest {
                vars: PromptfooVars {
                    input: case.input.as_str(),
                },
                assertions: vec![PromptfooAssertion {
                    assertion_type: "contains",
                    value: case.expected_output.as_deref().unwrap_or_default(),
                }],
                metadata: PromptfooMetadata {
                    case_id: case.id.as_str(),
                    trace_id: case.trace_id.as_str(),
                    rubric: case.rubric.as_deref(),
                },
            })
            .collect::<Vec<_>>();

        let rendered = serde_json::to_string_pretty(&PromptfooFile { tests })?;
        std::fs::write(path.as_ref(), rendered)
            .with_context(|| format!("failed to write {}", path.as_ref().display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn writes_promptfoo_tests() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("promptfoo.json");
        let cases = vec![EvalCase::new("case-1", "trace-1", "input").with_expected_output("ideal")];

        PromptfooExporter::write_json(&path, &cases).unwrap();
        let rendered = fs::read_to_string(path).unwrap();

        assert!(rendered.contains(r#""tests""#));
        assert!(rendered.contains(r#""case_id": "case-1""#));
        assert!(rendered.contains(r#""value": "ideal""#));
    }
}
