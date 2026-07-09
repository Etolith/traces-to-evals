use std::path::Path;

use anyhow::{Context, Result};

use crate::model::EvalCase;

#[derive(Debug, Default, Clone, Copy)]
pub struct EvalCaseCsvExporter;

impl EvalCaseCsvExporter {
    pub fn write(path: impl AsRef<Path>, cases: &[EvalCase]) -> Result<()> {
        let mut writer = csv::Writer::from_path(path.as_ref())
            .with_context(|| format!("failed to create CSV {}", path.as_ref().display()))?;

        writer.write_record([
            "id",
            "trace_id",
            "input",
            "actual_output",
            "expected_output",
            "rubric",
        ])?;

        for case in cases {
            writer.write_record([
                case.id.as_str(),
                case.trace_id.as_str(),
                case.input.as_str(),
                case.actual_output.as_deref().unwrap_or_default(),
                case.expected_output.as_deref().unwrap_or_default(),
                case.rubric.as_deref().unwrap_or_default(),
            ])?;
        }

        writer.flush()?;
        Ok(())
    }
}
