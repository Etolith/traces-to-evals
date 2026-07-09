use std::path::Path;

use serde::Serialize;

use crate::Result;
use crate::io::jsonl::JsonlFile;
use crate::model::EvalCase;

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiEvalExporter;

#[derive(Debug, Serialize)]
struct OpenAiEvalRow<'a> {
    input: &'a str,
    ideal: &'a str,
    metadata: OpenAiEvalMetadata<'a>,
}

#[derive(Debug, Serialize)]
struct OpenAiEvalMetadata<'a> {
    case_id: &'a str,
    trace_id: &'a str,
}

impl OpenAiEvalExporter {
    pub fn write_jsonl(path: impl AsRef<Path>, cases: &[EvalCase]) -> Result<()> {
        let rows = cases.iter().map(|case| OpenAiEvalRow {
            input: case.input.as_str(),
            ideal: case.expected_output.as_deref().unwrap_or_default(),
            metadata: OpenAiEvalMetadata {
                case_id: case.id.as_str(),
                trace_id: case.trace_id.as_str(),
            },
        });

        JsonlFile::new(path).write_iter(rows)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn writes_openai_eval_rows() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("openai.jsonl");
        let cases = vec![EvalCase::new("case-1", "trace-1", "input").with_expected_output("ideal")];

        OpenAiEvalExporter::write_jsonl(&path, &cases).unwrap();
        let rendered = fs::read_to_string(path).unwrap();

        assert!(rendered.contains(r#""input":"input""#));
        assert!(rendered.contains(r#""ideal":"ideal""#));
    }
}
