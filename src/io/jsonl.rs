use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct JsonlFile<P> {
    path: P,
}

impl<P> JsonlFile<P> {
    pub fn new(path: P) -> Self {
        Self { path }
    }
}

impl<P> JsonlFile<P>
where
    P: AsRef<Path>,
{
    pub fn read_all<T>(&self) -> Result<Vec<T>>
    where
        T: serde::de::DeserializeOwned,
    {
        let file = File::open(self.path.as_ref())
            .with_context(|| format!("failed to open {}", self.path.as_ref().display()))?;

        JsonlReader::new(
            BufReader::new(file),
            self.path.as_ref().display().to_string(),
        )
        .read_all()
    }

    pub fn write_all<T>(&self, values: &[T]) -> Result<()>
    where
        T: Serialize,
    {
        self.write_iter(values.iter())
    }

    pub fn write_iter<T, I>(&self, values: I) -> Result<()>
    where
        T: Serialize,
        I: IntoIterator<Item = T>,
    {
        let file = File::create(self.path.as_ref())
            .with_context(|| format!("failed to create {}", self.path.as_ref().display()))?;

        JsonlWriter::new(BufWriter::new(file)).write_iter(values)
    }
}

pub struct JsonlReader<R> {
    reader: R,
    source_name: String,
}

impl<R> JsonlReader<R> {
    pub fn new(reader: R, source_name: impl Into<String>) -> Self {
        Self {
            reader,
            source_name: source_name.into(),
        }
    }
}

impl<R> JsonlReader<R>
where
    R: BufRead,
{
    pub fn read_all<T>(self) -> Result<Vec<T>>
    where
        T: serde::de::DeserializeOwned,
    {
        let mut values = Vec::new();

        for (index, line) in self.reader.lines().enumerate() {
            let line = line.with_context(|| {
                format!(
                    "failed to read line {} from {}",
                    index + 1,
                    self.source_name
                )
            })?;

            if line.trim().is_empty() {
                continue;
            }

            let value = serde_json::from_str(&line).with_context(|| {
                format!(
                    "failed to parse JSON on line {} from {}",
                    index + 1,
                    self.source_name
                )
            })?;
            values.push(value);
        }

        Ok(values)
    }
}

pub struct JsonlWriter<W> {
    writer: W,
}

impl<W> JsonlWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W> JsonlWriter<W>
where
    W: Write,
{
    pub fn write_iter<T, I>(mut self, values: I) -> Result<()>
    where
        T: Serialize,
        I: IntoIterator<Item = T>,
    {
        for value in values {
            serde_json::to_writer(&mut self.writer, &value)?;
            self.writer.write_all(b"\n")?;
        }

        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    use super::*;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Row {
        id: String,
    }

    #[test]
    fn reads_jsonl_and_skips_blank_lines() {
        let rows: Vec<Row> =
            JsonlReader::new(Cursor::new("{\"id\":\"a\"}\n\n{\"id\":\"b\"}\n"), "memory")
                .read_all()
                .unwrap();

        assert_eq!(
            rows,
            vec![
                Row {
                    id: "a".to_string()
                },
                Row {
                    id: "b".to_string()
                }
            ]
        );
    }

    #[test]
    fn writes_and_reads_jsonl_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.jsonl");
        let rows = vec![
            Row {
                id: "a".to_string(),
            },
            Row {
                id: "b".to_string(),
            },
        ];

        JsonlFile::new(&path).write_all(&rows).unwrap();
        let round_tripped: Vec<Row> = JsonlFile::new(&path).read_all().unwrap();

        assert_eq!(round_tripped, rows);
    }
}
