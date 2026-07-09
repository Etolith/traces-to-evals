use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};

#[derive(Debug, Clone)]
pub struct JsonFile<P> {
    path: P,
}

impl<P> JsonFile<P> {
    pub fn new(path: P) -> Self {
        Self { path }
    }
}

impl<P> JsonFile<P>
where
    P: AsRef<Path>,
{
    pub fn read<T>(&self) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let file = File::open(self.path.as_ref())
            .with_context(|| format!("failed to open {}", self.path.as_ref().display()))?;

        serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("failed to parse {}", self.path.as_ref().display()))
    }

    pub fn write_pretty<T>(&self, value: &T) -> Result<()>
    where
        T: Serialize,
    {
        let file = File::create(self.path.as_ref())
            .with_context(|| format!("failed to create {}", self.path.as_ref().display()))?;

        serde_json::to_writer_pretty(BufWriter::new(file), value)
            .with_context(|| format!("failed to write {}", self.path.as_ref().display()))
    }
}
