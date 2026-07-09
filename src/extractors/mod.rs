pub mod openinference;
pub mod simple;

use anyhow::Result;

use crate::model::{EvalCase, Trace};

pub trait EvalCaseExtractor {
    fn extract_case(&self, trace: &Trace) -> Result<EvalCase>;

    fn extract_cases(&self, traces: &[Trace]) -> Result<Vec<EvalCase>> {
        traces
            .iter()
            .map(|trace| self.extract_case(trace))
            .collect()
    }
}

pub use openinference::OpenInferenceExtractor;
pub use simple::SimpleExtractor;
