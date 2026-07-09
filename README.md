# traces-to-evals

`traces-to-evals` turns GenAI execution traces into reusable evaluation cases, then grades or exports those cases.

The crate is organized around four jobs:

- model traces as `Trace`, `Span`, and `EvalCase`
- extract eval cases with explicit extractor types
- grade cases with deterministic graders or an optional OpenAI judge
- export cases/results as JSONL, CSV, OpenAI eval rows, or promptfoo tests

## Current Capabilities

**Trace and eval models**

- `Trace` contains ordered `Span` records plus trace metadata.
- `SpanKind` covers common GenAI span types: `Llm`, `Agent`, `Tool`, `Chain`, `Retriever`, `Reranker`, `Embedding`, `Guardrail`, `Evaluator`, `Prompt`, and `Other`.
- `EvalCase` stores `input`, optional `actual_output`, optional `expected_output`, optional `rubric`, and metadata.

**Extractors**

- `SimpleExtractor` extracts cases from the first useful input span and the last useful output span.
- `OpenInferenceExtractor` understands OpenInference-style attributes such as `openinference.span.kind`, `input.value`, and `output.value`.

**Graders and judges**

- `NonEmptyOutputGrader`
- `ExactMatchGrader`
- `ContainsGrader`
- optional `OpenAiJudge` behind `--features llm-judge-openai`

The OpenAI judge asks only for subjective judgment fields (`score`, `criteria`, `evaluation`) using strict structured outputs. Pass/fail is computed locally from the score threshold.

**Calibration and scoring**

- `HumanRating` captures historical human labels.
- `calibrate_judge_results()` compares judge output against human ratings.
- `ScoredResult` normalizes deterministic and judge scores into a common shape.

**Export and I/O**

- `JsonlFile`, `JsonlReader`, and `JsonlWriter` handle JSONL.
- `EvalCaseCsvExporter` writes eval cases as CSV.
- `OpenAiEvalExporter` writes OpenAI eval JSONL rows.
- `PromptfooExporter` writes promptfoo-compatible test JSON.

## CLI

Run tests:

```bash
cargo test
```

Grade eval cases deterministically:

```bash
cargo run --bin traceeval -- grade \
  --cases eval_cases.jsonl \
  --grader exact-match \
  --out grade_results.jsonl
```

Other deterministic graders:

```bash
cargo run --bin traceeval -- grade \
  --cases eval_cases.jsonl \
  --grader non-empty-output \
  --out grade_results.jsonl

cargo run --bin traceeval -- grade \
  --cases eval_cases.jsonl \
  --grader contains \
  --contains "expected phrase" \
  --out grade_results.jsonl
```

Use the OpenAI judge:

```bash
cargo run --features llm-judge-openai --bin traceeval -- grade \
  --cases eval_cases.jsonl \
  --judge openai-dive \
  --model gpt-4o \
  --out judge_results.jsonl
```

Compatibility command:

```bash
cargo run --features llm-judge-openai --bin traceeval -- judge \
  --cases eval_cases.jsonl \
  --provider openai-dive \
  --model gpt-4o \
  --out judge_results.jsonl
```

The OpenAI path reads credentials from the environment, so set `OPENAI_API_KEY` before running it.

## Library Example

```rust
use traces_to_evals::extractors::SimpleExtractor;
use traces_to_evals::graders::{DeterministicGrader, ExactMatchGrader};
use traces_to_evals::io::jsonl::JsonlFile;
use traces_to_evals::{Span, Trace};

# fn main() -> anyhow::Result<()> {
let trace = Trace::new("trace-1")
    .with_span(Span::llm("input", "prompt").with_input("What is 2 + 2?"))
    .with_span(Span::llm("output", "completion").with_output("4"));

let mut case = SimpleExtractor.extract_trace(&trace)?;
case.expected_output = Some("4".to_string());

let result = ExactMatchGrader.grade(&case)?;
JsonlFile::new("grade_results.jsonl").write_all(&[result])?;
# Ok(())
# }
```

## API Shape

Prefer the struct-based APIs:

```rust
JsonlFile::new(path).read_all::<EvalCase>()?;
JsonlFile::new(path).write_all(&results)?;

OpenAiEvalExporter::write_jsonl(path, &cases)?;
PromptfooExporter::write_json(path, &cases)?;
EvalCaseCsvExporter::write(path, &cases)?;
```

`src/exporters.rs` is only a compatibility namespace that re-exports these types. It should not grow new wrapper functions.

## Planned Work

See [docs/scoring-design.md](docs/scoring-design.md) for the planned calibrated scoring, cluster-aware scoring, weighted aggregate reports, and optional ML/library stack.

Near-term implementation priorities:

- add an OpenInference import command
- add validation/report commands
- add calibration model output, not only calibration summary metrics
- add weighted aggregate run reports
