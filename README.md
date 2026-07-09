# traces-to-evals

`traces-to-evals` turns GenAI execution traces into reusable evaluation cases, then evaluates, calibrates, aggregates, or exports those cases.

The crate is organized around a composable evaluation pipeline:

- model traces as `Trace`, `Span`, and `EvalCase`
- extract eval cases with explicit extractor types
- evaluate cases with deterministic graders or an optional OpenAI judge
- normalize all evaluator output into `EvaluationResult`
- collect cases and results in `EvaluationRun`
- calibrate historical evaluator scores with `CalibrationModel`
- aggregate run quality with `WeightedAggregate`
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

Deterministic graders implement `Evaluator`. LLM judges implement `AsyncEvaluator`. Both produce the same `EvaluationResult` shape.

The OpenAI judge asks only for subjective judgment fields (`score`, `criteria`, `evaluation`) using strict structured outputs. Pass/fail is computed locally from the score threshold.

**Evaluation, calibration, and scoring**

- `EvaluationResult` is the canonical output for deterministic graders, LLM judges, and future ML scorers.
- `EvaluationRun` composes cases, multiple evaluator passes, and aggregate scoring.
- `WeightedAggregate` computes weighted run scores by evaluator and cluster.
- `HumanRating` captures historical human labels.
- `CalibrationModel::fit()` learns score bins from previous `EvaluationResult` datasets and human ratings.

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

Extract eval cases from OpenInference-style traces:

```bash
cargo run --bin traceeval -- extract \
  --format openinference \
  --traces fixtures/openinference/traces.jsonl \
  --out eval_cases.jsonl
```

Validate cases or evaluation results:

```bash
cargo run --bin traceeval -- validate \
  --profile runnable-cases \
  --cases eval_cases.jsonl \
  --out validation.json
```

Evaluate cases deterministically:

```bash
cargo run --bin traceeval -- grade \
  --cases eval_cases.jsonl \
  --grader exact-match \
  --out evaluation_results.jsonl
```

Other deterministic graders:

```bash
cargo run --bin traceeval -- grade \
  --cases eval_cases.jsonl \
  --grader non-empty-output \
  --out evaluation_results.jsonl

cargo run --bin traceeval -- grade \
  --cases eval_cases.jsonl \
  --grader contains \
  --contains "expected phrase" \
  --out evaluation_results.jsonl
```

Use the OpenAI judge:

```bash
cargo run --features llm-judge-openai --bin traceeval -- grade \
  --cases eval_cases.jsonl \
  --judge openai-dive \
  --model gpt-4o \
  --out evaluation_results.jsonl
```

Compatibility command:

```bash
cargo run --features llm-judge-openai --bin traceeval -- judge \
  --cases eval_cases.jsonl \
  --provider openai-dive \
  --model gpt-4o \
  --out evaluation_results.jsonl
```

The OpenAI path reads credentials from the environment, so set `OPENAI_API_KEY` before running it.

Fit calibration from historical evaluation results:

```bash
cargo run --bin traceeval -- calibrate \
  --human-ratings fixtures/eval/human_ratings.jsonl \
  --results fixtures/eval/historical_results.jsonl \
  --out calibration.json
```

Assign known clusters and annotate results:

```bash
cargo run --bin traceeval -- cluster assign \
  --cases eval_cases.jsonl \
  --clusters fixtures/eval/clusters.jsonl \
  --out cluster_assignments.jsonl \
  --results evaluation_results.jsonl \
  --results-out clustered_results.jsonl
```

Add a custom assignment rule in Rust:

```rust
use traces_to_evals::prelude::*;

let clusters = vec![EvalCluster {
    id: "billing".to_string(),
    label: "Billing".to_string(),
    description: None,
    weight: 1.0,
    metadata: Default::default(),
}];

let assigner = RuleBasedClusterAssigner::new(clusters).with_rule(
    FnClusterAssignmentRule::new("billing_keyword", |case, _clusters| {
        if case.input.to_ascii_lowercase().contains("invoice") {
            Some(ClusterRuleMatch::new("billing", 0.9))
        } else {
            None
        }
    }),
);
```

Build an aggregate report:

```bash
cargo run --bin traceeval -- report \
  --results clustered_results.jsonl \
  --calibration calibration.json \
  --clusters fixtures/eval/clusters.jsonl \
  --out report.json
```

## Library Example

```rust
use traces_to_evals::prelude::*;
use traces_to_evals::io::jsonl::JsonlFile;

# fn main() -> anyhow::Result<()> {
let cases = vec![
    EvalCase::new("case-1", "trace-1", "What is 2 + 2?")
        .with_actual_output("4")
        .with_expected_output("4"),
];

let run = EvaluationRun::new(cases)
    .evaluate_with(&NonEmptyOutputGrader)?
    .evaluate_with(&ExactMatchGrader)?;

let score = run.aggregate();
assert_eq!(score.weighted_score, 1.0);

JsonlFile::new("evaluation_results.jsonl").write_all(run.results())?;
# Ok(())
# }
```

Async judges compose through the same run type:

```rust
# #[cfg(feature = "llm-judge-openai")]
# async fn example() -> anyhow::Result<()> {
use traces_to_evals::prelude::*;
use traces_to_evals::judge::openai::OpenAiJudge;

let cases = vec![
    EvalCase::new("case-1", "trace-1", "Summarize the trace")
        .with_actual_output("The agent retrieved context and answered.")
        .with_rubric("Reward concise, faithful summaries."),
];

let judge = OpenAiJudge::from_env("gpt-4o");
let run = EvaluationRun::new(cases)
    .evaluate_with_async(&judge)
    .await?;
# Ok(())
# }
```

Calibration composes with the same result type:

```rust
use traces_to_evals::calibration::{CalibrationModel, HumanRating};
use traces_to_evals::prelude::*;

# fn main() -> anyhow::Result<()> {
let historical_results = vec![
    EvaluationResult::from_ids(
        "case-1",
        "trace-1",
        "openai/gpt-4o",
        3.0,
        ScoreScale::FourPoint,
        true,
        "mostly correct",
    ),
];
let new_run = EvaluationRun::new(Vec::new()).add_results(historical_results.clone());
let human_ratings = vec![
    HumanRating {
        case_id: "case-1".to_string(),
        trace_id: "trace-1".to_string(),
        score: 4,
        passed: None,
        notes: None,
    },
];

let model = CalibrationModel::fit(&human_ratings, &historical_results, 3)?;
let calibrated_run = model.apply_run(new_run);
# Ok(())
# }
```

## API Shape

Prefer the struct-based APIs:

```rust
JsonlFile::new(path).read_all::<EvalCase>()?;
JsonlFile::new(path).write_all(run.results())?;

EvaluationRun::new(cases)
    .evaluate_with(&ExactMatchGrader)?
    .aggregate_with(&WeightedAggregate::default());

OpenAiEvalExporter::write_jsonl(path, &cases)?;
PromptfooExporter::write_json(path, &cases)?;
EvalCaseCsvExporter::write(path, &cases)?;
```

Library APIs return `traces_to_evals::Result<T>` with `TraceEvalError` variants for common failures such as missing outputs, invalid scores, validation failures, and calibration overlap.

## Planned Work

See [docs/api-and-product-roadmap.md](docs/api-and-product-roadmap.md) for API/product cleanup, [docs/missing.md](docs/missing.md) for remaining work, [docs/scoring-design.md](docs/scoring-design.md) for scoring and calibration design, and [docs/cluster-discovery.md](docs/cluster-discovery.md) for the full cluster discovery, embedding, and LLM labeling spec.

Near-term implementation priorities:

- add calibration mismatch warnings
- revisit public API visibility
- add Markdown report output
- add real cluster discovery and LLM cluster labeling
