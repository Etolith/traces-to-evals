# traces-to-evals

`traces-to-evals` turns GenAI execution traces into reusable evaluation cases, then evaluates, calibrates, aggregates, or exports those cases.

Licensed under GPL-3.0-or-later. See [`LICENSE`](LICENSE). Security reports and
contribution checks are documented in [`SECURITY.md`](SECURITY.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).

The repository binary is currently named `traceeval`; downstream projects can
rename the binary or wrap it without changing parser code. Persisted cluster
artifacts also support a custom namespace through `ProjectName`, so generated
schema versions do not have to use the default `traceeval.*` prefix.

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

**Agent behavior findings**

- `OpenInferenceBehaviorNormalizer` converts trace spans into versioned `AgentBehaviorTrace` records with bounded tool, approval, state-change, policy, and final-outcome facts.
- `DeterministicDetectorSet` detects terminal and repeated tool failures, call loops, uncertain mutations, false success claims, approval bypasses, policy violations, excessive tool usage, unresolved escalations, and missing resolutions.
- Recovery analysis suppresses a terminal failure when an idempotent retry, verified state, safe escalation, or accurate failure response resolves it.
- Operation effect, retry safety, requiredness, and final claims come from structured telemetry or a versioned `BehaviorAdapterConfig`; the kernel does not infer them from tool names or assistant prose.
- Finding IDs and failure signatures are deterministic SHA-256 identities. Error messages are represented by hashes rather than copied into findings.
- `FindingEvalCandidateGenerator` creates reviewable `candidate` records; it never promotes generated behavior into an accepted eval suite.
- `DetectionRunner` processes one trace at a time through async source/sink traits, skips completed finding IDs, and checkpoints only after durable finding writes and event side effects complete.
- `SemanticBehaviorDetector` optionally evaluates a bounded behavior projection with a model, validates every cited evidence key, and merges failed or abstained judgments into the same finding/evidence/candidate path.
- Structured-only judgments, failures below the configured confidence threshold, and model abstentions are informational `semantic_review_required` findings rather than actionable defects. A specific semantic failure kind and severity requires explicitly pre-redacted summaries and the confidence threshold. All semantic findings retain `requires_human_review=true`.

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

Normalize agent behavior and detect deterministic findings:

```bash
cargo run --bin traceeval -- detect \
  --format openinference \
  --traces fixtures/behavior/traces.jsonl \
  --adapter-config fixtures/behavior/adapter.json \
  --normalized-out agent_behavior.jsonl \
  --out behavior_findings.jsonl \
  --candidates-out eval_candidates.jsonl \
  --evidence-packet-out evidence_packet.json \
  --projections-out finding_projections.jsonl \
  --projection-cases-out finding_projection_cases.jsonl \
  --signature-groups-out signature_groups.jsonl \
  --projection-metadata-key fixture
```

Add evidence-grounded semantic judging to the same detection run:

```bash
cargo run --features llm-judge-openai --bin traceeval -- detect \
  --format openinference \
  --traces fixtures/behavior/traces.jsonl \
  --adapter-config fixtures/behavior/adapter.json \
  --semantic-judge openai-dive \
  --semantic-model <openai-chat-model> \
  --semantic-results-out semantic_results.jsonl \
  --semantic-projections-out semantic_projections.jsonl \
  --out combined_findings.jsonl
```

The default `--semantic-content structured-only` projection excludes user and
assistant summaries, arbitrary trace metadata, raw tool arguments/results,
error messages, and state payloads. It contains only bounded tool, approval,
policy, state-observation, and final-outcome facts. Invalid free-form semantic
labels are replaced by hashes. To evaluate response quality, callers may use
`--semantic-content pre-redacted-summaries`, but only after redacting those
summaries upstream; the flag is an explicit attestation, not a redactor.

The semantic evaluator returns `pass`, `fail`, or `abstain`, a 1-4 score,
confidence, criteria, and evidence keys. Failed judgments must cite keys that
exist in the projection. The OpenAI adapter downgrades unsupported citations to
a zero-confidence review-only abstention; custom evaluators fail validation.
Only thresholded failures evaluated with explicitly pre-redacted summaries
retain the model-proposed bounded failure kind and severity. Structured-only
judgments, low-confidence failures, and abstentions become informational review
findings by default; use
`--semantic-ignore-abstentions` to omit abstention findings. Model explanations
are stored in the optional evaluation-results artifact, while immutable
findings keep only its content hash and structured provenance.

Use `--semantic-rubric-file` with `--semantic-rubric-version` for a versioned
application rubric and `--semantic-min-confidence` to configure the actionable
failure threshold. Semantic result rows contain an evaluator specification hash
and can be passed to paired remediation verification alongside deterministic
grader results after the relevant review policy accepts them.

The local CLI accepts threshold overrides such as `--max-repeated-failures`,
`--max-equivalent-calls`, `--max-tool-calls`, and
`--max-total-tool-duration-ms`. Candidate output remains unreviewed and keeps
source finding/evidence provenance. Cross-trace importance scoring, issue
aggregation, storage, sandboxes, deployments, and approval policy remain
platform responsibilities.

Semantic projections include only fixed finding fields and explicitly
allowlisted metadata. They are deterministically ordered, size-bounded,
versioned, hashed, and always exclude common tenant, deployment, revision, and
credential fields. Library callers can supply a `FindingRedactor` before any
projected value is sent to an embedding provider.
`--projection-cases-out` wraps only the safe projected text as `EvalCase`
records, allowing the existing `cluster embed`, `cluster discover`, and
assignment paths to perform semantic grouping without receiving raw traces.

`KnownSignatureGrouper` provides deterministic local grouping by the existing
mechanical `failure_signature` and deduplicates repeated delivery of the same
`finding_id`. It does not assign business impact, materialize issues, or manage
issue lifecycle state.

Evidence packets contain sorted scoped references, detector/adapter versions,
and explicit telemetry gaps, with deterministic packet and content hashes.
Generated candidates link to that packet, hash their proposed definition, and
must pass the typed review transition before becoming accepted; changing the
proposed expected behavior invalidates the definition hash.

Candidate generation does not copy the normalized trace input by default,
because a bounded input summary is not automatically a redacted fixture.
Library callers may explicitly attach a `RedactedCandidateInput` containing a
synthetic or redacted summary, its redaction-policy version, and an evidence
reference. That input and its provenance are included in the candidate
definition hash.

The normalizer uses explicit structured evidence when present. Supported
adapter attributes include `agent.tool.status`, `agent.operation`,
`agent.operation.effect`, `agent.operation.retry_safety`,
`agent.tool.requirement`,
`agent.approval.required`, `agent.approval.outcome`,
`agent.state.predicate`, `agent.state.observation`,
`agent.policy.outcome`, `agent.final.status`, `agent.outcome.claims`, and
`agent.escalation.status`. It also reads OpenInference and OpenTelemetry GenAI
names such as `openinference.span.kind`, `gen_ai.tool.name`, and
`gen_ai.tool.call.id`. A bounded `state_delta` becomes a referenced artifact;
its private payload is not copied into findings and does not imply verified
success.

Verify that a paired candidate run removes an incident signature and introduces
no severe novel finding:

```bash
cargo run --bin traceeval -- verify-findings \
  --case-id incident-case \
  --baseline baseline-findings.jsonl \
  --candidate candidate-findings.jsonl \
  --target-signature sha256:... \
  --out finding-verification.json
```

This is the deterministic finding gate only. To combine every offline Stage 10
gate, create a versioned request such as:

```json
{
  "schema_version": "traceeval.remediation_verification_request.v1",
  "case_id": "incident-case",
  "target_failure_signatures": ["sha256:..."],
  "incident_case_id": "incident-case",
  "suite_case_ids": ["accepted-case-1"],
  "severe_threshold": "high",
  "policy_gate": {
    "status": "passed",
    "evidence": [{"kind": "policy_report", "identity": "sha256:..."}]
  },
  "approval_gate": {
    "status": "passed",
    "evidence": [{"kind": "approval_record", "identity": "approval:..."}]
  },
  "baseline_budget": {
    "tool_call_count": 4,
    "latency_ms": 1200,
    "cost_microunits": 300
  },
  "candidate_budget": {
    "tool_call_count": 4,
    "latency_ms": 1150,
    "cost_microunits": 290
  },
  "policy": {
    "max_new_suite_failures": 0,
    "max_suite_score_drop": 0.0,
    "max_tool_call_increase": 0,
    "max_latency_increase_ms": 0,
    "max_cost_increase_microunits": 0
  }
}
```

Then run:

```bash
cargo run --bin traceeval -- verify-remediation \
  --request remediation-request.json \
  --baseline-findings baseline-findings.jsonl \
  --candidate-findings candidate-findings.jsonl \
  --baseline-results baseline-results.jsonl \
  --candidate-results candidate-results.jsonl \
  --out remediation-verification.json
```

The combined verifier requires exact case/evaluator/version pairing, a result
for every declared suite case, valid non-duplicate result records, evidence-backed
policy and approval gates, bounded score regressions, and bounded tool-call,
latency, and cost increases. Evaluation rows must include either
`metadata.evaluator_spec_hash` or `metadata.evaluator_version`. Changed evaluator
versions are reported as an unpaired baseline result plus an unexpected
candidate result; aggregate pass rate cannot conceal a per-case regression.
The CLI binds the report and verification ID to SHA-256 digests plus byte and
record counts for all four findings/results files. A request may predeclare
`input_artifacts`; if it does, the command rejects any digest mismatch.
The report is written even when a gate fails, and the command then exits
non-zero.

Passing offline remediation verification authorizes no rollout and does not
close an issue. Canary deployment, production proof, promotion, rollback, and
issue lifecycle remain platform responsibilities.

`FindingRecurrenceComparator` compares affected-trace rates for baseline and
canary windows, reports recurrence deltas/ratios and severe novel findings, and
records whether each window is exact or sampled. Sampled reports explicitly
forbid interpreting observed rates as exact customer prevalence; Karura retains
promotion and rollback policy.

The same comparison is available to a rollout controller through the CLI:

```json
{
  "schema_version": "traceeval.finding_recurrence_request.v1",
  "target_failure_signatures": ["sha256:..."],
  "baseline_window": {
    "window_id": "stable-2026-07-10T10:00Z",
    "observed_trace_count": 1000,
    "population_basis": "sampled"
  },
  "candidate_window": {
    "window_id": "canary-2026-07-10T10:00Z",
    "observed_trace_count": 120,
    "population_basis": "sampled"
  },
  "severe_threshold": "high"
}
```

```bash
cargo run --bin traceeval -- compare-recurrence \
  --request recurrence-request.json \
  --baseline-findings stable-findings.jsonl \
  --candidate-findings canary-findings.jsonl \
  --out recurrence-comparison.json
```

The comparison rejects conflicting duplicate finding records and windows with
more affected traces than observed traces. Its content identity includes the
window counts, population basis, severity threshold, targets, and finding IDs.
It also reports affected-trace rates for every finding kind, so rollout policy
can separately inspect tool failures, false-success findings, and policy
violations. Zero-observation or unknown-population windows produce explicit
evidence gaps and an instruction to pause; the comparator never emits a
promotion or rollback decision.

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

Rule-based assignment reads cluster-oriented metadata keys by default:
`cluster_id`, `cluster`, `task_cluster`, and the `tags` array. Application
metadata has no built-in meaning. Select an additional scalar field explicitly
with the repeatable `--metadata-key` option:

```bash
cargo run --bin traceeval -- cluster assign \
  --cases eval_cases.jsonl \
  --clusters fixtures/eval/clusters.jsonl \
  --metadata-key route \
  --metadata-key product_area \
  --out cluster_assignments.jsonl
```

All extractors preserve arbitrary trace metadata. The OpenInference extractor
also preserves arbitrary root-span attributes, so callers can use their own
fields without adding application concepts to the crate.

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

Behavior normalization and detection also compose as library APIs:

```rust
use traces_to_evals::prelude::*;

# fn example(trace: &Trace) -> traces_to_evals::Result<Vec<BehaviorFinding>> {
let behavior = OpenInferenceBehaviorNormalizer::default().normalize(trace)?;
let findings = DeterministicDetectorSet::default().detect(&behavior);
let candidates = FindingEvalCandidateGenerator.generate_all(
    std::slice::from_ref(&behavior),
    &findings,
);

assert!(candidates
    .iter()
    .all(|candidate| candidate.status == EvalCandidateStatus::Candidate));
# Ok(findings)
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

See [docs/api-and-product-roadmap.md](docs/api-and-product-roadmap.md) for API/product cleanup, [docs/missing.md](docs/missing.md) for remaining work, [docs/scoring-design.md](docs/scoring-design.md) for scoring and calibration design, [docs/cluster-discovery.md](docs/cluster-discovery.md) for the cluster discovery, embedding, and LLM labeling spec, and [docs/vector-index.md](docs/vector-index.md) for the proposed vector index trait and Paimon backend.

Near-term implementation priorities:

- add calibration mismatch warnings
- revisit public API visibility
- add Markdown report output
- add local embeddings and non-K-Means discovery options
