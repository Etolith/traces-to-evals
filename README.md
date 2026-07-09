# traces-to-evals

Rust utilities for turning execution traces into evaluation cases, running deterministic graders, exporting eval data, and optionally judging cases with OpenAI Chat Completions through `openai_dive`.

The core crate has no LLM dependency by default:

```bash
cargo test
```

Enable the OpenAI judge adapter only when needed:

```bash
cargo run --features llm-judge-openai --bin traceeval -- judge \
  --cases eval_cases.jsonl \
  --provider openai-dive \
  --model gpt-4o \
  --out judge_results.jsonl
```

The judge command reads `OPENAI_API_KEY` through `openai_dive::Client::new_from_env()`. It asks the model only for the subjective payload (`score`, `criteria`, and `evaluation`) using strict JSON Schema structured outputs. Operational pass/fail is computed locally from `score >= pass_threshold`.

Human labels can be represented as `HumanRating` values and compared with `JudgeResult` values through `calibrate_judge_results()` to report exact-match rate, within-one rate, pass agreement, and mean absolute error.
