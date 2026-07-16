# Contributing

Thank you for helping improve `traces-to-evals`.

Before opening a pull request, run:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features --locked
```

Keep finding identities deterministic, keep raw payloads out of findings, and
add regression coverage for changes to normalizers, detectors, grouping,
comparison, or candidate generation.
