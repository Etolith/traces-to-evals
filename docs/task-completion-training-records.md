# Task-completion training records

`traces-to-evals` provides the public, model-agnostic boundary between sealed
trace projections and private model training.

`TaskCompletionTrainingRecordV1::from_projection` accepts a validated
`CompactTaskCompletionProjectionV1` and produces a label-free record containing:

- immutable target, revision, context-binding, projector, and projection identities;
- the bounded goal bundle, evidence facts, and recovery chains;
- a versioned 39-value structured evidence vector; and
- content-derived feature and training-record identifiers.

The transformation deliberately excludes source names, dataset splits, rewards,
human or teacher labels, evaluator outputs, and provider judgments. Training
systems must join those values separately after verifying the training record
against its sealed projection. This prevents benchmark outcomes or source
identity from becoming accidental model inputs.

The structured features describe evidence availability and execution shape. They
do not decide whether the task succeeded. A learned evaluator remains
responsible for that decision.

Repository responsibilities remain separate:

- `traces-to-evals` owns this reusable transformation and its versioned schemas.
- Perseval owns immutable trace-revision binding, artifact verification, local
  inference, calibration loading, and evidence resolution.
- Private training systems own label joins, sampling, hard-negative mining,
  fitting, diagnostics, and export.

