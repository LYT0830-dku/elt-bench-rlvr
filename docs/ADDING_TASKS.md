# Add an ELT task

A task is supported when its declared assets, source protocols, warehouse, target types, row alignment and evaluator are compatible with the implemented interfaces. Validate each new task before reporting execution support.

1. Pin the official task revision and obtain the actual source assets, target schema and reference data. Record hashes and provenance.
2. Create an execution manifest and `TaskSpec`. Declare every source, required target, field type, semantic role and verified key. Keep reference paths outside the actor workspace.
3. Select existing source adapters and a warehouse backend. Implement a new trusted adapter only when a required protocol or format is missing. Unsupported sources fail explicitly.
4. Select the matching evaluator and validate its metrics and target identities. For the official provider, preserve the original data, evaluator and success definitions.
5. Run a local interface/integration check, then one authorized real task execution. Confirm loading fidelity, warehouse ownership, submission consistency, reward and cleanup.
6. Select the specification in an experiment YAML. Use the same `elt-rl train` entry point and unchanged Cookbook optimizer loop.

Existing source implementations include PostgreSQL, HTTP, HTTPS, S3 and host-declared local files. MongoDB and undeclared complex formats are unsupported until an appropriate adapter is implemented and validated. Tasks without reliable keys do not receive guessed row matching. The reference scoring limit is two million rows per target; larger tasks need a deliberate streaming implementation.

The verified-task evidence lists each actual task/model execution separately. A successful additional task demonstrates the extension path for its source/target combination. It does not establish that every benchmark task has complete assets, supported protocols or a qualified evaluator.
