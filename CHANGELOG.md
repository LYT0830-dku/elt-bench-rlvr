# Release notes

## 0.6.1

Correct official SRDT to count fully correct target data models, preserving fractional values independently of whole-pipeline success. Keep all declared models in the denominator after loading failure, handle per-target errors, and micro-average model counts where available. The pinned evaluator, Pure Cell Reward and Cookbook optimizer recipe remain unchanged.

Add multi-target SRDT and offline-reaggregation regressions, an external-reference reward-verification option, selected execution evidence, and final documentation. Existing records are immutable: cars statistics are unchanged; two Human Resources episode SRDT values become 0.5 and the cohort becomes 0.125. The superseded first experiment is excluded from the handoff; only cars training and Human Resources compatibility validation are included. No new paid training was run.

## 0.6.0

Final native release with pure cell accuracy as the default training and evaluation reward. Historical verified rewards remain explicitly selectable. Native loading, scoped dbt/Snowflake execution, official evaluation, Cookbook importance-sampling training, resumable checkpoints and partial evaluation recovery are retained.

The Qwen XML renderer preserves declared string arguments, including JSON documents, while retaining strict framing and single-tool execution. An explicit evaluation opt-out supports a small engineering verification and remains disabled after interruption recovery. The plan command reports only enabled rollouts. Fresh supported models do not require previous checkpoints. Quota administration and expense accounting are absent.

Removed unused legacy plotting/helpers and the inactive Terraform submission fallback. Tests for the explicitly retired quota subsystem were archived privately; all active unit, integration, security and interface tests and GitHub Actions CI remain. Raw experimental history is privately archived, with sanitized evidence and actual validation results in the release.

Recorded experiments retain their original identities: cars used 0.5.2 for 16 batches, 15 effective updates and fixed final evaluation; human_resources with Qwen3.6-35B-A3B used 0.5.8 for eight rollouts and one effective update. The final cleanup does not relabel historical experiments or claim arbitrary task/model support.
