# Ordinary user Linux deployment

Use a project-owned directory, Python 3.12 control environment and separate fixed dbt environment. Install from the README. The recorded final validation used a newly created control environment, not a path into an older environment. Source services bind loopback interfaces on independent ports and use private per-rollout directories. Official PostgreSQL sources require an available user-executable PostgreSQL distribution; configure `postgres_bin` in the private runtime file. No daemon access or server-global change is needed.

Copy `examples/runtime.example.json` to `private/runtime.json` with restricted permissions. Configure authorized Snowflake runner and resource-manager identities and private key paths. The runner receives only the owned namespace role; the manager creates and removes project-owned resources. Keep these credentials and dbt profiles outside actor workspaces and archives. Configure Tinker with `elt-rl configure-tinker` or the existing private environment mechanism.

The server hosts native sources, validates loading/dbt inputs and orchestrates scoring. Tinker performs inference and gradient work; Snowflake runs official warehouse SQL. No local GPU is allocated. Run a long experiment in a user-owned tmux session so the personal computer can disconnect. Eight concurrent rollouts were used in the recorded experiments; ordinary CPU count alone does not determine a useful concurrency limit.

## Installation when Git transport is unavailable

A direct install of the pinned Cookbook Git dependency failed TLS on the validation host. An exact-commit cached wheel was used instead, with fresh dependencies and no borrowed site-packages. The source remains pinned to commit `8bf7dd6192a3f1fa64640a04ec54a7f6866bbd83`; this fallback changes distribution transport, not the Cookbook implementation.

On a machine with working access, build that dependency wheel:

```sh
python -m pip wheel --no-deps \
  'git+https://github.com/thinking-machines-lab/tinker-cookbook.git@8bf7dd6192a3f1fa64640a04ec54a7f6866bbd83' \
  --wheel-dir wheelhouse
```

Transfer the wheel and its commit provenance through an authorized channel. After creating `.venv` and installing CPU Torch as in the README, install the same dependency set without requesting a second Git clone:

```sh
mkdir -p runs/install
sed '/^tinker-cookbook /d' tools/native-linux-control.lock > runs/install/control.lock
.venv/bin/pip install wheelhouse/tinker_cookbook-0.7.2.dev1+g8bf7dd619-py3-none-any.whl \
  -c runs/install/control.lock
.venv/bin/pip install -r runs/install/control.lock
.venv/bin/pip install --no-deps -e .
```

Install the separate dbt environment with `tools/python-tools.lock`. The final validation records the actual wheel source and installation steps. Dependency wheels are not included in the source ZIP. Direct installation requires working upstream access; the fallback requires a verified same-commit wheel, not a previously trained model checkpoint.

## Resume and reconciliation

Set `resume_checkpoint` in a copy of the original experiment YAML and keep its original output directory, task, reward, model, seed plan and limits. Use the latest confirmed checkpoint. The controller checks source/input fingerprints, lineage and unresolved optimizer/save journals. Never launch two controllers for the same directory or reset counters to bypass an unknown update.

Partial evaluation recovery validates completed slots, preserves cancelled attempts and fills only missing original indices. An entirely ungraded cohort stopped by infrastructure failure may receive one explicit retry via `tools/retry_failed_training_cohort.py`; it checks cleanup and absence of optimizer dispatch, archives the attempts and preserves sampling counters. It does not sample or optimize itself. Graded groups cannot be discarded through this path.

If the optimizer and training-state save are confirmed but sampler saving fails, preserve the original journal/error, restore the exact training state with optimizer state, and finish only the missing save. Attach real receipts and an explicit recovery record. Do not repeat sampling, gradients or the optimizer. The Human Resources evidence records that checkpoint-only recovery.

Historical cars and Human Resources runs used releases 0.5.2 and 0.5.8. Their exact wheels and frozen inputs remain private. Release 0.6.1 corrects reporting; it is not substituted into old checkpoints by changing a hash or disabling integrity checks.

## Isolation and optional qualification

The actor cannot execute arbitrary shell/Python, providers or dbt macros/hooks. Host validation restricts paths, declared source capabilities, JSON/YAML/Jinja, SELECT SQL and warehouse privileges. Python environments and subprocesses package trusted executors; they are not the security sandbox.

`elt_rl.native_qualification.qualify_sources(task, output, warehouse=False)` is an optional operator API for source type/value fidelity and simultaneous source isolation. `warehouse=True` additionally uses authorized warehouse round trips. It is an engineering operation, not model training data or an automatic replay prerequisite.

Do not include keys, credentials, dbt profiles, official driver destination credentials or cloud checkpoint access addresses in release packages. The application performs no quota preflight or expense accounting and does not change billing plans. Token/context/time limits remain execution controls.
