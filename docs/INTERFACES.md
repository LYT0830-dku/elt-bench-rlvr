# Interface contracts

All model actions pass through the stateful environment. Source, warehouse and model implementations are host-controlled components. A new task supplies a verified specification; it does not change the RL training loop.

## Task contract

`OfficialTaskLoader.load(path)` returns a `TaskSpec`. It contains task identity, the pinned official revision, an execution manifest, source declarations, target fields and types, keys, asset digests, an evaluator declaration and required tools. `verify()` validates target correspondence, source/reference fingerprints, supported field semantics and row alignment. `public()` returns only actor-visible task material. `fingerprint()` binds private contracts to the run without sending reference values to the actor.

Each `RewardTarget` declares a table name, required `RewardField` definitions and a tuple of unique alignment keys. Fields declare kind, semantic role, nullability, domain bounds and comparator. Keys may be composite. The loader checks reference key completeness and uniqueness before execution. Source and reference assets remain separate from the model's editable project.

## Source contract

`SourceAdapter` defines `initialize(services)`, `schema(source)`, `check_connection(services, stream)`, `read(services, info, format, columns)`, `load(backend, **arguments)`, `status(services)` and `cleanup(services)`. `SOURCES.register(name, adapter)` installs a trusted implementation. Readers accept host-issued connection capabilities, not model-selected URLs, filesystem paths or credentials.

The actor writes `elt/loading.json` with the versioned loading format, assigned destination and a list of connections. Each connection selects an existing source, a supported connector/format, a destination table, named columns and declared SQL types, and append or overwrite mode. `loading validate`, `plan` and `apply` validate/register the configuration. `apply` does not load data. `sync` starts real extraction/loading; `wait_sync` returns the actual job state. Type/row/value fidelity is recorded by the executor.

The native loader accepts case-insensitive type names and normalizes `STRING` and `TEXT` to `VARCHAR`. These aliases preserve source values and warehouse types. Source text such as a formatted currency value is not silently cleaned or coerced into a number: the actor selects a compatible loading type and authors any cleaning in dbt. Cast errors identify the connection, source and column, with a bounded source-value example. An already failed `sync` returns its error immediately; `wait_sync` remains necessary to confirm successful jobs. An unobserved background infrastructure failure is raised during cleanup and excludes that episode from policy-zero samples.

## Warehouse contract

`WarehouseBackend.create(root, configuration, on_resources)` allocates an isolated warehouse session. Backends implement `load`, `query`, `output`, `physical_types`, `profile`, `transformation` and `cleanup`. Loading and transformation use the same owned namespace. The backend maps logical source names to the assigned physical schema and enforces SQL visibility. Snowflake uses separate restricted execution credentials and a host resource manager. DuckDB provides the credential-free integration backend.

Models can query only their assigned warehouse scope. They cannot submit connection strings, change roles, choose arbitrary networks or read another rollout's schemas. A completed cleanup records owned resources and remaining state.

## Model contract

`ModelAdapter.validate(...)` checks sampling, tool calling, logprob, renderer, context and LoRA capabilities. `renderer(...)` loads the actual Cookbook tokenizer and renderer. Its tool codec defines `initial_messages`, `parse_action` and `observations`. Tool declaration, parsing, result correlation, stop tokens and logprobs follow that model's codec. `validate_checkpoint` rejects incompatible model identities.

The catalog contains the two model implementations used by the recorded experiments. Add a compatible model by declaring its checked capabilities and using or implementing a trusted tool codec. Catalog registration alone does not establish empirical task-solving quality.

## Environment contract

`ELTEnvironment.initial()` returns the initial observation. `step(action)` validates one structured tool action and returns a `Step` with observation, reward and terminal status. The tools inspect task/files, edit permitted project files, manage loading, synchronize sources, execute validated dbt commands, issue scoped SELECT queries and submit. There is no arbitrary shell or Python tool.

Submission requires current loading files, completed current sync jobs and current successfully executed dbt files. It freezes the recipe and checks the actual warehouse state. Removing a model, changing a connection or performing unreproducible appends invalidates stale execution state. File digests bind the final submission to the executed pipeline.

The dbt capability subset supports table/view materialization and plain model tags in project configuration. Source declarations belong in schema YAML files under `dbt/models`. Arbitrary macros, hooks, packages, role/schema overrides and executable metadata remain unavailable. A rejected submission reports the missing operational prerequisites and failed source jobs; it does not expose target values or scoring diagnostics. Compatibility with a registered model/task is separate from whether that policy can solve the task.

`CookbookELTEnv` implements the Cookbook async environment interface; `ELTGroupBuilder` allocates independent workspaces and cleanup for each rollout. The group scheduler selects one task for all trajectories in an advantage group.

## Structured tool messages

The model sends exactly one declared function call per response. All arguments are strings, every listed field is required, and additional fields are rejected. The native codec correlates the tool response with the original call ID. Internally the environment represents an action as `{"tool": "name", ...arguments}`; the model does not insert a second `tool` argument into a function call.

| Tool | Arguments | Effect or observation |
|---|---|---|
| inspect_task | none | Public task, source schemas and assigned destination |
| list_files | none | Permitted project file paths |
| read_file | path | Permitted file content |
| write_file | path, content | Validated project file write and digest |
| edit_file | path, old, new | Unique exact text replacement |
| delete_file | path | Remove an editable project file and invalidate stale state |
| loading | command | validate, plan or apply the current loading declaration |
| sync | connection | Launch actual extraction/loading for an issued connection capability |
| wait_sync | job_id | Wait for the owned job and return actual status and row/content evidence |
| dbt | command | compile or run the validated project through the fixed dbt interpreter |
| query | sql | Validated SELECT within the assigned warehouse namespace |
| submit | none | Validate and freeze the executed recipe, grade actual warehouse state and terminate |

`ELTEnvironment.step` returns `Step(observation: dict, reward: float, done: bool)`. Intermediate actions receive reward zero and real execution feedback. A correctable action or SQL error returns an error observation so the actor can repair its project. A valid submission returns the execution-derived terminal reward. A known policy limit may terminate with zero. Infrastructure and integrity exceptions propagate to the controller and cannot become training-eligible zero outcomes.

The versioned `elt/loading.json` has exactly three top-level fields: `format`, `destination` and `connections`. Each connection declares `name`, `source`, `connector`, `format`, `table`, `columns` and `sync_mode`. Source IDs and connector/format pairs must match host-issued capabilities; destination is the assigned database/schema. `columns` maps selected source columns to supported SQL types. `sync_mode` is `full_refresh_append` or `full_refresh_overwrite`. Model-provided URLs, credentials, arbitrary paths, executable code and transformation SQL are not loading fields. The credential-free retail fixture includes an example declaration; it is separate from the official task-solving experiment.

## Evaluation and reward contracts

`EVALUATORS.get(provider).evaluate(warehouse, task, output)` returns an `EvaluationOutcome`: independent metrics, success, loading status, actual predictions and table-qualified column matches. The original provider invokes the pinned official evaluator; the local fixture provider reports local correctness without inventing official metrics.

The active reward implementation accepts a `RewardContext` and validated `RewardConfig`, and returns a finite scalar in [0, 1] plus integer correct/required cell counts, binary outcomes for each keyed target cell, and a trusted scoring receipt. Pure cell accuracy is the default and micro-averages all target cells. Its loading gate is verified execution state, independent of official SRDEL/SRDT. The recipe feeds this scalar into the unchanged Cookbook advantage calculation. Evaluation reports mean, population standard deviation, minimum, maximum and per-cell correctness; checkpoint comparisons use the same metric. Historical rewards require an explicit configuration. Saved prediction/reference CSVs, source identities and result digests allow offline scoring without model or warehouse replay.

## Training and recovery contracts

One batch collects the configured group from the current checkpoint, computes Cookbook group-relative advantages and assembles importance-sampling tensors. Observation positions receive no task advantage. A constant-reward group completes without an optimizer update. Each confirmed nonzero-gradient update produces state/sampler checkpoint receipts and lineage to its parent.

The durable protocol state distinguishes attempted batches, completed batches and effective updates. Optimizer dispatch is journaled before the API request. Unknown outcomes block automatic retry. Resume requires the latest confirmed checkpoint, matching inputs/source, matching sampling policy and resolved resources. Partial evaluation recovery preserves verified completed slots and archives cancelled attempts, then fills only missing original indices with the original checkpoint and seed plan. Evaluation trajectories are excluded from optimization.

The run-directory ownership lock prevents duplicate controllers. Token/context/time limits bound execution. These controls protect correctness and shared-server isolation.

## Model adapter and experiment configuration

`MODELS.get(model_id)` resolves the declarative catalog profile: tokenizer, supported renderer, tool codec, context limit and required sampling/training capabilities. Register a new verified profile in `model_catalog.json` and, when necessary, a trusted codec/renderer. The training loop contains no model-name dispatch. A trainable model still needs a compatible native tool format and response logprobs; an arbitrary model ID is not automatically supported.

The `qwen3_5_strings` renderer uses the pinned Cookbook native XML framing validator and preserves every declared string argument as text. Valid JSON file contents remain strings. Duplicate parameters, malformed frames, multiple calls and extra visible action text cannot execute. The adapter retains thinking history for multi-turn training; raw sampled tokens and logprobs are archived unchanged.

Framework YAML (`elt_framework_v1`) selects task specifications, model, warehouse, reward, training and evaluation settings. `evaluation.enabled` defaults to true. False skips initial, intermediate and final evaluation and is suitable for a one-update engineering example; both plan counts and interruption recovery honor it. Changing task/model/reward creates a fresh run. No historical checkpoint or quota inventory is needed to initialize a supported base model.

## Official metric aggregation

`parse_target_columns` validates table-qualified official logs; `summarize_official_targets` returns `correct_target_models`, `required_target_models`, `target_success`, fractional `official_srdt`, loading status and separate boolean whole-pipeline success. Target errors remain local to that model. `group_metrics` preserves fractional observations and uses summed model counts when provided. This reporting correction does not change official column comparison, Pure Cell Reward or optimizer inputs. Historical logs are reaggregated into separate outputs by `tools/reaggregate_official_metrics.py`.
