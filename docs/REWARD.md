# Pure cell accuracy

The default training and evaluation metric is:

```text
reward = correct_required_cells / all_official_target_cells
```

Every required reference cell has equal weight, including a primary-key field when it is a required target column. Each cell earns either 1 or 0. Across multiple tables, sum the integer correct-cell counts and divide by the sum of their required-cell counts. This is a micro average: a larger table contributes proportionally more target cells. No dataset name, fixed column count or fixed cell count enters the implementation.

Rows align through the task’s verified keys, including composite keys. Reordering rows does not change the score. Missing rows, columns, tables and incorrect cells keep all corresponding reference cells in the denominator and earn no credit for those cells. A partially missing table retains credit for its other correct rows. Null, duplicate, malformed or extra prediction keys invalidate that target; independently valid targets retain their cell credit. Extra rows never increase reward. Broken reference schemas or nonunique reference keys are infrastructure errors.

Numeric correctness uses each field’s verified comparator and tolerance. A value within tolerance earns 1; one outside earns 0 regardless of distance. Identifiers, categories, text, dates and booleans follow their declared exact rules. Illegal types, nonfinite values, out-of-domain values and nonnullable nulls are incorrect. Explicitly nullable fields follow the task’s declared null rule. Task-specific comparator behavior is recorded in the specification; the official evaluator is unchanged.

Submission validity and actual loading execution remain mandatory checks. The host supplies verified operational loading state after the current loading configuration and syncs pass submission checks. This gate is separate from official SRDEL. Neither official SRDEL, SRDT nor complete-column matching enters the pure-cell numerator or denominator. Infrastructure outages, unknown outcomes and broken trusted data are excluded from model-zero samples.

## Configuration

```yaml
reward:
  type: pure_cell
  version: pure_cell_accuracy_v1
```

Omitting the reward section selects this same default. Mixture weights and proximity parameters are rejected. A run binds reward configuration, source, task, prompt and tools to checkpoint lineage. A different reward requires a new run from original base weights; it cannot be introduced through resume.

## Training and evaluation

The scalar enters the unchanged Tinker Cookbook group-relative advantage calculation and importance-sampling training data. Group size, Adam, learning rate, LoRA rank, sampling and update policy remain independent of this change. Equal-reward groups complete without an ineffective optimizer update.

Intermediate and final evaluation use the same cell correctness rules. Every evaluation reports mean, population standard deviation, minimum, maximum and each keyed target cell’s correctness rate across all rollouts. A known invalid policy termination contributes zero for every required cell. Infrastructure or cancelled attempts do not enter evaluation denominators.

Trusted results contain reward, cell_accuracy, integer correct_cells and required_cells, binary cell_results, configuration identity and actual prediction/reference snapshots. cell_score is a compatibility alias for the same scalar. Private reference values and correctness masks stay outside actor tools.

## Explicit historical baselines

| Type | Historical formula | Version |
|---|---|---|
| column_execution | Complete-column fraction | hierarchical_execution_v1 |
| column_cell_execution | 0.5 column + 0.5 cell | hierarchical_execution_v1 |
| hierarchical_execution | 0.5 column + 0.3 cell + 0.2 value | hierarchical_execution_v1 |
| execution | The weighted implementation from release 0.4 | execution_reward_v1 |

Historical implementations retain their macro averaging, missing-key invalidation, execution gates and numeric proximity behavior. They require explicit selection. Original scores and advantages remain unchanged. Offline pure-cell rescoring is written separately with its own metric identity, coverage and source digests. It never changes historical optimizer evidence or claims old policies trained with the new reward. An absent actual prediction snapshot is marked unavailable, never fabricated or paid-replayed.
