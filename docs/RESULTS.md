# Recorded experimental results

The final handoff includes two selected experiments, both using Pure Cell Accuracy. Recorded configuration files preserve actual settings with private paths remapped. Earlier superseded records remain in the separate private archive.

| Experiment | Model | Task | Training release | Completed batches | Rollouts | Effective updates |
|---|---|---|---|---:|---:|---:|
| Cars training | thinkingmachines/Inkling-2-Mini | cars | 0.5.2 | 16 | 128 | 15 |
| Cross-model/task verification | Qwen/Qwen3.6-35B-A3B | human_resources | 0.5.8 | 1 | 8 | 1 |

Both used group size 8, LR 1e-5, LoRA rank 16, temperature 0.8, Cookbook centered group advantages, importance_sampling, Adam and one optimization substep. Complete update receipts and lineage are in `evidence/cars/` and `evidence/human_resources/`. No experiment was added during release preparation.

## Cars training and independent evaluation

The run completed 16 attempted/completed batches. Batch 13 had identical rewards and was skipped; the other 15 applied effective updates. It selected 128 training, 50 process-evaluation and 40 final-evaluation rollouts. Eight extra ungraded infrastructure attempts are archived separately and excluded from rewards/gradients. The fixed final policy is checkpoint 15, the last effective update.

| Final metric | Initial policy, n=20 | Last effective policy, n=20 |
|---|---:|---:|
| Mean cell accuracy | 0.557143 | 0.573810 |
| Population standard deviation | 0.072375 | 0.104843 |
| Minimum | 0.523810 | 0.523810 |
| Maximum | 0.857143 | 0.904762 |
| Official SRDEL | 1.0 | 1.0 |
| Official SRDT | 0.0 | 0.0 |
| End-to-end pipeline success | 0.0 | 0.0 |

The observed +1.67 percentage points is not evidence of a statistically significant improvement. No complete pipeline was solved in either final cohort. The final policies use matching tasks, prompts, tools, temperature, seed plans and execution limits; the SDK does not guarantee strict determinism. Initial process evaluation (n=10) is not substituted for the independent initial-policy final cohort (n=20).

Actual limits were 200 turns, 64000 context tokens, 49152 maximum response tokens and 262144 generated tokens per episode. The run took approximately 3 hours 14 minutes including interruption/recovery. All rewards and advantages are in `evidence/cars/rollouts.csv` and `training-batches.json`. Curves include all batches and evaluation cohorts, including decreases.

## Human Resources compatibility verification

Actual model: **Qwen/Qwen3.6-35B-A3B**, using the string-preserving native XML renderer. The task has two target models and 96 required cells. This was one batch of eight rollouts with 80 turns, 64000 context tokens, 16384 maximum response tokens and 262144 generated tokens per episode. Evaluation was explicitly disabled; configuration fields for disabled evaluations are not executed samples.

| Rollout index | Correct / required cells | Reward | Advantage |
|---:|---:|---:|---:|
| 0 | 0/96 | 0 | -0.417969 |
| 1 | 0/96 | 0 | -0.417969 |
| 2 | 0/96 | 0 | -0.417969 |
| 3 | 86/96 | 0.895833 | 0.477865 |
| 4 | 89/96 | 0.927083 | 0.509115 |
| 5 | 0/96 | 0 | -0.417969 |
| 6 | 57/96 | 0.593750 | 0.175781 |
| 7 | 89/96 | 0.927083 | 0.509115 |

Five episodes submitted actual scoring results; indices 0, 2 and 5 reached the turn limit. The batch mean was 0.417969. A confirmed effective update used 258426 training input tokens, 97955 nonzero-advantage tokens and 10 training datums. Gradient norm was 2977.919189 and loss sum 765.677002. Training-state and sampler checkpoints were saved. A checkpoint-only recovery restored optimizer state and saved the missing sampler without repeating the optimizer.

Official SRDEL was 4/8 (50%). After correction, SRDT is 2/16 target models (12.5%): `positions` passed for indices 4 and 7; `locations` did not. Whole-pipeline success remains 0%. No independent initial/final evaluation was run. This demonstrates execution and a real update across a different model and task, not improved policy quality or universal task/model support. Its different limits also preclude a controlled model-quality comparison with Cars.

Preparatory cancelled attempts and a prior native XML conversion failure remain privately archived. The host adapter had coerced JSON file text into an object; the fixed string renderer and its regression tests preserve declared text while keeping strict framing. Those invalid preparation attempts are not relabeled as successful model samples.

## Statistical correction and provenance

Original reports are immutable. Release 0.6.1 separately reaggregated pinned Stage 2 logs for all 218 selected cars and eight Human Resources episodes. Cars had no numerical change. Two Human Resources episode metrics changed from false/0 to 0.5; the cohort changed from 0 to 0.125. Reward, advantages, optimizer receipts, checkpoints and complete-pipeline success did not change. See `evidence/metric-corrections/`.

Actual costs were not measured because expense accounting was disabled. Token counters are execution evidence, not dollar charges. The public evidence is allowlisted and sanitized; full raw records, original paths and cloud checkpoint identities remain in the private archive. Learned cloud weights are not bundled as local binaries.
