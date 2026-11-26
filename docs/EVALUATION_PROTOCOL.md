# Evaluation and official metric definitions

Default reward: `pure_cell_accuracy_v1`, equal-weight binary correctness over all required target cells, micro-averaged across tables. Verified keys align rows and missing cells stay in the denominator. Official success metrics never enter this training scalar.

| Metric | Numerator | Denominator |
|---|---|---|
| SRDEL | Pipelines that pass official extraction/loading checks | All selected pipelines |
| SRDT | Target data models whose required columns pass official comparison | All declared target data models in selected pipelines |
| End-to-end success | Pipelines passing loading and every target model | All selected pipelines |
| Pure cell accuracy | Correct required cells under the task comparator | All official required cells |

These distinctions follow [ELT-Bench Section 3.1](https://arxiv.org/html/2504.04808v1#S3.SS1). The [pinned Stage 2 driver](https://github.com/uiuc-kang-lab/ELT-Bench/blob/e2e377766604f61930165f20d4fcab8400a3e997/evaluation/eva_stage2.py) emits per-target matched/unmatched/missing columns and runs only after Stage 1 succeeds. A loading failure contributes no successful targets while keeping all declared targets in the SRDT denominator. One fully correct table out of two gives SRDT 0.5 and end-to-end false. A task error for one target does not erase another successful target; malformed or incomplete trusted logs remain infrastructure errors.

Release 0.6.1 preserves the unmodified official comparison code and fixes only its surrounding statistic aggregation. Fractional SRDT is never cast to bool. Where target-model counts are available, cohort SRDT uses summed correct/required model counts, including unequal target counts. Legacy fixed-task cohorts retain their fractional scalar observations. Regressions cover SRDT 0, 0.5 and 1, loading failure, a target error and count-weighted aggregation.

## Formal Cars protocol

Pinned official revision: `e2e377766604f61930165f20d4fcab8400a3e997`. Original Inkling-2-Mini base weights, 16 batches of eight, importance_sampling, Cookbook centered advantages, Adam, LR 1e-5, rank 16 and one substep. Temperature 0.8, top_p 1, top_k -1 and concurrency 8. Limits: 200 turns, 64000 context tokens, 49152 response tokens and 262144 generated tokens per episode.

Process evaluation uses ten rollouts before training and after batches 4, 8, 12 and 16. Independent final evaluation uses 20 initial-policy and 20 last-effective-policy rollouts with root seed 200000000. The model selection rule is fixed to the last effective update, not the best intermediate result. Tasks, prompts, tools, temperature, limits and evaluator match. Supplying SDK seeds does not establish deterministic provider behavior.

Each cohort records mean cell accuracy, population standard deviation, minimum, maximum and per-cell correctness. Known policy terminations contribute zero for each required cell; infrastructure/cancelled attempts remain separate. All 90 evaluation trajectories are excluded from gradients. One zero-advantage batch completes without an update or added resampling.

## Compatibility protocol

The Human Resources/Qwen3.6-35B-A3B example uses one batch of eight and `evaluation.enabled: false`, 80 turns, 64000 context tokens and 16384 response tokens. It has no initial, process or independent final evaluation. Its eight training outcomes and real update support an execution/compatibility claim only.

## Historical reaggregation

`tools/reaggregate_official_metrics.py --run /private/original-run --task-spec /private/task.yaml --output runs/corrected-metrics` verifies selected trajectory identities/digests, reads original Stage 2 logs and writes separate corrected metrics. Its output must be outside the original run. It never samples, queries a warehouse, changes reward or rewrites original records.

Public `evidence/metric-corrections/` records both old and new SRDT for all selected episodes. Cars is numerically unchanged. Two Human Resources episodes change to 0.5, yielding 2/16 models and cohort SRDT 0.125. Complete-pipeline success and training rewards are unchanged. Raw records remain privately archived.
