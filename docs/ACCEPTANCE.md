# Engineering acceptance evidence

| Required case | Evidence | Actual result |
|---|---|---|
| Credential-free integration | evidence/native-integration.json | Real HTTP extraction, native loading, DuckDB and dbt; 16/16 correct cells |
| Official task rollout | evidence/representative-rollout/ | Model-authored loading and dbt SQL, actual prediction and feedback; cars reward 12/21, SRDEL true, SRDT 0 |
| Real Tinker optimization | evidence/cars/optimizer-receipts/batch-0001.json and checkpoint-lineage.json | Gradient norm 148.085495, 69700 nonzero-advantage tokens, saved training/sampler checkpoints |

The selected handoff includes cars Pure Cell training (16 batches, 15 updates) and Human Resources/Qwen3.6-35B-A3B verification (eight rollouts, one update). The superseded first experiment is excluded from the handoff. Actual configurations and all selected rewards/advantages accompany the evidence. These cases prove execution and optimization; no complete final cars pipeline or Human Resources post-training improvement is claimed.

Public evidence is allowlisted and sanitized. Full raw records and original checkpoint addresses are privately archived. Learned weights remain in the authorized Tinker account; metadata is not a local binary weight export. Official ground truth is fetched separately for reward verification and is not redistributed.
