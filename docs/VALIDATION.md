# Final release validation

Release 0.6.1 passed **252 tests in 257.29 seconds** on Linux with Python 3.12. Lint, formatting (96 Python files), wheel build, clean installation, credential-free native integration, CLI argument checks and pip dependency checks passed. The official-metric regressions include SRDT 0, 0.5 and 1, count-weighted aggregation, per-target errors and immutable historical reaggregation. Reward, Task and Model contract tests remain included.

The clean environment used a provenance-verified Cookbook wheel from the exact pinned commit after the host Git transport failed TLS. All dependencies were freshly installed; no old site-packages were borrowed. See DEPLOYMENT.md for that route. The installed implementation was compared with delivered source. No new paid model experiment was executed.

The representative actual reward recomputed exactly, with maximum difference 0.0, using separately held official references. CI YAML and embedded shell syntax passed local checks; hosted GitHub Actions will run after approved publication and are not claimed as already executed.

Receipts and pytest/lint/format logs are in evidence/validation/. The source experimental records remain unchanged; metric corrections are separate in evidence/metric-corrections/.
