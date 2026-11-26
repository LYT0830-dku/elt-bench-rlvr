# Third-party sources

This package declares dependencies; it does not bundle their implementations.
Their license files remain in their distributions. The following describes the
versions inspected for this review, not a replacement for their full notices.

| Component | Inspected version / revision | License |
|---|---|---|
| DuckDB | 2.0.1 | MIT |
| SQLGlot | 30.34.0 | MIT |
| PyYAML | 6.0.4 | MIT |
| HTTPX | 0.28.1 | BSD-3-Clause |
| Tinker | 0.39.2 | Apache-2.0 |
| Tinker Cookbook | `8bf7dd6192a3f1fa64640a04ec54a7f6866bbd83` | Apache-2.0 |
| Snowflake Connector | 4.8.1 | Apache-2.0 |
| psycopg (sources extra) | 3.3.9 | LGPL-3.0-only |
| boto3 (sources extra) | 1.44.10 | Apache-2.0 |
| pandas (audit extra) | 3.1.1 | BSD-3-Clause |
| NumPy (audit extra) | 2.5.5 | BSD-3-Clause and bundled notices |

The recorded Linux control dependencies are in `tools/native-linux-control.lock`.
The separate fixed dbt environment is recorded in `tools/python-tools.lock`.

[ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench) task specifications,
public data archives and reference tables are fetched separately. The official
grader adapter loads an explicitly supplied pinned Git checkout; no grader code
or benchmark data is redistributed in this package. A top-level license was not
found in upstream revision `e2e377766604f61930165f20d4fcab8400a3e997`; nested SWE-agent has its own license. Review
upstream/data terms before redistributing those materials.

The author has not yet selected a redistribution license for the project source.
Source visibility is public under the owner's publication approval; complete research records remain private. No open-source permission grant is implied by
this notice or by installing third-party dependencies.

Release 0.6.1 includes project code, the small retail integration fixture and
derived experiment statistics/figures and one model-generated cars prediction. It excludes official source/reference
tables, professor-provided documents, original evaluator code, model weights,
tokenizer downloads and raw human-resources records. Model names identify the
recorded services; their artifacts are not sublicensed or bundled here. Public
evidence contains correctness statistics rather than employee names, addresses
or cell values. Complete original research records remain private and must not
be uploaded with this source release.

The representative prediction and loading/dbt code are model-generated experiment outputs. Official reference tables are obtained separately for offline verification; their values are not included in this release. No Human Resources employee cell values are published. The selected delivery excludes the superseded first experiment.
