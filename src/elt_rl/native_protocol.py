"""Public native-tool guide; contains no pipeline, source values or reference answers."""

GUIDE = """Native Extraction and Loading protocol v1

The official source data and target data model are unchanged. This deployment
replaces Airbyte Platform/Terraform operations with a constrained declarative
loading executor. Original Airbyte documentation describes the historical tools;
use this protocol for the installed execution backend.

1. Inspect the task, data model and source schemas. inspect_task lists available
   logical source IDs, actual connector protocols, formats and column names.
2. Write elt/loading.json. The top-level keys are format, destination, connections.
   format is native_loading_v1. destination is an object with database assigned
   and schema airbyte_schema. connections is a list of your connection objects.
3. Each connection has exactly: name, source, connector, format, table, columns,
   sync_mode. name is a unique identifier of your choosing. source chooses an
   available source ID. connector/format choose its actual extraction protocol.
   table is the destination table identifier. columns maps chosen source columns
   to SQL types VARCHAR/BIGINT/DOUBLE/REAL/BOOLEAN/DATE/TIMESTAMP (case insensitive;
   STRING and TEXT are aliases for VARCHAR). sync_mode is
   full_refresh_append or full_refresh_overwrite. Values, Python, SQL, file paths,
   URLs, hooks and credentials are not loading configuration fields.
4. loading validate checks the declaration. loading plan returns intended work.
   loading apply registers it without extracting or loading data. Returned
   native_connection.NAME addresses are episode-owned capabilities.
5. sync(connection) launches a real extraction and loading job. wait_sync(job_id)
   waits and returns actual state, errors, row counts and content fingerprints.
   query inspects tables in the assigned warehouse. A new sync invalidates the
   prior transformation receipt; changed configuration must be applied again.
6. Author the dbt project with profile elt, models under dbt/models and source
   declarations in a YAML file under dbt/models with schema
   {{ var('destination_schema') }}. The dbt_project.yml configures the project,
   not source definitions. Model materialization and plain tags are supported.
   dbt compile/run execute the validated
   project in a trusted installed dbt interpreter. No arbitrary macros/hooks,
   shell commands, Python, packages, files or networks are model capabilities.
7. Submit freezes the exact loading/dbt inputs that successfully executed. The
   trusted evaluator checks the resulting warehouse. Replay uses a fresh source
   instance and warehouse namespace, and performs extraction/loading again.

All source credentials, endpoints and warehouse manager credentials are supplied
by the host and unavailable through file or query tools. The native backend uses
PostgreSQL, authenticated HTTP/HTTPS, an authenticated S3-compatible server and
the Snowflake connector. It does not claim to execute original Airbyte jobs.
"""
