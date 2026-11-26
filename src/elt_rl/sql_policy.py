"""Fail-closed SQL capability policy; no arbitrary SQL, shell, or filesystem tool."""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope

from elt_rl.task import identifier

# Explicit capabilities: built-in scalar/aggregate functions only, no dynamic/UDF calls.
FUNCTIONS = set(
    """
ABS AVG COUNT SUM MIN MAX COALESCE NULLIF IF CASE AND OR NOT CAST TRY_CAST ROUND FLOOR CEIL
LOWER UPPER LENGTH CHAR_LENGTH TRIM LTRIM RTRIM CONCAT CONCAT_WS SUBSTRING REPLACE
SPLIT_PART STR_POSITION LIKE ILIKE REGEXP_LIKE REGEXP_REPLACE REGEXP_EXTRACT
DATE DATE_TRUNC DATE_PART EXTRACT YEAR MONTH DAY DATEDIFF DATE_DIFF DATE_ADD
STRFTIME STRPTIME TRY_STRPTIME TO_DATE TO_TIMESTAMP TO_CHAR
ROW_NUMBER RANK DENSE_RANK LAG LEAD FIRST_VALUE LAST_VALUE NTH_VALUE
GREATEST LEAST POWER SQRT MOD MEDIAN QUANTILE_CONT PERCENTILE_CONT STDDEV VARIANCE
ARRAY_AGG LISTAGG STRING_AGG BOOL_AND BOOL_OR ANY_VALUE
""".split()
)


def validate(sql: str, dialect: str, *, schemas: set[str] | None = None) -> exp.Expression:
    schemas = schemas or {"airbyte_schema"}
    if not isinstance(sql, str) or len(sql) > 100_000:
        raise ValueError("SQL must be a string of at most 100000 characters")
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except SqlglotError as exc:
        raise ValueError(f"SQL parse error: {str(exc)[:1200]}") from exc
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Exactly one SQL statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise ValueError("Only SELECT queries are allowed")
    forbidden = (exp.Insert, exp.Delete, exp.Update, exp.Drop, exp.Command, exp.Into)
    if any(isinstance(n, forbidden) for n in tree.walk()):
        raise ValueError("SQL contains a forbidden operation")
    # Fold task identifiers before resolving CTE visibility.
    for node in tree.find_all(exp.Identifier):
        node.set("this", node.this.lower())
    try:
        cte_references = {
            id(node)
            for scope in traverse_scope(tree)
            for node, source in scope.selected_sources.values()
            if isinstance(node, exp.Table) and isinstance(source, Scope)
        }
    except SqlglotError as exc:
        raise ValueError(f"Invalid SQL scope: {str(exc)[:1200]}") from exc
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier) or table.catalog:
            raise ValueError("External catalogs and table functions are forbidden")
        identifier(table.name)
        if table.db:
            if table.db.lower() not in schemas:
                raise ValueError("Table is outside the assigned warehouse namespace")
        elif id(table) not in cte_references:
            raise ValueError("Use schema-qualified task tables in airbyte_schema")
    for function in tree.find_all(exp.Func):
        if isinstance(function, exp.Anonymous) or function.sql_name().upper() not in FUNCTIONS:
            raise ValueError(
                f"Function {function.sql_name()} is outside the SQL capability allowlist"
            )
    return tree
