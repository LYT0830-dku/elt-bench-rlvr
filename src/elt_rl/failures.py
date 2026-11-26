"""Conservative classification of failures from external ELT executors."""

from elt_rl.errors import InfrastructureError

INFRA_MARKERS = (
    "connection refused",
    "connection reset",
    "connection timed out",
    "connection was closed",
    "network is unreachable",
    "could not connect",
    "failed to connect",
    "certificate verify failed",
    "authentication failed",
    "incorrect username or password",
    "jwt token is invalid",
    "out of memory",
    "oomkilled",
    "warehouse is suspended",
    "service unavailable",
)


def check_dbt_result(result):
    if result.get("exit_code") == 0:
        return
    text = result.get("output", "").lower()
    if result.get("timed_out") or any(marker in text for marker in INFRA_MARKERS):
        raise InfrastructureError("dbt infrastructure failure; inspect private tool logs")
    if any(
        marker in text
        for marker in (
            "compilation error",
            "parsing error",
            "sql compilation error",
            "syntax error",
            "does not exist or not authorized",
            "could not find model",
            "not found in the dbt project",
            "conversion error",
            "numeric value",
            "division by zero",
            "invalid input syntax",
        )
    ):
        return
    raise InfrastructureError("Unclassified dbt failure; do not turn it into policy zero reward")
