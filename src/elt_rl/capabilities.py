"""Validate model-authored configuration before it reaches executable tools."""

from __future__ import annotations

from pathlib import Path

import yaml
from jinja2 import Environment, nodes

from elt_rl.errors import ActionError
from elt_rl.sql_policy import validate


def load_yaml(path: Path, *, source_schema=False):
    try:
        value = yaml.safe_load(path.read_text())
        seen = set()
        count = 0

        def check(node, trail=(), depth=0):
            nonlocal count
            count += 1
            if depth > 32 or count > 10000:
                raise ActionError("YAML exceeds structural budget")
            if isinstance(node, (dict, list)):
                if id(node) in seen:
                    raise ActionError("Cyclic YAML aliases are forbidden")
                seen.add(id(node))
                if isinstance(node, dict):
                    for key, child in node.items():
                        if not isinstance(key, str):
                            raise ActionError("YAML keys must be strings")
                        check(key, trail + ("$key",), depth + 1)
                        check(child, trail + (key,), depth + 1)
                else:
                    for child in node:
                        check(child, trail, depth + 1)
                seen.remove(id(node))
            elif isinstance(node, str) and ("{{" in node or "{%" in node):
                if not (
                    source_schema
                    and trail
                    and trail[-1] == "schema"
                    and node
                    in {"{{ var('destination_schema') }}", '{{ var("destination_schema") }}'}
                ):
                    raise ActionError("Executable YAML metadata is forbidden")

        check(value)
        return value
    except (yaml.YAMLError, RecursionError) as exc:
        raise ActionError("Invalid model-authored YAML") from exc


def rendered_sql(text: str, dialect: str) -> str:
    """Accept declarative dbt templating, excluding Python/Jinja execution escapes."""
    try:
        tree = Environment().parse(text)
    except Exception as exc:
        raise ActionError(f"Invalid dbt template: {exc}") from exc
    allowed = (
        nodes.Template,
        nodes.Output,
        nodes.TemplateData,
        nodes.Call,
        nodes.Name,
        nodes.Const,
        nodes.Keyword,
    )
    for node in tree.find_all(nodes.Node):
        if not isinstance(node, allowed):
            raise ActionError(f"Unsupported dbt template operation: {type(node).__name__}")
        if isinstance(node, nodes.Call):
            if not isinstance(node.node, nodes.Name) or node.node.name not in {
                "ref",
                "source",
                "config",
                "var",
            }:
                raise ActionError("Only ref/source/config/var dbt calls are allowed")
            if (
                any(not isinstance(a, nodes.Const) for a in node.args)
                or node.dyn_args
                or node.dyn_kwargs
            ):
                raise ActionError("dbt call arguments must be literals")
            if node.node.name == "config":
                if node.args or any(
                    k.key != "materialized"
                    or not isinstance(k.value, nodes.Const)
                    or k.value.value not in {"table", "view"}
                    for k in node.kwargs
                ):
                    raise ActionError("Only table/view materialization is model-configurable")
            elif node.kwargs:
                raise ActionError("Unexpected dbt keyword arguments")

    def table(*args):
        from elt_rl.task import identifier

        return "airbyte_schema." + identifier(args[-1])

    def variable(name):
        if name != "destination_schema":
            raise ActionError("Only the assigned schema is a public dbt variable")
        return "airbyte_schema"

    try:
        sql = (
            Environment()
            .from_string(text)
            .render(ref=table, source=table, config=lambda **kw: "", var=variable)
        )
        validate(sql, dialect, schemas={"airbyte_schema"})
    except (ValueError, TypeError) as exc:
        raise ActionError(str(exc)) from exc
    return sql


def dbt_policy(root: Path, dialect: str):
    project_path = root / "dbt_project.yml"
    if not project_path.is_file():
        raise ActionError("Write dbt/dbt_project.yml first")
    project = load_yaml(project_path)
    from elt_rl.task import identifier

    if not isinstance(project, dict):
        raise ActionError("dbt project must be a mapping")
    identifier(project.get("name"))
    if "{{" in project_path.read_text() or "{%" in project_path.read_text():
        raise ActionError("Executable project metadata is forbidden")
    allowed = {"name", "version", "config-version", "profile", "model-paths", "models"}
    unsupported = set(project) - allowed
    if unsupported:
        hint = (
            "; source declarations belong in a YAML file under dbt/models, "
            "with version: 2 and a sources list"
            if "sources" in unsupported
            else ""
        )
        raise ActionError(
            "dbt project has unsupported execution settings: "
            + ", ".join(sorted(map(str, unsupported)))
            + hint
        )
    if project.get("profile") != "elt" or project.get("model-paths", ["models"]) != ["models"]:
        raise ActionError("Use profile elt and model path models")

    def check_config(value):
        if not isinstance(value, dict):
            raise ActionError("Invalid dbt model settings")
        for key, child in value.items():
            if key in {"materialized", "+materialized"} and child in {"table", "view"}:
                continue
            if key in {"tags", "+tags"}:
                tags = [child] if isinstance(child, str) else child
                if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                    raise ActionError("dbt tags must be a string or a list of strings")
                continue
            if key.startswith("+") or not isinstance(child, dict):
                raise ActionError("Hooks, schema overrides and custom dbt execution are forbidden")
            identifier(key)
            check_config(child)

    check_config(project.get("models", {}))
    for path in root.rglob("*"):
        if not path.is_file() or path == project_path:
            continue
        if path.is_symlink() or "models" not in path.relative_to(root).parts:
            raise ActionError("Only project and model files are executable dbt inputs")
        try:
            identifier(path.stem)
        except ValueError as exc:
            raise ActionError("dbt model filenames must be plain identifiers") from exc
        if path.suffix == ".sql":
            rendered_sql(path.read_text(), dialect)
        elif path.suffix in {".yml", ".yaml"}:
            content = path.read_text()
            # YAML metadata cannot introduce templating or hooks. Schema variable is the one exception.
            content = content.replace("{{ var('destination_schema') }}", "airbyte_schema").replace(
                '{{ var("destination_schema") }}', "airbyte_schema"
            )
            if "{{" in content or "{%" in content:
                raise ActionError("Executable YAML templates are forbidden")
            data = load_yaml(path, source_schema=True)
            if not isinstance(data, dict):
                raise ActionError("dbt schema YAML must be a mapping")
            for section in ["sources", "models"]:
                entries = data.get(section, [])
                if not isinstance(entries, list) or any(
                    not isinstance(entry, dict) for entry in entries
                ):
                    raise ActionError(f"dbt {section} must be a list of mappings")
            for source in data.get("sources", []):
                tables = source.get("tables", [])
                if not isinstance(tables, list) or any(
                    not isinstance(table, dict) for table in tables
                ):
                    raise ActionError("dbt source tables must be a list of mappings")
            for source in data.get("sources", []):
                if source.get("schema") in {
                    "{{ var('destination_schema') }}",
                    '{{ var("destination_schema") }}',
                }:
                    source["schema"] = "airbyte_schema"
            if not isinstance(data, dict) or set(data) - {"version", "sources", "models"}:
                raise ActionError("Unsupported dbt schema YAML")
            for source in data.get("sources", []):
                identifier(source.get("name"))
                if (
                    set(source) - {"name", "schema", "tables", "description"}
                    or source.get("schema") != "airbyte_schema"
                ):
                    raise ActionError("Sources must use the assigned destination schema")
                for table in source.get("tables", []):
                    identifier(table.get("name"))
                    if set(table) - {"name", "description", "columns"}:
                        raise ActionError("Unsupported source table configuration")

            # Tests, macros and model configs can execute adapter code; host verification owns them.
            def inspect(value):
                if isinstance(value, dict):
                    if "name" in value:
                        identifier(value["name"])
                    if set(value) & {
                        "config",
                        "tests",
                        "data_tests",
                        "pre_hook",
                        "post_hook",
                        "identifier",
                        "database",
                    }:
                        raise ActionError("Executable dbt schema configuration is forbidden")
                    for child in value.values():
                        inspect(child)
                elif isinstance(value, list):
                    for child in value:
                        inspect(child)

            inspect(data)
        else:
            raise ActionError("Unexpected dbt file type")
