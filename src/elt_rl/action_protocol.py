"""One action field contract for environment dispatch and native tool declarations."""

from __future__ import annotations

import json

from elt_rl.errors import ActionError

NATIVE_TOOLS = "native_tools_v1"

# Both the model declaration and the environment dispatch use these field sets.
ACTION_FIELDS = {
    "inspect_task": (),
    "list_files": (),
    "read_file": ("path",),
    "write_file": ("path", "content"),
    "edit_file": ("path", "old", "new"),
    "delete_file": ("path",),
    "sync": ("connection",),
    "wait_sync": ("job_id",),
    "dbt": ("command",),
    "query": ("sql",),
    "submit": (),
    "loading": ("command",),
}
LOADING_FIELDS = ACTION_FIELDS


def validate_action_fields(action: dict) -> tuple[str, dict]:
    tool = action.get("tool")
    args = {key: value for key, value in action.items() if key != "tool"}
    fields = ACTION_FIELDS
    if not isinstance(tool, str) or tool not in fields or set(args) != set(fields[tool]):
        raise ActionError("Unknown tool or incorrect argument fields")
    return tool, args


def declared_tools(*, loading=False) -> list[dict]:
    """OpenAI tool declarations accepted by Cookbook's native tool_declare role."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {field: {"type": "string"} for field in fields},
                    "required": list(fields),
                    "additionalProperties": False,
                },
            },
        }
        for name, fields in ACTION_FIELDS.items()
    ]


def action_prefix(instructions: str, *, loading=False) -> list[dict]:
    return [
        {"role": "system", "content": instructions},
        {
            "role": "tool_declare",
            "content": json.dumps(declared_tools(loading=loading), separators=(",", ":")),
        },
    ]


def native_action(message, termination, text: str) -> dict:
    """Translate one valid parsed call, without completing or repairing model output."""
    calls = message.get("tool_calls") or []
    if not termination.is_clean or message.get("unparsed_tool_calls"):
        raise ActionError("Native tool call did not parse cleanly; no tools were executed")
    if len(calls) != 1 or text.strip():
        raise ActionError(
            "Send exactly one native tool call and no text action; no tools were executed"
        )
    call = calls[0]
    if call.type != "function":
        raise ActionError("Expected a function tool call; no tools were executed")
    try:
        args = json.loads(call.function.arguments)
    except (ValueError, TypeError) as exc:
        raise ActionError("Native arguments must be a JSON object; no tools were executed") from exc
    if not isinstance(args, dict) or "tool" in args:
        raise ActionError(
            "Native arguments must be an object without a tool field; no tools were executed"
        )
    canonical = {"tool": call.function.name, **args}
    validate_action_fields(canonical)
    if any(not isinstance(value, str) for value in args.values()):
        raise ActionError("Native tool arguments must match the declared string types")
    if len(json.dumps(canonical)) > 300_000:
        raise ActionError("Action exceeds size budget")
    return canonical


def observation_messages(message, observation: dict) -> list[dict]:
    """Correlate every rejected call, or the single dispatched call, without selecting one."""
    content = json.dumps(observation, default=str)
    calls = message.get("tool_calls") or []
    if not calls:
        return [{"role": "user", "content": content}]
    return [
        {
            "role": "tool",
            "content": content,
            "name": call.function.name,
            **({"tool_call_id": call.id} if call.id is not None else {}),
        }
        for call in calls
    ]
