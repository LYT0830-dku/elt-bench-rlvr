"""Preserve string tool payloads that happen to be valid JSON, without relaxing framing."""

import json

import pytest
from tinker_cookbook.renderers.base import ParseTermination, ToolCall, UnparsedToolCall

from elt_rl.action_protocol import native_action
from elt_rl.errors import ActionError
from elt_rl.native_renderers import StringToolQwenRenderer


def convert(raw):
    renderer = object.__new__(StringToolQwenRenderer)
    message = {
        "role": "assistant",
        "content": [],
        "unparsed_tool_calls": [UnparsedToolCall(raw_text=raw, error="Expected JSON")],
    }
    renderer._postprocess_parsed_message(message)
    return message


@pytest.mark.parametrize("content", ['{"connections":[]}', '"quoted"', "42", "null", "\n汉字\n\n"])
def test_declared_string_payload_survives_native_postprocessing(content):
    raw = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>\ndbt/models/example.sql\n</parameter>\n"
        f"<parameter=content>\n{content}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    message = convert(raw)
    action = native_action(message, ParseTermination.STOP_SEQUENCE, "")
    assert action == {"tool": "write_file", "path": "dbt/models/example.sql", "content": content}
    assert json.loads(message["tool_calls"][0].function.arguments)["content"] == content


@pytest.mark.parametrize(
    "body",
    [
        "<parameter=sql>SELECT 1</parameter><parameter=sql>SELECT 2</parameter>",
        "<parameter=sql>SELECT 1</parameter>unexpected text",
        "<parameter=sql>SELECT 1",
    ],
)
def test_malformed_or_duplicate_native_parameters_cannot_execute(body):
    message = convert(f"<tool_call><function=query>{body}</function></tool_call>")
    with pytest.raises(ActionError):
        native_action(message, ParseTermination.STOP_SEQUENCE, "")


def test_string_transport_still_rejects_multiple_calls_and_visible_text():
    message = convert(
        "<tool_call><function=query><parameter=sql>SELECT 1</parameter></function></tool_call>"
    )
    assert isinstance(message["tool_calls"][0], ToolCall)
    with pytest.raises(ActionError):
        native_action(message, ParseTermination.STOP_SEQUENCE, "run this")
    message["tool_calls"] *= 2
    with pytest.raises(ActionError):
        native_action(message, ParseTermination.STOP_SEQUENCE, "")
