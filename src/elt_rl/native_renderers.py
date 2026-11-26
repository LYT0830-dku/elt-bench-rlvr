"""Schema-faithful string arguments for Cookbook's native XML tool transport."""

from __future__ import annotations

import json
import re

from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import ToolCall, UnparsedToolCall
from tinker_cookbook.renderers.qwen3_5 import _FUNCTION_BLOCK_RE, Qwen3_5Renderer

_PARAMETER = re.compile(r"<parameter=(?P<name>[^>\n]+)>(?P<value>.*?)</parameter>", re.DOTALL)


class StringToolQwenRenderer(Qwen3_5Renderer):
    """Keep declared string parameters as text, including JSON document content.

    The upstream framing validator remains authoritative. Its untyped json.loads
    coercion is replaced only for this environment's all-string tool schema.
    """

    def _parse_qwen3_5_tool_call_xml(self, raw_text):
        parsed = super()._parse_qwen3_5_tool_call_xml(raw_text)
        if isinstance(parsed, UnparsedToolCall):
            return parsed
        match = _FUNCTION_BLOCK_RE.match(raw_text)
        if match is None:
            return UnparsedToolCall(raw_text=raw_text, error="Malformed native XML tool call")
        arguments = {}
        for parameter in _PARAMETER.finditer(match.group("body")):
            name = parameter.group("name").strip()
            if name in arguments:
                return UnparsedToolCall(raw_text=raw_text, error="Duplicate tool parameter")
            value = parameter.group("value")
            # The native template puts one framing newline around each value.
            value = (
                value.removeprefix("\r\n") if value.startswith("\r\n") else value.removeprefix("\n")
            )
            value = (
                value.removesuffix("\r\n") if value.endswith("\r\n") else value.removesuffix("\n")
            )
            arguments[name] = value
        return ToolCall(
            id=parsed.id,
            function=ToolCall.FunctionBody(
                name=parsed.function.name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )


def _string_renderer(tokenizer, image_processor=None):
    return StringToolQwenRenderer(
        tokenizer,
        image_processor=image_processor,
        strip_thinking_from_history=False,
    )


def register_native_renderers():
    if not renderers.is_renderer_registered("qwen3_5_strings"):
        renderers.register_renderer("qwen3_5_strings", _string_renderer)
