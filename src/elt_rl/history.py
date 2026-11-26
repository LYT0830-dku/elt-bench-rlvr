"""Append model history until its context budget requires a stable compaction."""

from __future__ import annotations

from collections.abc import Sequence

REASONING_RESPONSE_RESERVE = 16384
MODEL_HISTORY_POLICY_VERSION = "append_until_context_compaction_v2"
HISTORY_COMPACTION_RATIO = (2, 3)


def model_messages(messages: Sequence[dict]) -> list[dict]:
    """Preserve every observation while it fits; never mutate the raw transcript."""
    return list(messages)


def model_prompt(messages: Sequence[dict], renderer, context_tokens: int, *, elided=None):
    """Keep append-only prefixes between infrequent, budget-triggered compactions.

    The environment retains the omission set across turns. Initial instructions,
    task, declarations, visible actions and the latest tool result stay intact.
    Original messages, action tokens and sampling log probabilities are unchanged.
    """
    if elided is None:
        elided = set()
    view = model_messages(messages)
    observations = [i for i, m in enumerate(view) if m["role"] in {"user", "tool"}]
    thoughts = []
    for index, message in enumerate(view):
        content = message["content"]
        if message["role"] == "assistant" and isinstance(content, list):
            thoughts.extend(
                (index, part) for part, item in enumerate(content) if item["type"] == "thinking"
            )
        if ("observation", index) in elided:
            view[index] = {**message, "content": "(Earlier tool output omitted.)"}
        elif isinstance(content, list):
            retained = [
                item for part, item in enumerate(content) if ("thinking", index, part) not in elided
            ]
            if retained != content:
                view[index] = {**message, "content": retained}
    prompt = renderer.build_generation_prompt(view)
    target = context_tokens - min(REASONING_RESPONSE_RESERVE, context_tokens // 4)
    if prompt.length <= target:
        return prompt
    compact_target = target * HISTORY_COMPACTION_RATIO[0] // HISTORY_COMPACTION_RATIO[1]
    candidates = [("thinking", index, part) for index, part in thoughts]
    candidates += [("observation", index) for index in observations[1:-1]]
    for key in candidates:
        if key in elided:
            continue
        elided.add(key)
        index = key[1]
        message = messages[index]
        if key[0] == "observation":
            view[index] = {**message, "content": "(Earlier tool output omitted.)"}
        else:
            retained = [
                item
                for part, item in enumerate(message["content"])
                if ("thinking", index, part) not in elided
            ]
            view[index] = {**message, "content": retained}
        prompt = renderer.build_generation_prompt(view)
        if prompt.length <= compact_target:
            break
    return prompt
