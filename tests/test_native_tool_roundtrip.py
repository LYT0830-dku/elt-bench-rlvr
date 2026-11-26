"""One offline regression: pinned native renderer through the existing env and RL data path."""

import asyncio
import copy
import gzip
import hashlib
import json
import time
from types import SimpleNamespace

import pytest
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.renderers.base import ParseTermination, ToolCall
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.rollout_runner import run_rollout

from elt_rl.action_protocol import ACTION_FIELDS, NATIVE_TOOLS, native_action
from elt_rl.cookbook_env import CookbookELTEnv
from elt_rl.environment import INSTRUCTIONS, ELTEnvironment, initial_messages
from elt_rl.errors import ActionError
from elt_rl.history import model_messages
from elt_rl.model_runtime import resolve_renderer
from elt_rl.project import Project
from elt_rl.timing import Timings
from elt_rl.train import (
    TrainConfig,
    implementation_record,
    prepare_data,
    save_training_payload,
)


def test_native_tools_real_renderer_env_history_and_raw_training_roundtrip(tmp_path):
    _, renderer = resolve_renderer("thinkingmachines/Inkling-2-Mini", effort=0.0)
    literal = '# offline transport fixture\nliteral: "汉字 } [ ] \\""\n'
    args = {"path": "dbt/transport_fixture.yml", "content": literal}
    assistant = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "offline transport fixture"}],
        "tool_calls": [
            ToolCall(
                id="fixture-id",
                function=ToolCall.FunctionBody(
                    name="write_file", arguments=json.dumps(args, ensure_ascii=False)
                ),
            )
        ],
    }
    model_input, _ = renderer.build_supervised_example([assistant])
    all_ids = model_input.to_ints()
    tokens = all_ids[all_ids.index(200001) :]
    parsed, termination = renderer.parse_response(tokens)
    assert termination == ParseTermination.STOP_SEQUENCE
    assert native_action(parsed, termination, "") == {"tool": "write_file", **args}

    # Exercise the real ELTEnvironment.step and Project.write. Only warehouse/initial
    # service allocation and persistence are replaced; this fixture cannot contact services.
    core = object.__new__(ELTEnvironment)
    core.dispatch_check = lambda: None
    core.done = core.closed = False
    core.started = time.monotonic()
    core.turns = core.invalid_repeats = 0
    core.invalid_json = core.report = core.termination = None
    core.max_seconds, core.max_turns = 60, 1
    core.trace, core.initial_messages = [], []
    core.timings = Timings()
    core.project = Project(tmp_path / "project", {})
    core.runtime = SimpleNamespace(runner=None)
    core._runtime = lambda: core.runtime
    core.persist = lambda: None
    observation = {"instructions": INSTRUCTIONS, "task": {"name": "transport-fixture"}}
    core.initial = lambda: observation
    env = CookbookELTEnv(core, renderer, context_tokens=64000)
    # Synthetic local fixture probabilities/ID; no provider sampling occurs in this test.
    logprobs = [-0.125] * len(tokens)
    sequence_id = "offline-synthetic-transport-fixture"
    event = {"tokens": tokens.copy(), "logprobs": logprobs.copy(), "sequence_id": sequence_id}
    original_event = copy.deepcopy(event)
    env.sampling_trace = [event]

    async def policy(model_input, stop, *, max_tokens=None):
        assert stop == [200006]
        return TokensWithLogprobs(
            tokens=tokens.copy(),
            maybe_logprobs=logprobs.copy(),
            stop_reason="stop",
            sequence_id=sequence_id,
        )

    trajectory = asyncio.run(run_rollout(policy, env))
    assert core.turns == 1 and core.project.read(args["path"]) == literal
    assert core.trace[0]["action"] == {"tool": "write_file", **args}
    assert env.messages[-2] == parsed
    assert env.messages[-1]["role"] == "tool"
    assert env.messages[-1]["name"] == "write_file"
    assert env.messages[-1].get("tool_call_id") == parsed["tool_calls"][0].id
    assert event["tokens"] == original_event["tokens"]
    assert event["logprobs"] == original_event["logprobs"]
    assert event["sequence_id"] == original_event["sequence_id"]
    assert event["parsed_message"]["tool_calls"][0]["function"]["name"] == "write_file"
    assert trajectory.transitions[0].ac.tokens == tokens
    assert trajectory.transitions[0].ac.logprobs == logprobs
    data = trajectory_to_data(trajectory, traj_advantage=0.5)
    assert len(data) == 1
    datum = data[0]
    mask = datum.loss_fn_inputs["mask"].data
    assert [
        v for v, m in zip(datum.loss_fn_inputs["target_tokens"].data, mask, strict=True) if m
    ] == tokens
    assert [
        v for v, m in zip(datum.loss_fn_inputs["logprobs"].data, mask, strict=True) if m
    ] == logprobs
    prepared, _ = prepare_data(data)
    payload_path = tmp_path / "offline-payload.json.gz"
    save_training_payload(payload_path, data, prepared)
    payload = json.loads(gzip.decompress(payload_path.read_bytes()))
    exact = payload["datums"][0]
    assert sequence_id in json.dumps(exact["datum"]["loss_fn_input_spans"])
    assert exact["cookbook_mask"] == list(mask)
    assert [
        v
        for v, m in zip(
            exact["datum"]["loss_fn_inputs"]["target_tokens"]["data"], mask, strict=True
        )
        if m
    ] == tokens

    initial = initial_messages(observation)
    assert initial == env.messages[: len(initial)]
    assert initial[1]["role"] == "tool_declare"
    schemas = json.loads(initial[1]["content"])
    assert {
        s["function"]["name"]: tuple(s["function"]["parameters"]["required"]) for s in schemas
    } == ACTION_FIELDS
    assert "You must configure and execute extraction/loading yourself" in initial[0]["content"]
    native_prompt = renderer.tokenizer.decode(renderer.build_generation_prompt(initial).to_ints())
    assert "tool_declare<|content_xml|>" in native_prompt and "write_file" in native_prompt
    assert "Thinking effort level: 0" in native_prompt

    history = copy.deepcopy(initial)
    for index in range(7):
        history.extend(
            [
                copy.deepcopy(parsed),
                {
                    "role": "tool",
                    "name": "write_file",
                    "tool_call_id": f"fixture-{index}",
                    "content": f"result {index}\nline",
                },
            ]
        )
    before = copy.deepcopy(history)
    view = model_messages(history)
    assert history == before and view[:3] == initial
    tool_results = [m for m in view if m["role"] == "tool"]
    assert [m["content"] for m in tool_results] == [f"result {i}\nline" for i in range(7)]
    assert [m["tool_call_id"] for m in tool_results] == [f"fixture-{i}" for i in range(7)]
    renderer.build_generation_prompt(view)  # Actual native re-render of complete history.

    # Reject whole responses, never execute a selected subset or salvage text JSON.
    for message, term, text in [
        ({"role": "assistant", "content": "{}"}, termination, "{}"),
        ({**parsed, "tool_calls": parsed["tool_calls"] * 2}, termination, ""),
        (parsed, termination, '{"tool":"submit"}'),
        (parsed, ParseTermination.MALFORMED, ""),
        ({**parsed, "unparsed_tool_calls": [object()]}, termination, ""),
    ]:
        with pytest.raises(ActionError):
            native_action(message, term, text)
    for bad_args in ({"tool": "submit"}, {"path": "x"}, {"path": 7, "content": "x"}):
        bad = {
            **parsed,
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="write_file", arguments=json.dumps(bad_args)
                    )
                )
            ],
        }
        with pytest.raises(ActionError):
            native_action(bad, termination, "")
    raw_bad = '<|message_model|>inspect_task<|content_invoke_tool_json|>{"name":"inspect_task"}<|end_message|><|content_model_end_sampling|>'
    # Actual pinned Inkling2 frame IDs; ordinary text is encoded separately.
    bad_tokens = (
        [200001]
        + renderer.tokenizer.encode("inspect_task")
        + [200049]
        + renderer.tokenizer.encode('{"name":"inspect_task"}')
        + [200010, 200006]
    )
    assert renderer.tokenizer.decode(bad_tokens) == raw_bad
    bad_message, bad_term = renderer.parse_response(bad_tokens)
    assert bad_term == ParseTermination.MALFORMED
    with pytest.raises(ActionError):
        native_action(bad_message, bad_term, bad_message["content"])

    config = TrainConfig(
        task="unused",
        output="unused",
        model="thinkingmachines/Inkling-2-Mini",
        renderer="tml_v0",
        renderer_effort=0.0,
        action_protocol=NATIVE_TOOLS,
    )
    task = SimpleNamespace(public=lambda: {"name": "transport-fixture"})
    record = implementation_record(config, task)
    recorded = record["public_initial_messages"]
    assert recorded[0]["content"] == initial[0]["content"]
    assert recorded[1] == initial[1]
    assert (
        record["public_initial_messages_sha256"]
        == hashlib.sha256(json.dumps(recorded, sort_keys=True, default=str).encode()).hexdigest()
    )
    assert record["action_protocol"] == NATIVE_TOOLS


def test_remote_execution_requires_authorization_before_service(tmp_path, monkeypatch):
    import tinker

    from elt_rl import train as trainer

    monkeypatch.setattr(
        tinker, "ServiceClient", lambda *a, **k: pytest.fail("Service created before authorization")
    )
    config = TrainConfig("unprepared", str(tmp_path / "unused"))
    with pytest.raises(ValueError, match="verified task specification"):
        trainer.preflight(config)
    assert not (tmp_path / "unused").exists()
