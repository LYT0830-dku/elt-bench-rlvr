"""Retained backend failure classification and sampler uncertainty regressions."""

import pytest

from elt_rl.warehouse import policy_error


async def test_group_strategy_uses_real_token_budget_for_environment_dispatch(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from elt_rl import model_runtime

    closed = []
    core = SimpleNamespace(
        report=None, done=True, persist=lambda: None, close=lambda: closed.append(True)
    )
    env = SimpleNamespace(core=core, renderer=SimpleNamespace(tokenizer=None))
    builder = SimpleNamespace(make_envs=AsyncMock(return_value=[env]))
    policy = model_runtime.ModelPolicy(
        None,
        model_runtime.TokenBudget(100),
        max_tokens=10,
        context_tokens=50,
        max_episode_tokens=20,
        temperature=0.8,
        seed=0,
    )

    async def rollout(bound, selected):
        selected.core.dispatch_check()
        return SimpleNamespace(stop_reason=None)

    monkeypatch.setattr(model_runtime, "run_rollout", rollout)
    result = await model_runtime.SeededGroupStrategy().execute(builder, policy)
    assert len(result.trajectories) == 1 and closed == [True]


@pytest.mark.parametrize("state,expected", [("08001", False), ("28000", False), ("42000", True)])
def test_snowflake_connectivity_is_not_policy_feedback(state, expected):
    errors = pytest.importorskip("snowflake.connector.errors")
    assert policy_error(errors.ProgrammingError(msg="test", sqlstate=state)) is expected


async def test_invalid_sampler_response_is_persisted_as_uncertain():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    tinker = pytest.importorskip("tinker")
    from elt_rl.model_runtime import ModelPolicy, TokenBudget

    budget = TokenBudget(100)
    events = []
    sampler = SimpleNamespace(sample=AsyncMock(return_value=SimpleNamespace(sequences=[])))
    policy = ModelPolicy(
        sampler,
        budget,
        max_tokens=10,
        context_tokens=50,
        max_episode_tokens=20,
        temperature=1,
        seed=0,
        progress=events.append,
    )
    with pytest.raises(RuntimeError, match="no sequences"):
        await policy(tinker.ModelInput.from_ints([1, 2]), [])
    assert events[-1]["status"] == "failed_or_cancelled_usage_uncertain"
    assert events[-1]["sampling_seconds"] >= 0
    assert budget.reserved == 12


def test_model_history_keeps_all_observations_before_context_compaction():
    from copy import deepcopy

    from elt_rl.history import model_messages

    history = [{"role": "system", "content": "tools"}, {"role": "user", "content": "task"}]
    for index in range(7):
        history.extend(
            [
                {"role": "assistant", "content": f"action {index}"},
                {"role": "user", "content": f"result {index}\nfull output"},
            ]
        )
    original = deepcopy(history)
    view = model_messages(history)
    assert history == original and view[:2] == original[:2]
    assert [m for m in view if m["role"] == "assistant"] == [
        m for m in original if m["role"] == "assistant"
    ]
    results = [m["content"] for m in view[2:] if m["role"] == "user"]
    assert results == [f"result {index}\nfull output" for index in range(7)]


async def test_sampling_controls_forward_exactly_and_bind_checkpoint_protocol(
    tmp_path, monkeypatch
):
    import json
    from dataclasses import asdict, replace
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    tinker = pytest.importorskip("tinker")
    from elt_rl.model_runtime import ModelPolicy, TokenBudget
    from elt_rl.train import TrainConfig, checked_checkpoint, fingerprint

    defaults = TrainConfig("task.yaml", "run", task_spec_path="unit-contract.yaml", run_id="unit")
    sdk_defaults = tinker.SamplingParams()
    assert (defaults.temperature, defaults.top_p, defaults.top_k) == (0.8, 1, -1)
    assert (sdk_defaults.temperature, sdk_defaults.top_p, sdk_defaults.top_k) == (1, 1, -1)
    config = replace(defaults, temperature=0.6, top_p=0.95, top_k=20)
    config.validate()
    sequence = SimpleNamespace(tokens=[7, 8], logprobs=[-0.125, -1.5], stop_reason="stop")
    sampler = SimpleNamespace(
        sample=AsyncMock(
            return_value=SimpleNamespace(sequences=[sequence], prompt_cache_hit_tokens=0)
        )
    )
    policy_args = dict(max_tokens=10, context_tokens=50, max_episode_tokens=20, seed=3)
    policy = ModelPolicy(
        sampler,
        TokenBudget(100),
        **policy_args,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )
    episode = policy.for_episode(2)
    assert (episode.temperature, episode.top_p, episode.top_k) == (0.6, 0.95, 20)
    result = await episode(tinker.ModelInput.from_ints([1, 2]), [])
    dispatched = sampler.sample.call_args.kwargs["sampling_params"]
    assert (dispatched.temperature, dispatched.top_p, dispatched.top_k) == (0.6, 0.95, 20)
    assert dispatched.seed == 3 + 2 * 1009
    assert tuple(episode.events[0][key] for key in ("temperature", "top_p", "top_k")) == (
        0.6,
        0.95,
        20,
    )
    assert result.tokens == episode.events[0]["tokens"] == sequence.tokens
    assert result.maybe_logprobs == episode.events[0]["logprobs"] == sequence.logprobs

    task = SimpleNamespace(fingerprint=lambda: {"task": "sampling-contract"})
    inputs = fingerprint(config, task)
    assert tuple(
        inputs["evaluation_protocol"][key] for key in ("temperature", "top_p", "top_k")
    ) == (0.6, 0.95, 20)
    monkeypatch.setattr("elt_rl.resume.validate_resume_owner", lambda _: None)
    checkpoint = tmp_path / "checkpoint.json"
    for key, value in (("temperature", 0.7), ("top_p", 0.9), ("top_k", 10)):
        changed = replace(config, **{key: value})
        changed_inputs = fingerprint(changed, task)
        assert changed_inputs != inputs
        checkpoint.write_text(json.dumps({"fingerprint": inputs, "config": asdict(changed)}))
        with pytest.raises(ValueError, match=f"configuration differs: {key}"):
            checked_checkpoint(str(checkpoint), config, inputs)
        with pytest.raises(ValueError, match="fingerprints differ"):
            checked_checkpoint(str(checkpoint), changed, changed_inputs, for_evaluation=True)
    for key, value in (
        ("temperature", 0),
        ("temperature", float("inf")),
        ("top_p", 0),
        ("top_p", 1.01),
        ("top_p", float("nan")),
        ("top_k", 0),
        ("top_k", -2),
        ("top_k", 2.5),
        ("top_k", True),
    ):
        invalid = replace(config, **{key: value})
        with pytest.raises(ValueError):
            invalid.validate()
        with pytest.raises(ValueError):
            ModelPolicy(
                sampler,
                TokenBudget(100),
                **policy_args,
                temperature=invalid.temperature,
                top_p=invalid.top_p,
                top_k=invalid.top_k,
            )


def test_model_prompt_compacts_oldest_thinking_when_context_budget_is_reached():
    pytest.importorskip("tinker_cookbook")
    import json
    from copy import deepcopy

    import tinker

    from elt_rl.history import model_messages, model_prompt

    class ByteRenderer:
        def build_generation_prompt(self, messages):
            return tinker.ModelInput.from_ints(list(json.dumps(messages).encode()))

    renderer = ByteRenderer()
    history = [{"role": "system", "content": "tools"}, {"role": "user", "content": "task"}]
    for index in range(7):
        history.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": f"thought {index}: " + "x" * 512},
                        {"type": "text", "text": f"action {index}"},
                    ],
                },
                {"role": "user", "content": f"observation {index}\nfull output"},
            ]
        )
    original = deepcopy(history)
    full = renderer.build_generation_prompt(model_messages(history))
    assert model_prompt(history, renderer, full.length * 4).to_ints() == full.to_ints()
    bounded = model_prompt(history, renderer, 4000)
    prompt = bytes(bounded.to_ints()).decode()
    assert bounded.length <= (4000 - 4000 // 4) * 2 // 3
    assert "thought 0:" not in prompt and "thought 6:" in prompt
    assert all(f"action {index}" in prompt for index in range(7))
    assert "tools" in prompt and "task" in prompt
    assert all(f"observation {index}" in prompt for index in range(7))
    assert "(Earlier tool output omitted.)" not in prompt
    assert history == original
    visible_only = [{"role": "user", "content": "visible task " * 500}]
    assert model_prompt(visible_only, renderer, 4000).to_ints() == (
        renderer.build_generation_prompt(visible_only).to_ints()
    )
    thinking_only = [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "x" * 1000}]}
    ]
    assert model_prompt(thinking_only, renderer, 100).to_ints() == (
        renderer.build_generation_prompt([{"role": "assistant", "content": []}]).to_ints()
    )
    assert thinking_only[0]["content"][0]["thinking"] == "x" * 1000
