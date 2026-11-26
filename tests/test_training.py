"""Real Cookbook conversion, loss masks, payload evidence and production gates."""

import gzip
import hashlib
import json
from types import SimpleNamespace

import pytest
import tinker
from tinker_cookbook.rl.types import InitialObservationOverflow

from elt_rl.cookbook_env import CookbookELTEnv
from elt_rl.environment import ELTEnvironment
from elt_rl.train import TrainConfig, prepare_data, save_training_payload, train


def test_submission_receipt_agrees_with_scored_warehouse():
    from elt_rl.artifacts import valid_submission_receipt

    official = {"official_score": 1.0, "official_success": True}
    raw = {
        "execution_result": {
            "kind": "submission",
            "execution_success": True,
            "pipeline_valid": True,
            "execution": {},
            "official": official,
        },
        "report": {
            **official,
            "execution_success": True,
            "pipeline_valid": True,
            "success": True,
            "training_reward": 1.0,
        },
    }
    assert valid_submission_receipt(raw)
    raw["report"]["official_score"] = 0.0
    assert not valid_submission_receipt(raw)
    raw["execution_result"] = None
    assert not valid_submission_receipt(raw)


@pytest.mark.parametrize("steps", [1, 2])
def test_configured_update_count_controls_evaluation_and_resume(steps, tmp_path):
    from dataclasses import asdict

    from elt_rl.control import RunControl
    from elt_rl.errors import IntegrityError
    from elt_rl.resume import canonical_checkpoint

    config = TrainConfig(
        "task.yaml",
        "run",
        batches=steps,
        concurrency=2,
        task_spec_path="unit-contract.yaml",
        run_id="unit",
    )
    config.validate()
    assert config.group_size == 8
    assert config.eval_episodes == 10
    control = RunControl(steps=config.batches)
    control.claim_evaluation(steps)
    with pytest.raises(ValueError, match="evaluation step"):
        control.claim_evaluation(steps + 1)
    checkpoint = tmp_path / f"checkpoint-{steps:04d}.json"
    saved = {
        "updates": steps,
        "config": asdict(config),
        "fingerprint": {"training_protocol": {"batches": steps}},
    }
    checkpoint.write_text(json.dumps(saved))
    assert canonical_checkpoint(checkpoint) == checkpoint
    checkpoint.write_text(json.dumps({**saved, "updates": steps + 1}))
    with pytest.raises(IntegrityError, match="update identity"):
        canonical_checkpoint(checkpoint)


async def test_local_fixture_cannot_enter_production_training(monkeypatch, tmp_path, task_path):
    monkeypatch.setattr(
        tinker,
        "ServiceClient",
        lambda *args, **kwargs: pytest.fail("Service created before the native-mode gate"),
    )
    output = tmp_path / "train"
    with pytest.raises(ValueError, match="verified task specification"):
        await train(TrainConfig(str(task_path), str(output)))
    assert not output.exists()


async def test_initial_context_overflow_never_samples(task):
    renderer = SimpleNamespace(
        renderer_name="tml_v0",
        build_generation_prompt=lambda _: SimpleNamespace(length=100),
        get_stop_sequences=lambda: [],
    )
    with ELTEnvironment(task) as core:
        result = await CookbookELTEnv(core, renderer, context_tokens=10).initial_observation()
        assert isinstance(result, InitialObservationOverflow)
        assert result.reward == 0 and core.done


def test_training_payload_preserves_mask_and_provenance(tmp_path):
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[2, 3], dtype="int64", shape=[2]),
            "logprobs": tinker.TensorData(data=[0.0, -0.5], dtype="float32", shape=[2]),
            "advantages": tinker.TensorData(data=[0.0, 0.1], dtype="float32", shape=[2]),
            "mask": tinker.TensorData(data=[0.0, 1.0], dtype="float32", shape=[2]),
        },
    )
    datum = datum.with_provenance(
        model_input=[
            tinker.PromptProvenanceSpan(sequence_id="test-sequence", length=1),
            tinker.SampledProvenanceSpan(sequence_id="test-sequence", length=1),
        ],
        loss_fn_inputs=[tinker.SampledProvenanceSpan(sequence_id="test-sequence", length=2)],
    )
    prepared, counts = prepare_data([datum])
    sent = prepared[0]
    assert set(sent.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
    payload = tmp_path / "payload.json.gz"
    receipt = save_training_payload(payload, [datum], prepared)
    exact = json.loads(gzip.decompress(payload.read_bytes()))["datums"][0]
    assert exact["datum"]["model_input"] == sent.model_input.model_dump(mode="json")
    assert exact["cookbook_mask"] == [0, 1]
    assert exact["datum"]["loss_fn_inputs"]["advantages"] == {
        "data": list(sent.loss_fn_inputs["advantages"].data),
        "dtype": "float32",
        "shape": [2],
    }
    assert [span["type"] for span in exact["datum"]["model_input_spans"]] == [
        "PromptProvenanceSpan",
        "SampledProvenanceSpan",
    ]
    assert exact["datum"]["loss_fn_input_spans"] == [
        {"type": "SampledProvenanceSpan", "sequence_id": "test-sequence", "offset": 0, "length": 2}
    ]
    assert receipt["sha256"] == hashlib.sha256(payload.read_bytes()).hexdigest()
    assert counts["training_input_tokens"] == 2 and counts["nonzero_advantage_tokens"] == 1


def test_fixed_renderer_effort_binds_all_prompt_nodes(monkeypatch, tmp_path, task):
    from dataclasses import replace
    from inspect import signature

    from tinker_cookbook import renderers, tokenizer_utils
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
    from tinker_cookbook.rl.data_processing import trajectory_to_data
    from tinker_cookbook.rl.types import Trajectory, Transition

    from elt_rl import train as trainer
    from elt_rl.history import model_prompt
    from elt_rl.model_runtime import resolve_renderer

    # Pin the real upstream API; the local probe avoids tokenizer downloads and service calls.
    methods = ("build_generation_prompt", "build_supervised_example", "build_supervised_examples")
    assert all(
        signature(getattr(TmlV0Renderer, name)).parameters["effort"].default == 0.9
        for name in methods
    )

    class RendererProbe:
        tokenizer = object()

        def __init__(self):
            self.efforts = []

        def build_generation_prompt(self, messages, role="assistant", prefill=None, effort=0.9):
            self.efforts.append(effort)
            thinking = sum(
                len(part["text"])
                for message in messages
                if isinstance(message["content"], list)
                for part in message["content"]
                if part["type"] == "thinking"
            )
            return tinker.ModelInput.from_ints(
                [int(effort * 10), len(messages), 1] + [7] * thinking
            )

        def build_supervised_example(self, messages, train_on_what=None, effort=0.9):
            return self.build_generation_prompt(messages, effort=effort), [1]

        def build_supervised_examples(self, messages, train_on_what=None, effort=0.9):
            return [self.build_supervised_example(messages, train_on_what, effort=effort)]

        def get_stop_sequences(self):
            return [99]

        def parse_response(self, tokens):
            return {"role": "assistant", "content": "unchanged"}, True

    probe = RendererProbe()
    tokenizer_calls = []
    monkeypatch.setattr(
        tokenizer_utils,
        "get_tokenizer",
        lambda model: tokenizer_calls.append(model) or probe.tokenizer,
    )
    monkeypatch.setattr(renderers, "get_renderer", lambda *args: probe)
    name, fixed = resolve_renderer("thinkingmachines/Inkling-2-Mini", effort=0.0)
    assert name == "tml_v0" and fixed.tokenizer is probe.tokenizer
    assert fixed.get_stop_sequences() == [99]
    assert fixed.parse_response([1]) == probe.parse_response([1])
    history = [{"role": "user", "content": "initial"}]
    prompts = [model_prompt(history, fixed, 16)]
    history += [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "text": "x" * 32}, {"type": "text", "text": "action"}],
        },
        {"role": "user", "content": "next"},
    ]
    prompts.append(model_prompt(history, fixed, 16))  # full render, then thinking-elision rebuild
    assert probe.efforts == [0.0, 0.0, 0.0]
    assert history[1]["content"][0]["text"] == "x" * 32
    assert fixed.build_supervised_example(history)[0].to_ints()[0] == 0
    assert fixed.build_supervised_examples(history)[0][0].to_ints()[0] == 0
    assert fixed.build_generation_prompt(history, "assistant", None, 0.0).to_ints()[0] == 0
    with pytest.raises(ValueError, match="frozen renderer_effort"):
        fixed.build_generation_prompt(history, effort=0.9)

    transitions = [
        Transition(
            ob=prompt,
            ac=TokensWithLogprobs(
                tokens=[30 + i, 99], maybe_logprobs=[-0.5, -0.5], sequence_id=f"effort-node-{i}"
            ),
            reward=0,
            episode_done=i == 1,
        )
        for i, prompt in enumerate(prompts)
    ]
    data = trajectory_to_data(Trajectory(transitions, prompts[-1]), 1.0)
    prepared, _ = prepare_data(data)
    assert len(prepared) == 2
    for i, datum in enumerate(prepared):
        assert datum.model_input.to_ints()[: prompts[i].length] == prompts[i].to_ints()
        assert datum.model_input.to_ints()[0] == 0
        assert {span.sequence_id for span in datum.model_input_spans} == {f"effort-node-{i}"}
        assert {span.sequence_id for span in datum.loss_fn_input_spans} == {f"effort-node-{i}"}
    payload = tmp_path / "effort-payload.json.gz"
    save_training_payload(payload, data, prepared)
    assert all(
        row["datum"]["model_input"]["chunks"][0]["tokens"][0] == 0
        for row in json.loads(gzip.decompress(payload.read_bytes()))["datums"]
    )

    _, unconditioned = resolve_renderer("thinkingmachines/Inkling-2-Mini", "tml_v0", effort=None)
    assert (
        unconditioned is probe and unconditioned.build_generation_prompt(history).to_ints()[0] == 9
    )
    before = len(tokenizer_calls)
    with pytest.raises(ValueError, match="require the tml_v0 renderer"):
        resolve_renderer("thinkingmachines/Inkling-2-Mini", "qwen3", effort=0.0)
    assert len(tokenizer_calls) == before

    config = TrainConfig(
        "task.yaml",
        "run",
        model="thinkingmachines/Inkling-2-Mini",
        renderer="tml_v0",
        renderer_effort=0.0,
        task_spec_path="unit-contract.yaml",
        run_id="unit",
    )
    config.validate()
    for invalid in (True, "0.0", -0.1, 1.0, float("nan")):
        with pytest.raises(ValueError, match="renderer_effort"):
            replace(config, renderer_effort=invalid).validate()
    monkeypatch.setattr(trainer, "implementation_fingerprint", lambda: "local-probe-source")
    conditioning = {"renderer": "tml_v0", "effort": 0.0}
    assert trainer.fingerprint(config, task)["renderer_conditioning"] == conditioning
    assert trainer.implementation_record(config, task)["renderer_conditioning"] == conditioning
    assert trainer.fingerprint(config, task) != trainer.fingerprint(
        replace(config, renderer_effort=0.9), task
    )
