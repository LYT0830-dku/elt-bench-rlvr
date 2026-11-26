from dataclasses import replace

import pytest
import tinker

from elt_rl.train import prepare_data


def test_prepare_data_preserves_masks_and_rejects_observation_advantages():
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[2, 3], dtype="int64", shape=[2]),
            "logprobs": tinker.TensorData(data=[0, -0.5], dtype="float32", shape=[2]),
            "advantages": tinker.TensorData(data=[0, 0.1], dtype="float32", shape=[2]),
            "mask": tinker.TensorData(data=[0, 1], dtype="float32", shape=[2]),
        },
    )
    sent, counts = prepare_data([datum])
    assert set(sent[0].loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
    assert list(datum.loss_fn_inputs["mask"].data) == [0, 1]
    assert counts["nonzero_advantage_tokens"] == 1
    bad = replace(
        datum,
        loss_fn_inputs={
            **datum.loss_fn_inputs,
            "advantages": tinker.TensorData(data=[0.1, 0.1], dtype="float32", shape=[2]),
        },
    )
    with pytest.raises(ValueError, match="Observation tokens"):
        prepare_data([bad])
