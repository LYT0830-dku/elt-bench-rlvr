"""Lossless local Cookbook group recovery; never samples or executes an environment."""

import gzip
import hashlib
import json
import os
import tempfile

import tinker
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition


def save_group(path, group):
    rows = []
    for trajectory in group.trajectories_G:
        rows.append(
            {
                "transitions": [
                    {
                        "ob": t.ob.model_dump(mode="json"),
                        "ac": {
                            "tokens": t.ac.tokens,
                            "maybe_logprobs": t.ac.logprobs,
                            "sequence_id": t.ac.sequence_id,
                            "stop_reason": t.ac.stop_reason,
                        },
                        "reward": t.reward,
                        "episode_done": t.episode_done,
                        "metrics": t.metrics,
                    }
                    for t in trajectory.transitions
                ],
                "final_ob": trajectory.final_ob.model_dump(mode="json"),
                "stop_reason": trajectory.stop_reason,
            }
        )
    value = {
        "format": "cookbook_group_v1",
        "trajectories": rows,
        "final_rewards": group.final_rewards_G,
        "metrics": group.metrics_G,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(json.dumps(value, allow_nan=False).encode(), mtime=0)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"artifact_path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def load_group(ref):
    from pathlib import Path

    data = Path(ref["artifact_path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("Cookbook group archive digest mismatch")
    value = json.loads(gzip.decompress(data))
    if value.get("format") != "cookbook_group_v1":
        raise ValueError("Unknown Cookbook group archive")
    trajectories = [
        Trajectory(
            transitions=[
                Transition(
                    ob=tinker.ModelInput.model_validate(t["ob"]),
                    ac=TokensWithLogprobs(**t["ac"]),
                    reward=t["reward"],
                    episode_done=t["episode_done"],
                    metrics=t["metrics"],
                )
                for t in row["transitions"]
            ],
            final_ob=tinker.ModelInput.model_validate(row["final_ob"]),
            stop_reason=row["stop_reason"],
        )
        for row in value["trajectories"]
    ]
    return TrajectoryGroup(trajectories, value["final_rewards"], value["metrics"])
