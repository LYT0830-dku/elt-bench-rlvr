"""Pluggable reward functions returning one standard trusted result contract."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Protocol

from elt_rl.registry import Registry
from elt_rl.rewards import (
    REWARD_VERSIONS,
    RewardConfig,
    RewardContext,
    RewardInfrastructureError,
    score_execution,
)


class RewardFunction(Protocol):
    def score(self, context: RewardContext, configuration: dict) -> dict: ...


class ExecutionReward:
    def score(self, context, configuration):
        configuration = configuration or RewardConfig().configuration()
        return score_execution(context, RewardConfig.from_dict(configuration))


class RewardRegistry(Registry[RewardFunction]):
    def __init__(self):
        super().__init__("reward function")

    def score(self, context, configuration):
        if (
            context.infrastructure_error
            or context.stage1_passed is None
            or context.submission_valid is None
        ):
            raise RewardInfrastructureError(
                context.infrastructure_error or "Unknown execution gate"
            )
        result = self.get(configuration.get("type", "pure_cell")).score(context, configuration)
        if not context.stage1_passed or not context.submission_valid:
            if result.get("reward") != 0:
                raise RewardInfrastructureError("Reward plugin bypassed the execution gate")
        result.setdefault("reward_type", configuration.get("type", "pure_cell"))
        result.setdefault("version", configuration.get("version", "plugin_v1"))
        result.setdefault(
            "config_sha256",
            hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
        )
        for name in ("reward", "cell_accuracy", "column_score", "cell_score", "value_score"):
            if name != "reward" and name not in result:
                continue
            value = result.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise RewardInfrastructureError(
                    "Reward implementation violated result contract: " + name
                )
        if not isinstance(result.get("targets", {}), dict) or not result.get("config_sha256"):
            raise RewardInfrastructureError("Reward implementation omitted its audit contract")
        return result


REWARDS = RewardRegistry()
for _name in REWARD_VERSIONS:
    REWARDS.register(_name, ExecutionReward())
