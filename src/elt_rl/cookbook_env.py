"""Thin adapters over real, pinned Tinker Cookbook interfaces (no service calls)."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import message_to_jsonable
from tinker_cookbook.rl.rollout_limits import RolloutLimits
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, InitialObservationOverflow, StepResult

from elt_rl.artifacts import write_json
from elt_rl.environment import DEFAULT_EPISODE_SECONDS, ELTEnvironment
from elt_rl.errors import ActionError
from elt_rl.history import model_prompt
from elt_rl.task import Task


async def run_blocking(function, *args, completed=None):
    """Join an uncancellable worker even after repeated parent cancellation.

    Register newly allocated resources synchronously before re-raising cancellation.
    A thread cannot be killed by cancelling its asyncio wrapper.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
    result = worker.result()
    if completed is not None:
        completed(result)
    if cancelled:
        raise asyncio.CancelledError
    return result


class CookbookELTEnv(Env):
    def __init__(
        self,
        core: ELTEnvironment,
        renderer,
        context_tokens: int = 64000,
        model_adapter=None,
    ):
        if model_adapter is None and getattr(renderer, "renderer_name", None) != "tml_v0":
            raise ValueError("native_tools_v1 currently requires the tml_v0 renderer")
        from elt_rl.model_registry import TMLToolCodec

        self.codec = model_adapter.codec if model_adapter is not None else TMLToolCodec()
        if model_adapter is not None:
            model_adapter.validate(renderer=getattr(renderer, "renderer_name", None))
        self.core = core
        self.renderer = renderer
        self.context_tokens = context_tokens
        self.messages: list = []
        self.elided_history: set = set()
        self.sampling_trace: list[dict] = []
        self.runner_stop_reason: str | None = None

    async def initial_observation(self):
        self.messages = self.codec.initial_messages(
            await run_blocking(self.core.initial), self.renderer
        )
        prompt = model_prompt(
            self.messages, self.renderer, self.context_tokens, elided=self.elided_history
        )
        if prompt.length > self.context_tokens:
            self.core.done = True
            self.core.termination = "initial_context_overflow"
            return InitialObservationOverflow(reward=0, metrics={"stop/max_tokens": 1.0})
        return prompt, self.renderer.get_stop_sequences()

    async def step(self, action, *, extra=None):
        if extra and extra.get("stop_reason") == "length":
            self.core.done = True
            self.core.trace.append(
                {"truncated_response_tokens": list(action), "sampling": self.sampling_trace[-1:]}
            )
            return self._terminal("max_tokens")
        message, termination = self.renderer.parse_response(action)
        text = renderers.get_text_content(message)
        self.messages.append(message)
        if self.sampling_trace:
            self.sampling_trace[-1]["parsed_message"] = message_to_jsonable(message)
            self.sampling_trace[-1]["parse_termination"] = termination.value
        try:
            tool_action = self.codec.parse_action(message, termination, text)
        except ActionError as exc:
            reason = str(exc)
            result = await run_blocking(lambda: self.core.step(text, format_error=reason))
        else:
            result = await run_blocking(self.core.step, tool_action)
        self.messages.extend(self.codec.observations(message, result.observation))
        prompt = model_prompt(
            self.messages, self.renderer, self.context_tokens, elided=self.elided_history
        )
        if prompt.length > self.context_tokens and not result.done:
            self.core.done = True
            return self._terminal("context_overflow")
        reason = "completed"
        if result.done and not self.core.report:
            reason = {
                "Maximum turns reached": "max_turns",
                "Episode time budget exceeded": "rollout_timeout",
                "Repeated identical invalid JSON": "invalid_action_json",
            }.get(self.core.termination, "environment_terminated")
        return StepResult(
            reward=result.reward,
            episode_done=result.done,
            next_observation=prompt,
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics={
                "elt/turns": self.core.turns,
                **({f"stop/{reason}": 1.0} if result.done else {}),
                **({"elt/success": float(self.core.report["success"])} if self.core.report else {}),
            },
        )

    def _terminal(self, reason):
        self.core.termination = reason
        return StepResult(
            reward=0,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[],
            metrics={f"stop/{reason}": 1.0},
        )


@dataclass
class ELTGroupBuilder(EnvGroupBuilder):
    task_path: str
    renderer: object
    destination: str = "duckdb"
    group_size: int = 5
    max_turns: int = 200
    max_seconds: float | None = DEFAULT_EPISODE_SECONDS
    context_tokens: int = 64000
    artifact_dir: str | None = None
    artifact_prefix: str | None = None
    max_episode_tokens: int = 262144
    sampling_timeout_seconds: float | None = None
    allow_remote: bool = False
    episode_indices: tuple[int, ...] | None = None
    task_spec_path: str | None = None
    reward_config: dict | None = None
    model_id: str | None = None
    run_id: str | None = None
    policy_checkpoint: dict | None = None
    _envs: list = field(default_factory=list, init=False, repr=False)

    async def make_envs(self):
        if self._envs:
            raise RuntimeError("Builder is single-use")
        episode_indices = self.episode_indices
        if episode_indices is None:
            episode_indices = tuple(range(self.group_size))
        elif (
            len(episode_indices) != self.group_size
            or any(type(index) is not int or index < 0 for index in episode_indices)
            or len(set(episode_indices)) != len(episode_indices)
        ):
            raise ValueError(
                "episode_indices must contain one unique nonnegative index per episode"
            )
        if self.artifact_dir and self.artifact_prefix is None:
            self.artifact_prefix = f"episode-{uuid.uuid4().hex}"

        def register(core):
            from elt_rl.model_registry import MODELS

            adapter = MODELS.get(self.model_id) if self.model_id else None
            env = CookbookELTEnv(core, self.renderer, self.context_tokens, model_adapter=adapter)
            env.episode_index = episode_indices[len(self._envs)]
            self._envs.append(env)
            env.rollout_limits = RolloutLimits(
                max_turns=self.max_turns,
                max_sampled_tokens=self.max_episode_tokens,
                rollout_timeout_seconds=self.max_seconds,
                sampling_turn_timeout_seconds=self.sampling_timeout_seconds,
            )

        try:
            from elt_rl.task_spec import load_task_spec

            for episode_index in episode_indices:
                path = None
                if self.artifact_dir:
                    name = self.artifact_prefix or "episode"
                    path = Path(self.artifact_dir) / f"{name}-{episode_index:03d}.json"
                await run_blocking(
                    lambda path=path: ELTEnvironment(
                        load_task_spec(self.task_spec_path)
                        if self.task_spec_path
                        else Task.load(self.task_path),
                        self.destination,
                        self.max_turns,
                        self.max_seconds,
                        path,
                        allow_remote=self.allow_remote,
                        reward_config=self.reward_config,
                        run_id=self.run_id,
                        model_id=self.model_id,
                        policy_checkpoint=self.policy_checkpoint,
                    ),
                    completed=register,
                )
        except BaseException:
            await self.cleanup()
            raise
        return self._envs

    async def cleanup(self):
        failures = []
        cancelled = False
        for local_index, env in enumerate(self._envs):
            index = getattr(env, "episode_index", local_index)
            try:
                await run_blocking(env.core.close)
            except asyncio.CancelledError:
                cancelled = True
            except Exception as exc:
                failures.append(f"Warehouse {index}: {str(exc)[:1000]}")
            try:
                if self.artifact_dir:
                    directory = Path(self.artifact_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    name = self.artifact_prefix or f"episode-{uuid.uuid4().hex}"
                    write_json(
                        directory / f"{name}-{index:03d}.json",
                        {
                            **env.core.artifact(),
                            "episode_id": f"{name}-{index:03d}",
                            "runner_stop_reason": env.runner_stop_reason,
                            "sampling": env.sampling_trace,
                        },
                    )
            except Exception as exc:
                failures.append(f"Artifact {index}: {type(exc).__name__}")
        if cancelled:
            raise asyncio.CancelledError
        if failures:
            raise RuntimeError(f"Environment cleanup failed: {failures}")

    def logging_tags(self):
        return ["elt", self.destination]
