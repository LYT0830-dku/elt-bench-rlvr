"""Seeded, bounded model sampling over the pinned Cookbook rollout interfaces."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from inspect import signature

import tinker
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.rollout_runner import run_rollout
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy

from elt_rl.sampling import validate_renderer_effort, validate_sampling


@dataclass(frozen=True)
class FixedEffortRenderer:
    """Bind one conditioning value while delegating all rendering to Cookbook."""

    renderer: object
    effort: float

    def __getattr__(self, name):
        return getattr(self.renderer, name)

    def _render(self, method, *args, **kwargs):
        build = getattr(self.renderer, method)
        bound = signature(build).bind(*args, **kwargs)
        if "effort" in bound.arguments and bound.arguments["effort"] != self.effort:
            raise ValueError("Cannot override the frozen renderer_effort")
        bound.arguments["effort"] = self.effort
        return build(*bound.args, **bound.kwargs)

    def build_generation_prompt(self, *args, **kwargs):
        return self._render("build_generation_prompt", *args, **kwargs)

    def build_supervised_example(self, *args, **kwargs):
        return self._render("build_supervised_example", *args, **kwargs)

    def build_supervised_examples(self, *args, **kwargs):
        return self._render("build_supervised_examples", *args, **kwargs)


def resolve_renderer(model: str, name: str | None = None, effort: float | None = None):
    from elt_rl.model_registry import MODELS

    validate_renderer_effort(effort)
    recommended, renderer = MODELS.get(model).renderer(name, effort)
    if effort is not None:
        renderer = FixedEffortRenderer(renderer, effort)
    return recommended, renderer


@dataclass
class TokenBudget:
    """Reserve prompt + maximum completion before dispatch, including concurrency.

    These are conservative application token limits, not a provider dollar cap.
    Unknown usage after a failed/cancelled request keeps its full reservation.
    """

    limit: int
    input_tokens: int = 0
    output_tokens: int = 0
    reserved: int = 0
    exhausted: bool = False
    deadline_epoch: float | None = None

    def check_deadline(self):
        from elt_rl.control import RunDeadlineExceeded

        if self.deadline_epoch is not None and time.time() >= self.deadline_epoch:
            raise RunDeadlineExceeded("Global run deadline reached before dispatch")

    def reserve(self, tokens: int) -> bool:
        self.check_deadline()
        if self.input_tokens + self.output_tokens + self.reserved + tokens > self.limit:
            self.exhausted = True
            return False
        self.reserved += tokens
        return True

    def settle(self, reservation: int, inputs: int, outputs: int):
        self.reserved -= reservation
        self.input_tokens += inputs
        self.output_tokens += outputs

    def public(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reserved_or_uncertain_tokens": self.reserved,
            "sampling_token_limit": self.limit,
            "exhausted": self.exhausted,
        }


class ModelPolicy(TokenCompleter):
    def __init__(
        self,
        sampler,
        budget: TokenBudget,
        *,
        max_tokens: int,
        context_tokens: int,
        max_episode_tokens: int,
        temperature: float,
        seed: int,
        top_p: float = 1.0,
        top_k: int = -1,
        progress=None,
    ):
        validate_sampling(temperature, top_p, top_k)
        self.sampler = sampler
        self.budget = budget
        self.max_tokens = max_tokens
        self.context_tokens = context_tokens
        self.max_episode_tokens = max_episode_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.progress = progress
        self.events: list[dict] = []
        self.generated = 0
        self.record_state = None

    def for_episode(self, index: int):
        return ModelPolicy(
            self.sampler,
            self.budget,
            max_tokens=self.max_tokens,
            context_tokens=self.context_tokens,
            max_episode_tokens=self.max_episode_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=(self.seed + index * 1009) % (2**31),
            progress=(
                (lambda event: self.progress({"episode_index": index, **event}))
                if self.progress is not None
                else None
            ),
        )

    async def __call__(self, model_input, stop, *, max_tokens=None):
        cap = min(
            self.max_tokens,
            self.context_tokens - model_input.length,
            self.max_episode_tokens - self.generated,
            max_tokens if max_tokens is not None else self.max_tokens,
        )
        seed = (self.seed + len(self.events)) % (2**31)
        try:
            return await self._sample_once(model_input, stop, cap, seed)
        except tinker.InternalServerError as exc:
            if exc.status_code != 520 or not self.events:
                raise
            previous = self.events[-1]
            if (
                previous.get("http_status") != 520
                or previous.get("status") != "failed_or_cancelled_usage_uncertain"
                or previous.get("seed") != seed
            ):
                raise
            # The historical gateway explicitly asks for at least 60 seconds.
            # Keep the failed request's token reservation; retry has its own reservation.
            previous["retry_wait_seconds"] = 60
            await asyncio.sleep(60)
            self.budget.check_deadline()
        return await self._sample_once(model_input, stop, cap, seed, previous_request=previous)

    async def _sample_once(self, model_input, stop, cap, seed, *, previous_request=None):
        def record_event(event):
            if previous_request is None:
                self.events.append(event)
            else:
                event["previous_gateway_request"] = previous_request
                self.events[-1] = event

        reservation = model_input.length + cap
        if cap <= 0 or not self.budget.reserve(reservation):
            record_event({"stop_reason": "run_token_budget" if cap > 0 else "episode_token_budget"})
            return TokensWithLogprobs(tokens=[], maybe_logprobs=[], stop_reason="length")
        event = {
            "seed": seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "input_tokens": model_input.length,
            "max_output_tokens": cap,
            "status": "in_flight",
        }
        record_event(event)
        if self.record_state is not None:
            self.record_state()
        if self.progress is not None:
            self.progress(dict(event))
        sampling_started = time.perf_counter()
        try:
            result = await self.sampler.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    stop=stop,
                    max_tokens=cap,
                    temperature=self.temperature,
                    seed=seed,
                    top_p=self.top_p,
                    top_k=self.top_k,
                ),
            )
            if not result.sequences:
                raise RuntimeError("Sampler returned no sequences")
            sequence = result.sequences[0]
            if sequence.logprobs is None or len(sequence.logprobs) != len(sequence.tokens):
                raise RuntimeError("Sampler returned missing or misaligned token log probabilities")
            if len(sequence.tokens) > cap or not all(math.isfinite(x) for x in sequence.logprobs):
                raise RuntimeError("Sampler returned invalid tokens/log probabilities")
            cached_input_tokens = getattr(result, "prompt_cache_hit_tokens", None)
            if (
                type(cached_input_tokens) is not int
                or not 0 <= cached_input_tokens <= model_input.length
            ):
                raise RuntimeError("Sampler returned invalid prompt cache hit token count")
        except BaseException as exc:
            if isinstance(exc, tinker.APIStatusError):
                event["http_status"] = exc.status_code
            event["status"] = "failed_or_cancelled_usage_uncertain"
            event["sampling_seconds"] = time.perf_counter() - sampling_started
            if self.record_state is not None:
                self.record_state()
            if self.progress is not None:
                self.progress(dict(event))
            raise
        self.budget.settle(reservation, model_input.length, len(sequence.tokens))
        self.generated += len(sequence.tokens)
        event.update(
            status="sampled",
            sampling_seconds=time.perf_counter() - sampling_started,
            output_tokens=len(sequence.tokens),
            prompt_cache_hit_tokens=cached_input_tokens,
            tokens=sequence.tokens,
            logprobs=sequence.logprobs,
            stop_reason=sequence.stop_reason,
            sequence_id=getattr(sequence, "sequence_id", None),
        )
        if self.record_state is not None:
            self.record_state()
        if self.progress is not None:
            self.progress(
                {k: v for k, v in event.items() if k not in {"tokens", "logprobs", "sequence_id"}}
            )
        return TokensWithLogprobs(
            tokens=sequence.tokens,
            maybe_logprobs=sequence.logprobs,
            stop_reason=sequence.stop_reason,
            sequence_id=getattr(sequence, "sequence_id", None),
            prompt_cache_hit_tokens=cached_input_tokens,
        )


class SeededGroupStrategy(RolloutStrategy):
    """One deterministic seed stream per episode, with sibling cancellation/join."""

    def __init__(self, concurrency=1):
        self.concurrency = concurrency

    async def execute(self, env_group_builder, policy):
        semaphore = asyncio.Semaphore(self.concurrency)
        envs = await env_group_builder.make_envs()

        async def run_episode(index, env):
            from tinker_cookbook.rl.rollout_limits import DEFAULT_LIMIT_STOP_REASONS

            bound = policy.for_episode(getattr(env, "episode_index", index))
            bound.budget.check_deadline()
            env.core.dispatch_check = bound.budget.check_deadline
            env.core.run_deadline = None
            if bound.budget.deadline_epoch is not None:
                env.core.run_deadline = time.monotonic() + max(
                    0, bound.budget.deadline_epoch - time.time()
                )
            env.sampling_trace = bound.events
            env.core.sampling = bound.events
            bound.record_state = env.core.persist
            try:
                trajectory = await run_rollout(bound, env)
            except BaseException as exc:
                env.core.termination = f"infrastructure_{type(exc).__name__}"
                env.core.persist()
                raise
            reason = trajectory.stop_reason
            env.runner_stop_reason = getattr(reason, "value", str(reason)) if reason else None
            tokenizer = getattr(env.renderer, "tokenizer", None)
            if tokenizer is not None:
                for event in bound.events:
                    if "tokens" in event:
                        event["text"] = tokenizer.decode(event["tokens"])
            if reason in DEFAULT_LIMIT_STOP_REASONS and env.core.report:
                env.core.report.update(
                    reward=0, training_reward=0, success=False, termination=env.runner_stop_reason
                )
                env.core.termination = env.runner_stop_reason
            if not env.core.done:
                env.core.done = True
                env.core.termination = env.runner_stop_reason or "runner_terminated"
            env.core.persist()
            return trajectory

        async def episode(index, env):
            from elt_rl.cookbook_env import run_blocking

            async with semaphore:
                try:
                    return await run_episode(index, env)
                finally:
                    await run_blocking(env.core.close)

        tasks = [asyncio.create_task(episode(i, env)) for i, env in enumerate(envs)]
        try:
            trajectories = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return RolloutResult(trajectories=trajectories, envs=envs, errors=[])
