"""SDK-independent validation of bounded stochastic sampling settings."""

import math


def validate_renderer_effort(effort: float | None) -> None:
    if effort is not None and (
        type(effort) not in (int, float) or not math.isfinite(effort) or not 0 <= effort < 1
    ):
        raise ValueError("renderer_effort must be a finite number in [0, 1) or None")


def validate_sampling(temperature: float, top_p: float, top_k: int) -> None:
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive for on-policy training")
    if type(top_p) not in (int, float) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if type(top_k) is not int or (top_k != -1 and top_k < 1):
        raise ValueError("top_k must be -1 (disabled) or a positive integer")
