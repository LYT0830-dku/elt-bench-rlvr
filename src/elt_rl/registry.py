"""Explicit, host-controlled component registration without dynamic agent plugins."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class UnsupportedCapability(ValueError):
    """A requested configuration is unsupported; fail before dispatch or charges."""


class Registry(Generic[T]):
    def __init__(self, component: str):
        self.component = component
        self._entries: dict[str, T] = {}

    def register(self, name: str, implementation: T):
        if not isinstance(name, str) or not name or name in self._entries:
            raise ValueError(f"Invalid/duplicate {self.component} registration: {name!r}")
        self._entries[name] = implementation
        return implementation

    def get(self, name: str) -> T:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise UnsupportedCapability(
                f"Unsupported {self.component}: {name!r}; registered: {', '.join(self.names())}"
            ) from exc

    def names(self):
        return tuple(sorted(self._entries))
