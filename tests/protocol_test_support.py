"""Offline typed service results; no model or warehouse calls."""

from types import SimpleNamespace


class Result(SimpleNamespace):
    def model_dump(self, **kwargs):
        return vars(self)


def future(value):
    async def result_async():
        return value

    return SimpleNamespace(result_async=result_async)
