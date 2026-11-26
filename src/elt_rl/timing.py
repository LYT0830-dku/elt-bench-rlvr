"""Small, host-side phase timings; phases can overlap or contain subphases."""

from contextlib import contextmanager
from time import perf_counter


class Timings:
    def __init__(self):
        self.events: list[dict] = []

    @contextmanager
    def measure(self, phase: str, **labels):
        start = perf_counter()
        status = "ok"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            self.events.append(
                {"phase": phase, "seconds": perf_counter() - start, "status": status, **labels}
            )
