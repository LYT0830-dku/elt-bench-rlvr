"""ELT environments and a Tinker training recipe."""

from pathlib import Path

__version__ = "0.6.1"


def example_dir() -> Path:
    """Location of the installed retail task and example pipeline."""
    return Path(__file__).parent / "fixtures" / "retail"
