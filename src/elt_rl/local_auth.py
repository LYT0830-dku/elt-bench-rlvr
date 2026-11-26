"""Interactive local credential setup. Keys never go in CLI arguments or output."""

from __future__ import annotations

import getpass
import os
import sys
import tempfile
from pathlib import Path


def credential_file() -> Path:
    root = Path(__file__).resolve().parents[2]
    return (root if (root / "pyproject.toml").is_file() else Path.cwd()) / ".env"


def load_local_key(path: Path | None = None) -> bool:
    """An explicit environment variable takes precedence over the project file."""
    if os.environ.get("TINKER_API_KEY"):
        return True
    path = path or credential_file()
    if not path.is_file():
        return False
    for line in path.read_text().splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "TINKER_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if value:
                os.environ["TINKER_API_KEY"] = value
                return True
    return False


def save_local_key(key: str, path: Path | None = None) -> Path:
    path = path or credential_file()
    key = key.strip()
    if not key or len(key) > 8192 or any(c.isspace() for c in key):
        raise ValueError("Invalid key format; paste the API key without spaces or newlines")
    if path.is_symlink():
        raise ValueError("Credential path must not be a symbolic link")
    previous = path.read_text().splitlines() if path.exists() else []
    lines = [line for line in previous if line.partition("=")[0].strip() != "TINKER_API_KEY"]
    lines.append("TINKER_API_KEY=" + key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tinker-key-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as file:
            file.write("\n".join(lines) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def configure_interactively() -> Path:
    if not sys.stdin.isatty():
        raise RuntimeError("Run configure-tinker in your own interactive terminal")
    key = getpass.getpass("Paste Tinker API key (input is hidden): ")
    return save_local_key(key)
