"""Trusted source asset decoding and schema checks before service initialization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from elt_rl.errors import IntegrityError


def source_rows(task) -> dict[str, list[dict]]:
    result = {}
    for source in task.sources:
        path = Path(source.options["path"])
        if not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
            raise IntegrityError("Source asset exceeds the file budget or is unavailable")
        if path.suffix == ".jsonl":
            data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            with path.open(newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None or len({k.lower() for k in reader.fieldnames}) != len(
                    reader.fieldnames
                ):
                    raise IntegrityError("Source header contains duplicate or missing columns")
                data = list(reader)
        if len(data) > 2_000_000:
            raise IntegrityError("Source asset exceeds the row budget")
        normalized = []
        for row in data:
            if not isinstance(row, dict) or any(not isinstance(k, str) for k in row):
                raise IntegrityError("Source record must be a named-column object")
            keys = [k.lower() for k in row]
            if len(set(keys)) != len(keys):
                raise IntegrityError("Source record contains duplicate case-insensitive columns")
            folded = {k.lower(): v for k, v in row.items()}
            if not set(source.columns) <= set(folded):
                raise IntegrityError("Source record is missing required columns")
            normalized.append(folded)
        result[source.name] = normalized
    return result
