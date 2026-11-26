"""Explicit official success metrics, derived reward, and model request usage."""

import json
import math
import statistics


def zero_cell_results(task):
    """The fixed target cells of a known policy failure; no prediction values are invented."""
    from elt_rl.execution_scoring import load_table
    from elt_rl.rewards import _rows

    targets = {target.name: target for target in task.targets}
    cells = []
    for target in task.reward_targets:
        table = load_table(targets[target.name].expected, target, max_rows=task.max_reference_rows)
        for key in _rows(table, target, reference=True):
            for field in target.fields:
                cells.append(
                    {
                        "target": target.name,
                        "key": [str(k) for k in key],
                        "column": field.name,
                        "correct": False,
                    }
                )
    return cells


def cell_statistics(rows):
    if not rows or any(row.get("cell_accuracy") is None for row in rows):
        return {}
    values = [row["cell_accuracy"] for row in rows]
    masks = []
    for row in rows:
        entries = row.get("cell_results")
        if not entries:
            raise ValueError("Cell evaluation is missing its fixed target-cell outcomes")
        mask = {
            json.dumps([c["target"], c["key"], c["column"]], sort_keys=True): c for c in entries
        }
        if len(mask) != len(entries) or any(type(c["correct"]) is not bool for c in entries):
            raise ValueError("Duplicate or invalid target-cell outcomes")
        if masks and mask.keys() != masks[0].keys():
            raise ValueError("Evaluation rollouts have different target-cell identities")
        if sum(c["correct"] for c in entries) / len(entries) != row["cell_accuracy"]:
            raise ValueError("Cell accuracy differs from recorded binary outcomes")
        masks.append(mask)
    return {
        "mean_cell_accuracy": statistics.mean(values),
        "std_cell_accuracy": statistics.pstdev(values),
        "min_cell_accuracy": min(values),
        "max_cell_accuracy": max(values),
        "cell_accuracy_values": values,
        "per_cell_accuracy": [
            {
                "target": cell["target"],
                "key": cell["key"],
                "column": cell["column"],
                "correct_rollouts": sum(m[key]["correct"] for m in masks),
                "rollouts": len(rows),
                "accuracy": sum(m[key]["correct"] for m in masks) / len(rows),
            }
            for key, cell in masks[0].items()
        ],
    }


def request_events(events):
    for event in events:
        previous = event.get("previous_gateway_request")
        if previous:
            yield from request_events([previous])
        yield event


def episode_metrics(raw):
    report = raw.get("report") or {}
    samples = list(request_events(raw.get("sampling", [])))
    confirmed = [event for event in samples if event.get("status") == "sampled"]
    return {
        "graded": report.get("official_score") is not None,
        "official_srdel": report.get("official_srdel"),
        "official_srdt": report.get("official_srdt", report.get("official_success")),
        "correct_target_models": report.get("correct_target_models"),
        "required_target_models": report.get("required_target_models"),
        "matched_target_columns": report.get(
            "matched_target_columns", len(report.get("matched", []))
        ),
        **{
            key: report.get(key)
            for key in ("cell_accuracy", "correct_cells", "required_cells", "cell_results")
        },
        "tool_calls": len(raw.get("trace", [])),
        "input_tokens": sum(e.get("input_tokens", 0) for e in confirmed),
        "output_tokens": sum(e.get("output_tokens", 0) for e in confirmed),
        "cached_input_tokens": sum(e.get("prompt_cache_hit_tokens", 0) for e in confirmed),
        "sampling_requests": len(samples),
        "uncertain_sampling_requests": sum(
            e.get("status") == "failed_or_cancelled_usage_uncertain" for e in samples
        ),
        "wall_clock_seconds": raw.get("wall_clock_seconds"),
    }


def group_metrics(rows):
    rewards = [row["reward"] for row in rows]
    counts = [(r.get("correct_target_models"), r.get("required_target_models")) for r in rows]
    if all(correct is not None and total is not None for correct, total in counts):
        if any(
            type(correct) is not int
            or type(total) is not int
            or not 0 <= correct <= total
            or total <= 0
            for correct, total in counts
        ):
            raise ValueError("Invalid official target-model counts")
        correct_models, required_models = map(sum, zip(*counts, strict=True))
        srdt = correct_models / required_models
    else:
        # Legacy fixed-task cohorts and known policy failures lack count fields.
        # Preserve fractional observations; None policy outcomes contribute zero.
        values = [
            0.0 if r.get("official_srdt") is None else float(r["official_srdt"]) for r in rows
        ]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Invalid official SRDT observation")
        srdt = sum(values) / len(rows)
        correct_models = required_models = None
    return {
        **cell_statistics(rows),
        "reward_std": statistics.pstdev(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "official_srdel": sum(bool(r.get("official_srdel")) for r in rows) / len(rows),
        "official_srdt": srdt,
        "correct_target_models": correct_models,
        "required_target_models": required_models,
        "end_to_end_success_rate": sum(bool(r.get("success")) for r in rows) / len(rows),
        "matched_columns": [r.get("matched_target_columns", 0) for r in rows],
        "graded_episodes": sum(bool(r.get("graded")) for r in rows),
        "tool_calls": sum(r.get("tool_calls", 0) for r in rows),
        "input_tokens": sum(r.get("input_tokens", 0) for r in rows),
        "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
    }
