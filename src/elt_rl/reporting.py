"""Reproducible CSV, figures and analysis from recorded execution evidence only."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from elt_rl.artifacts import write_json
from elt_rl.metrics import episode_metrics


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p, denominator = successes / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def write_csv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def render_run_report(out, state=None):
    out = Path(out)
    state = state or json.loads((out / "protocol-state.json").read_text())
    directory = out / "report"
    directory.mkdir(exist_ok=True)
    batches = state.get("batches", [])
    reward_configuration = state.get("config", {}).get("reward") or {}
    pure_cell = reward_configuration.get("type") == "pure_cell"
    # Include an executed but not locally finalized batch journal without duplicating it.
    for path in sorted(out.glob("optimization-batch-*.json")):
        event = json.loads(path.read_text())
        if event.get("batch") not in {b["batch"] for b in batches}:
            batches = [*batches, event]
    training = {
        f"train-batch-{batch['batch']:04d}-{i:03d}": {
            "advantage": advantage,
            "gradient_computed": bool(batch.get("forward_backward_result")),
            "optimizer_applied": batch.get("status") == "effective_update_confirmed",
        }
        for batch in batches
        for i, advantage in enumerate(batch.get("advantages", []))
    }
    selected_cells = {
        row["episode_id"]: {
            key: row.get(key) for key in ("cell_accuracy", "correct_cells", "required_cells")
        }
        for cohort in [
            *(b.get("rollout", {}) for b in batches),
            *state.get("process_evaluations", []),
            *state.get("final_evaluations", {}).values(),
        ]
        for row in cohort.get("episodes", [])
        if row.get("cell_accuracy") is not None
    }
    rows = []
    for path in sorted((out / "episodes").glob("*.json")):
        raw = json.loads(path.read_text())
        report = raw.get("report") or {}
        samples = raw.get("sampling", [])
        identifier = raw.get("episode_id", path.stem)
        rows.append(
            {
                "episode_id": identifier,
                "phase": identifier.rsplit("-", 1)[0],
                "started": bool(samples),
                "done": raw.get("done"),
                "termination": raw.get("termination"),
                "reward": report.get(
                    "training_reward",
                    0
                    if raw.get("done")
                    and not str(raw.get("termination", "")).startswith("infrastructure")
                    else None,
                ),
                "pipeline_success": report.get("success", False),
                "task_id": raw.get("task"),
                "model_id": raw.get("model_id"),
                **{
                    key: report.get(key)
                    for key in (
                        "official_score",
                        "cell_accuracy",
                        "correct_cells",
                        "required_cells",
                        "column_score",
                        "cell_score",
                        "value_score",
                    )
                },
                "seed": samples[0].get("seed") if samples else None,
                **episode_metrics(raw),
                **selected_cells.get(identifier, {}),
                **training.get(
                    identifier,
                    {
                        "advantage": None,
                        "gradient_computed": False,
                        "optimizer_applied": False,
                    },
                ),
            }
        )
    columns = [
        "episode_id",
        "phase",
        "seed",
        "started",
        "done",
        "termination",
        "reward",
        "task_id",
        "model_id",
        "official_score",
        "cell_accuracy",
        "correct_cells",
        "required_cells",
        "column_score",
        "cell_score",
        "value_score",
        "advantage",
        "gradient_computed",
        "optimizer_applied",
        "official_srdel",
        "official_srdt",
        "correct_target_models",
        "required_target_models",
        "pipeline_success",
        "matched_target_columns",
        "graded",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "sampling_requests",
        "uncertain_sampling_requests",
        "wall_clock_seconds",
    ]
    write_csv(directory / "rollouts.csv", rows, columns)
    process = state.get("process_evaluations", [])
    final = state.get("final_evaluations", {})
    write_csv(
        directory / "evaluation.csv",
        [{"model": f"batch-{r['batch']}", **r} for r in process]
        + [{"model": name, **r} for name, r in final.items()],
        [
            "model",
            "batch",
            "updates",
            "n",
            "mean_reward",
            "mean_cell_accuracy",
            "std_cell_accuracy",
            "min_cell_accuracy",
            "max_cell_accuracy",
            "mean_official_score",
            "mean_cell_score",
            "mean_value_score",
            "reward_std",
            "official_srdel",
            "official_srdt",
            "end_to_end_success_rate",
            "seconds",
        ],
    )
    evaluations = [{"model": f"batch-{r['batch']}", **r} for r in process] + [
        {"model": name, **r} for name, r in final.items()
    ]
    write_csv(
        directory / "evaluation-cells.csv",
        [
            {"model": result["model"], **cell}
            for result in evaluations
            for cell in result.get("per_cell_accuracy", [])
        ],
        ["model", "target", "key", "column", "correct_rollouts", "rollouts", "accuracy"],
    )
    write_csv(
        directory / "checkpoint-performance.csv",
        evaluations,
        [
            "model",
            "checkpoint",
            "batch",
            "updates",
            "n",
            "mean_cell_accuracy",
            "std_cell_accuracy",
            "min_cell_accuracy",
            "max_cell_accuracy",
            "mean_reward",
            "official_srdel",
            "official_srdt",
        ],
    )
    summary = {
        "reward_configuration": reward_configuration,
        "primary_metric": "cell_accuracy" if pure_cell else "historical_reward",
        "status": state["status"],
        "attempted_batches": state["attempted_batches"],
        "completed_batches": state["completed_batches"],
        "effective_updates": state["updates"],
        "rollouts_started": sum(r["started"] for r in rows),
        "episode_artifacts": len(rows),
        "final_selection": state.get("final_selection"),
        "final": {},
    }
    for name, result in final.items():
        summary["final"][name] = {
            k: result.get(k)
            for k in [
                "n",
                "mean_reward",
                "mean_cell_accuracy",
                "std_cell_accuracy",
                "min_cell_accuracy",
                "max_cell_accuracy",
                "mean_official_score",
                "mean_cell_score",
                "mean_value_score",
                "reward_std",
                "official_srdel",
                "official_srdt",
                "end_to_end_success_rate",
            ]
        }
        summary["final"][name]["success_rate_wilson95"] = wilson(result["successes"], result["n"])
    if set(final) == {"base", "final"}:
        base, trained = final["base"], final["final"]
        if base["episode_seed_set"] != trained["episode_seed_set"]:
            raise ValueError("Final models were not evaluated on the same seed set")
        summary["paired_reward_differences"] = [
            b - a for a, b in zip(base["rewards"], trained["rewards"], strict=True)
        ]
        summary["final_minus_base_mean_reward"] = trained["mean_reward"] - base["mean_reward"]
        if "mean_cell_accuracy" in base and "mean_cell_accuracy" in trained:
            summary["final_minus_base_mean_cell_accuracy"] = (
                trained["mean_cell_accuracy"] - base["mean_cell_accuracy"]
            )
    write_json(directory / "results.json", summary)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    charts = []
    score_label = "Cell accuracy" if pure_cell else "Historical execution reward"

    def plot(name, title, draw, x="", y=""):
        figure = Figure(figsize=(6.5, 4), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        draw(axes)
        axes.set(title=title, xlabel=x, ylabel=y)
        axes.grid(alpha=0.2)
        for extension in ("png", "svg"):
            figure.savefig(directory / f"{name}.{extension}", dpi=180)
        charts.append(name)

    observed = [b for b in batches if b.get("rewards")]
    if observed:
        plot(
            "training_mean_reward",
            "Training reward by effective update",
            lambda a: a.plot(
                [b.get("updates", b["updates_before"]) for b in observed],
                [b["mean_reward"] for b in observed],
                "o-",
            ),
            "Effective optimization step (ties include zero-advantage groups)",
            "Mean " + score_label.lower(),
        )
        plot(
            "group_reward_distribution",
            "Reward distribution by training batch",
            lambda a: a.boxplot(
                [b["rewards"] for b in observed], tick_labels=[str(b["batch"]) for b in observed]
            ),
            "Training batch",
            score_label,
        )
        plot(
            "advantage_distribution",
            "Recorded group-relative advantages",
            lambda a: a.boxplot(
                [b["advantages"] for b in observed], tick_labels=[str(b["batch"]) for b in observed]
            ),
            "Training batch",
            "Advantage",
        )
        plot(
            "effective_updates",
            "Completed batches and effective updates",
            lambda a: a.step(
                [b["batch"] for b in observed],
                [b.get("updates", b["updates_before"]) for b in observed],
                where="post",
            ),
            "Training batch",
            "Effective updates",
        )
    if process:
        plot(
            "evaluation_mean_reward",
            "Process evaluation",
            lambda a: a.errorbar(
                [r["batch"] for r in process],
                [r["mean_reward"] for r in process],
                yerr=[r.get("reward_std", 0) for r in process],
                marker="o",
                capsize=3,
            ),
            "Completed training batch",
            score_label + " mean +/- population SD",
        )
        plot(
            "evaluation_matched_columns",
            "Matched target columns by evaluation node",
            lambda a: a.boxplot(
                [r["matched_columns"] for r in process],
                tick_labels=[str(r["batch"]) for r in process],
            ),
            "Training batch",
            "Matched required columns (official diagnostic)",
        )

        def official(axes):
            axes.plot(
                [r["batch"] for r in process],
                [r.get("official_srdel", 0) for r in process],
                "o-",
                label="Official SRDEL",
            )
            axes.plot(
                [r["batch"] for r in process],
                [r.get("official_srdt", 0) for r in process],
                "s-",
                label="Official SRDT",
            )
            axes.set_ylim(-0.02, 1.02)
            axes.legend()

        plot(
            "official_success_rates",
            "Official loading and complete-model success",
            official,
            "Training batch",
            "Success fraction",
        )
    if set(final) == {"base", "final"}:
        plot(
            "base_vs_final",
            "Independent final evaluation",
            lambda a: a.bar(
                ["Base", "Final"], [final["base"]["mean_reward"], final["final"]["mean_reward"]]
            ),
            "Policy",
            "Mean " + score_label.lower(),
        )
    durations = [
        r["wall_clock_seconds"] for r in rows if r["wall_clock_seconds"] is not None and r["done"]
    ]
    if durations:
        plot(
            "rollout_duration",
            "Observed completed rollout duration",
            lambda a: a.hist(durations, bins=min(20, max(1, len(durations)))),
            "Seconds",
            "Rollouts",
        )
    write_json(
        directory / "figure-index.json",
        {
            "generated": charts,
            "expected_full_protocol": 9,
            "note": "Figures requiring missing observations are omitted, never fabricated.",
        },
    )
    conclusion = "Final evaluation has not completed; no training-effect claim is made."
    if "final_minus_base_mean_reward" in summary:
        delta = summary["final_minus_base_mean_reward"]
        conclusion = f"Final minus Base mean reward: {delta:+.6f}. " + (
            "The observed mean increased."
            if delta > 0
            else "The observed mean decreased."
            if delta < 0
            else "The observed means tied."
        )
    reward_description = (
        "Pure cell accuracy is correct required cells divided by all required reference cells, "
        "micro-averaged across target tables. Missing rows and cells remain in the denominator. "
        "Numeric correctness is binary under each task's verified tolerance; no proximity or mixture weights are used. "
        if pure_cell
        else "This run uses an explicitly selected historical reward baseline. "
    )
    evaluation_table = (
        "\n| Evaluation | N | Mean cell accuracy | Std | Min | Max |\n|---|---:|---:|---:|---:|---:|\n"
        + "".join(
            f"|{r['model']}|{r['n']}|{r.get('mean_cell_accuracy', 'N/A')}|{r.get('std_cell_accuracy', 'N/A')}|{r.get('min_cell_accuracy', 'N/A')}|{r.get('max_cell_accuracy', 'N/A')}|\n"
            for r in evaluations
        )
    )
    (directory / "REPORT.md").write_text(
        "# ELT training results\n\n"
        f"Status: {state['status']}. Attempted batches: {state['attempted_batches']}; "
        f"completed batches: {state['completed_batches']}; effective updates: {state['updates']}.\n\n"
        f"{conclusion}\n\n"
        f"{reward_description}"
        "Official extraction/loading and transformation success are reported separately.\n\n"
        f"{evaluation_table}\nPer-cell correctness is recorded in evaluation-cells.csv.\n\n"
        "Evaluation trajectories never enter training. Zero-advantage groups do not update parameters. "
        "Incomplete and infrastructure attempts remain in the execution records.\n"
    )
    return summary
