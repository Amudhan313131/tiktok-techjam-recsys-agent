"""Export a judge-auditable evidence bundle from SQLite and the event chain.

The JSON evidence index remains the lossless source of truth.  The additional
summary files are deliberately derived from it so judges can inspect an
iteration without reverse engineering the relational schema.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rex.execution.artifacts import artifact_ref, atomic_write_json
from rex.store.db import Database
from rex.store.event_log import export_events, verify_event_chain


TABLES = (
    "runs",
    "process_sessions",
    "baseline_gates",
    "experiments",
    "transitions",
    "attempts",
    "experiment_repairs",
    "metrics",
    "artifacts",
    "artifact_links",
    "promotions",
    "search_promotions",
    "convergence_transactions",
    "llm_calls",
    "resource_usage",
    "lessons",
    "interventions",
)


def _run_artifact_ids(connection: Any, run_id: str) -> set[str]:
    """Resolve experiment and run-level evidence without relying on artifact ownership.

    Baseline verification and an LLM failure can happen before an experiment
    exists.  Those artifacts consequently have no ``experiment_id`` and must
    be reached through the run-scoped rows that cite them.
    """

    result = {
        str(row["artifact_id"])
        for row in connection.execute(
            "SELECT artifact.artifact_id FROM artifacts artifact "
            "JOIN experiments experiment ON experiment.experiment_id=artifact.experiment_id "
            "WHERE experiment.run_id=? "
            "UNION "
            "SELECT link.artifact_id FROM artifact_links link "
            "JOIN experiments experiment ON experiment.experiment_id=link.experiment_id "
            "WHERE experiment.run_id=? "
            "UNION "
            "SELECT link.artifact_id FROM artifact_links link "
            "JOIN attempts attempt ON attempt.attempt_id=link.attempt_id "
            "JOIN experiments experiment ON experiment.experiment_id=attempt.experiment_id "
            "WHERE experiment.run_id=?",
            (run_id, run_id, run_id),
        ).fetchall()
    }
    baseline = connection.execute(
        "SELECT evidence_json FROM baseline_gates WHERE run_id=?", (run_id,)
    ).fetchone()
    if baseline is not None:
        evidence = json.loads(str(baseline["evidence_json"]))
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            raise RuntimeError(f"baseline evidence for run {run_id} is malformed")
        result.update(evidence)
    for call in connection.execute(
        "SELECT request_artifact_id,response_artifact_id FROM llm_calls WHERE run_id=?",
        (run_id,),
    ).fetchall():
        result.update(
            str(artifact_id)
            for artifact_id in (call["request_artifact_id"], call["response_artifact_id"])
            if artifact_id is not None
        )
    return result


def _rows(database: Database, table: str, run_id: str) -> list[dict[str, Any]]:
    with database.connect() as connection:
        columns = {item[1] for item in connection.execute(f"PRAGMA table_info({table})")}
        if table == "artifacts":
            artifact_ids = sorted(_run_artifact_ids(connection, run_id))
            if not artifact_ids:
                result = []
            else:
                placeholders = ",".join("?" for _ in artifact_ids)
                result = connection.execute(
                    f"SELECT * FROM artifacts WHERE artifact_id IN ({placeholders})",
                    artifact_ids,
                ).fetchall()
        elif table == "artifact_links":
            artifact_ids = sorted(_run_artifact_ids(connection, run_id))
            if not artifact_ids:
                result = []
            else:
                placeholders = ",".join("?" for _ in artifact_ids)
                result = connection.execute(
                    f"SELECT link.* FROM artifact_links link "
                    f"WHERE link.artifact_id IN ({placeholders}) AND ("
                    "(link.experiment_id IS NULL AND link.attempt_id IS NULL) OR "
                    "EXISTS (SELECT 1 FROM experiments experiment "
                    "WHERE experiment.experiment_id=link.experiment_id AND experiment.run_id=?) OR "
                    "EXISTS (SELECT 1 FROM attempts attempt JOIN experiments experiment "
                    "ON experiment.experiment_id=attempt.experiment_id "
                    "WHERE attempt.attempt_id=link.attempt_id AND experiment.run_id=?))",
                    (*artifact_ids, run_id, run_id),
                ).fetchall()
        elif "run_id" in columns:
            result = connection.execute(f"SELECT * FROM {table} WHERE run_id=?", (run_id,)).fetchall()
        elif "experiment_id" in columns:
            result = connection.execute(
                f"SELECT item.* FROM {table} item JOIN experiments e "
                "ON item.experiment_id=e.experiment_id WHERE e.run_id=?",
                (run_id,),
            ).fetchall()
        else:
            result = []
        return [dict(row) for row in result]


def _elapsed_seconds(run: dict[str, Any]) -> float:
    try:
        start = datetime.fromisoformat(str(run["created_at"]))
        stop = datetime.fromisoformat(str(run["updated_at"]))
    except (KeyError, TypeError, ValueError):
        return 0.0
    return max(0.0, (stop - start).total_seconds())


def _iteration_rows(data: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    metrics_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attempts_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    artifacts_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transitions_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lessons_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data["metrics"]:
        metrics_by_experiment[str(row["experiment_id"])].append(row)
    for row in data["attempts"]:
        attempts_by_experiment[str(row["experiment_id"])].append(row)
    artifact_by_id = {str(row["artifact_id"]): row for row in data["artifacts"]}
    for link in data["artifact_links"]:
        experiment_id = link.get("experiment_id")
        artifact = artifact_by_id.get(str(link["artifact_id"]))
        if experiment_id and artifact:
            artifacts_by_experiment[str(experiment_id)].append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "kind": artifact["kind"],
                    "sha256": artifact["sha256"],
                    "path": link.get("artifact_path") or artifact.get("path"),
                }
            )
    for row in data["transitions"]:
        transitions_by_experiment[str(row["experiment_id"])].append(row)
    for row in data["lessons"]:
        lessons_by_experiment[str(row["experiment_id"])].append(row)

    result: list[dict[str, Any]] = []
    for experiment in sorted(data["experiments"], key=lambda item: item["iteration_number"]):
        experiment_id = str(experiment["experiment_id"])
        attempts = attempts_by_experiment[experiment_id]
        failures = [item for item in attempts if item["status"] != "success"]
        result.append(
            {
                "iteration": experiment["iteration_number"],
                "experiment_id": experiment_id,
                "parent_id": experiment.get("parent_id"),
                "operator": experiment["operator"],
                "hypothesis": experiment["hypothesis"],
                "state": experiment["state"],
                "terminal_reason": experiment.get("terminal_reason"),
                "commit_sha": experiment.get("commit_sha"),
                "config_sha256": experiment.get("config_sha256"),
                "metrics": metrics_by_experiment[experiment_id],
                "attempts": attempts,
                "failures": failures,
                "transitions": transitions_by_experiment[experiment_id],
                "artifacts": artifacts_by_experiment[experiment_id],
                "lessons": lessons_by_experiment[experiment_id],
            }
        )
    return result


def _markdown_report(run_id: str, iterations: list[dict[str, Any]]) -> str:
    lines = [f"# Experiment report: {run_id}", ""]
    lines.extend(
        [
            "| # | Experiment | Operator | State | Best recorded primary | Failures | Hypothesis |",
            "|---:|---|---|---|---:|---:|---|",
        ]
    )
    for item in iterations:
        primaries = [
            float(metric["primary_score"])
            for metric in item["metrics"]
            if metric.get("primary_score") is not None
        ]
        primary = f"{max(primaries):.6f}" if primaries else "—"
        hypothesis = str(item["hypothesis"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item['iteration']} | {item['experiment_id']} | {item['operator']} | "
            f"{item['state']} | {primary} | {len(item['failures'])} | {hypothesis} |"
        )
    for item in iterations:
        lines.extend(
            [
                "",
                f"## {item['iteration']}. {item['experiment_id']}",
                "",
                f"- State: {item['state']}",
                f"- Operator: {item['operator']}",
                f"- Parent: {item['parent_id'] or 'none'}",
                f"- Commit: {item['commit_sha'] or 'not recorded'}",
                f"- Hypothesis: {item['hypothesis']}",
            ]
        )
        if item["terminal_reason"]:
            lines.append(f"- Terminal reason: {item['terminal_reason']}")
        if item["metrics"]:
            lines.extend(
                [
                    "",
                    "### Metrics",
                    "",
                    "| Split | Fold | Seed | GAUC | nDCG@5 | Primary |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for metric in item["metrics"]:
                lines.append(
                    f"| {metric['split']} | {metric.get('fold') or '—'} | "
                    f"{metric.get('seed') if metric.get('seed') is not None else '—'} | "
                    f"{float(metric['gauc']):.6f} | {float(metric['ndcg5']):.6f} | "
                    f"{float(metric['primary_score']):.6f} |"
                )
        if item["failures"]:
            lines.extend(["", "### Recovery evidence", ""])
            for attempt in item["failures"]:
                summary = str(attempt.get("error_summary") or "").replace("\n", " ")
                lines.append(
                    f"- {attempt['attempt_id']} ({attempt['rung']}, repair "
                    f"{attempt['repair_number']}): {attempt['status']} — {summary[:300]}"
                )
        if item["lessons"]:
            lines.extend(["", "### Evidence-bound lessons", ""])
            for lesson in item["lessons"]:
                lines.append(f"- {lesson['lesson']}")
        if item["artifacts"]:
            lines.extend(["", "### Artifacts", ""])
            for artifact in item["artifacts"]:
                lines.append(
                    f"- `{artifact['kind']}` `{artifact['artifact_id']}` "
                    f"SHA-256 `{artifact['sha256']}`"
                )
    return "\n".join(lines) + "\n"


def build_report(database: Database, run_id: str, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = {table: _rows(database, table, run_id) for table in TABLES}
    event_path = output / "events.jsonl"
    export_events(database, run_id, event_path, include_exported=True)
    if not verify_event_chain(event_path):
        raise RuntimeError("exported event hash chain is invalid")

    iterations = _iteration_rows(data)
    graph = {
        "run_id": run_id,
        "nodes": [
            {
                "experiment_id": item["experiment_id"],
                "parent_id": item["parent_id"],
                "iteration": item["iteration_number"],
                "operator": item["operator"],
                "state": item["state"],
            }
            for item in data["experiments"]
        ],
    }
    atomic_write_json(output / "experiment_graph.json", graph)
    run = data["runs"][0] if data["runs"] else {}
    component_wall_seconds = sum(
        item["wall_seconds"] for item in data["resource_usage"]
    )
    resources = {
        "agent_wall_seconds": _elapsed_seconds(run),
        "summed_component_wall_seconds": component_wall_seconds,
        # Backward-compatible name used by the fixture report contract.
        "wall_seconds": component_wall_seconds,
        "cpu_user_seconds": sum(item["cpu_user_seconds"] for item in data["resource_usage"]),
        "cpu_system_seconds": sum(item["cpu_system_seconds"] for item in data["resource_usage"]),
        "gpu_seconds": sum(item["gpu_seconds"] for item in data["resource_usage"]),
        "gpu_hours": sum(item["gpu_seconds"] for item in data["resource_usage"]) / 3600.0,
        "llm_tokens": sum(item["llm_tokens"] for item in data["resource_usage"]),
        "iterations": len(data["experiments"]),
        "manual_interventions": len(data["interventions"]),
    }
    atomic_write_json(output / "resources.json", resources)
    atomic_write_json(output / "interventions.json", data["interventions"])
    evidence = data
    evidence_path = output / "evidence_index.json"
    atomic_write_json(evidence_path, evidence)
    iteration_path = output / "iteration_logs.json"
    atomic_write_json(iteration_path, iterations)
    (output / "experiments.md").write_text(
        _markdown_report(run_id, iterations), encoding="utf-8"
    )
    return {
        "event_log": artifact_ref(event_path, "event_log").model_dump(mode="json"),
        "evidence_index": artifact_ref(evidence_path, "evidence_index").model_dump(mode="json"),
        "resources": resources,
        "experiments": len(data["experiments"]),
        "iteration_logs": artifact_ref(
            iteration_path, "iteration_logs"
        ).model_dump(mode="json"),
    }
