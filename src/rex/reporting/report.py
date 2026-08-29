"""Export a judge-auditable evidence bundle from SQLite and the event chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rex.execution.artifacts import artifact_ref, atomic_write_json
from rex.store.db import Database
from rex.store.event_log import export_events, verify_event_chain


TABLES = (
    "runs",
    "process_sessions",
    "experiments",
    "transitions",
    "attempts",
    "metrics",
    "artifacts",
    "artifact_links",
    "promotions",
    "llm_calls",
    "resource_usage",
    "lessons",
    "interventions",
)


def _rows(database: Database, table: str, run_id: str) -> list[dict[str, Any]]:
    with database.connect() as connection:
        columns = {item[1] for item in connection.execute(f"PRAGMA table_info({table})")}
        if "run_id" in columns:
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


def build_report(database: Database, run_id: str, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = {table: _rows(database, table, run_id) for table in TABLES}
    event_path = output / "events.jsonl"
    export_events(database, run_id, event_path, include_exported=True)
    if not verify_event_chain(event_path):
        raise RuntimeError("exported event hash chain is invalid")

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
    resources = {
        "wall_seconds": sum(item["wall_seconds"] for item in data["resource_usage"]),
        "cpu_user_seconds": sum(item["cpu_user_seconds"] for item in data["resource_usage"]),
        "cpu_system_seconds": sum(item["cpu_system_seconds"] for item in data["resource_usage"]),
        "gpu_seconds": sum(item["gpu_seconds"] for item in data["resource_usage"]),
        "llm_tokens": sum(item["llm_tokens"] for item in data["resource_usage"]),
    }
    atomic_write_json(output / "resources.json", resources)
    atomic_write_json(output / "interventions.json", data["interventions"])
    evidence = data
    evidence_path = output / "evidence_index.json"
    atomic_write_json(evidence_path, evidence)
    lines = [f"# Experiment report: {run_id}", ""]
    for experiment in data["experiments"]:
        lines.extend(
            [
                f"## {experiment['iteration_number']}. {experiment['experiment_id']}",
                "",
                f"- State: {experiment['state']}",
                f"- Operator: {experiment['operator']}",
                f"- Hypothesis: {experiment['hypothesis']}",
                "",
            ]
        )
    (output / "experiments.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "event_log": artifact_ref(event_path, "event_log").model_dump(mode="json"),
        "evidence_index": artifact_ref(evidence_path, "evidence_index").model_dump(mode="json"),
        "resources": resources,
        "experiments": len(data["experiments"]),
    }
