"""Export a judge-auditable evidence bundle from SQLite and the event chain.

The JSON evidence index remains the lossless source of truth.  The additional
summary files are deliberately derived from it so judges can inspect an
iteration without reverse engineering the relational schema.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rex.data.manifest import sha256_file
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

REPORT_SCHEMA_VERSION = "1.0"
ITERATION_CAP = 50
OFFICIAL_KUAIRAND_PURE_VALID = {
    "GAUC": 0.6674,
    "nDCG@5": 0.5357,
    "primary": 0.6016,
}
_DIFF_KINDS = frozenset({"patch", "repair_patch"})
_CONFIG_KINDS = frozenset({"experiment_config", "repaired_experiment_config"})
_AUTOMATED_INTERVENTION_ACTORS = frozenset(
    {"agent", "automation", "fixture_rehearsal", "rex", "system"}
)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"unparsed": str(value)}
    return decoded if isinstance(decoded, dict) else {"value": decoded}


def _is_manual_intervention(row: dict[str, Any]) -> bool:
    """Classify conservatively: unknown external actors count as manual.

    Controlled fault injection is recorded in the same table by fixture and
    rehearsal automation.  Counting those events as human help would
    understate autonomy, so the report distinguishes them explicitly.
    """

    actor = str(row.get("actor") or "").strip().lower()
    if actor in _AUTOMATED_INTERVENTION_ACTORS:
        return False
    return not actor.startswith(("automation_", "fixture_", "rehearsal_", "rex_"))


def _intervention_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    manual = [dict(row) for row in rows if _is_manual_intervention(row)]
    automated = [dict(row) for row in rows if not _is_manual_intervention(row)]
    count = len(manual)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manual_intervention_count": count,
        "summary": (
            "No manual interventions were recorded during the agent run."
            if count == 0
            else f"{count} manual intervention(s) were recorded during the agent run."
        ),
        "manual_interventions": manual,
        "automated_control_events_excluded": automated,
        "recorded_interventions_total": len(rows),
        "classification_policy": (
            "Known REX, fixture, rehearsal, system, agent, and automation actors are "
            "automated; unknown external actors are conservatively counted as manual."
        ),
    }


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
            result = connection.execute(
                f"SELECT * FROM {table} WHERE run_id=?", (run_id,)
            ).fetchall()
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
    repairs_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    llm_calls_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
    for row in data["experiment_repairs"]:
        repairs_by_experiment[str(row["experiment_id"])].append(row)
    for row in data["llm_calls"]:
        if row.get("experiment_id"):
            llm_calls_by_experiment[str(row["experiment_id"])].append(row)

    result: list[dict[str, Any]] = []
    for experiment in sorted(data["experiments"], key=lambda item: item["iteration_number"]):
        experiment_id = str(experiment["experiment_id"])
        attempts = attempts_by_experiment[experiment_id]
        failures = [item for item in attempts if item["status"] != "success"]
        proposal = _json_object(experiment.get("proposal_json"))
        artifacts = artifacts_by_experiment[experiment_id]
        diff_artifacts = [item for item in artifacts if item["kind"] in _DIFF_KINDS]
        config_artifacts = [item for item in artifacts if item["kind"] in _CONFIG_KINDS]
        applied_change = {
            "mode": (
                "source_patch"
                if diff_artifacts
                else "versioned_configuration"
                if config_artifacts
                else "no_applied_change_artifact"
            ),
            "diff_artifacts": diff_artifacts,
            "configuration_artifacts": config_artifacts,
            "source_diff_required": bool(diff_artifacts),
            "explanation": (
                "Exact accepted source diff(s) are embedded below."
                if diff_artifacts
                else "No source patch was applied; this iteration used a versioned configuration transaction."
                if config_artifacts
                else "No accepted source or configuration change artifact was recorded."
            ),
        }
        recovery_events: list[dict[str, Any]] = [
            {
                "kind": "attempt_failure",
                "attempt_id": item["attempt_id"],
                "rung": item["rung"],
                "repair_number": item["repair_number"],
                "status": item["status"],
                "error_type": item.get("error_type"),
                "error_summary": item.get("error_summary"),
            }
            for item in failures
        ]
        recovery_events.extend(
            {
                "kind": "bounded_repair",
                "repair_id": item["repair_id"],
                "repair_number": item["repair_number"],
                "phase": item["phase"],
                "failure_status": item["failure_status"],
                "plan": _json_object(item.get("plan_json")),
                "completed": item.get("completed_at") is not None,
                "created_at": item.get("created_at"),
                "completed_at": item.get("completed_at"),
            }
            for item in repairs_by_experiment[experiment_id]
        )
        recovery_events.extend(
            {
                "kind": "llm_error",
                "call_id": item["call_id"],
                "role": item["role"],
                "provider": item["provider"],
                "error": item["error"],
            }
            for item in llm_calls_by_experiment[experiment_id]
            if item.get("error")
        )
        result.append(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "iteration": experiment["iteration_number"],
                "experiment_id": experiment_id,
                "parent_id": experiment.get("parent_id"),
                "operator": experiment["operator"],
                "hypothesis": experiment["hypothesis"],
                "scientific_intent": {
                    "hypothesis": experiment["hypothesis"],
                    "mechanism": proposal.get("mechanism"),
                    "primary_change": proposal.get("primary_change"),
                    "expected_metric_effects": proposal.get("expected_metric_effects", {}),
                    "falsifier": proposal.get("falsifier"),
                    "why": proposal.get("mechanism") or experiment["hypothesis"],
                },
                "state": experiment["state"],
                "terminal_reason": experiment.get("terminal_reason"),
                "commit_sha": experiment.get("commit_sha"),
                "config_sha256": experiment.get("config_sha256"),
                "metrics": metrics_by_experiment[experiment_id],
                "resulting_metrics": metrics_by_experiment[experiment_id],
                "attempts": attempts,
                "failures": failures,
                "error_recovery_events": recovery_events,
                "transitions": transitions_by_experiment[experiment_id],
                "applied_change": applied_change,
                "artifacts": artifacts,
                "lessons": lessons_by_experiment[experiment_id],
            }
        )
    return result


def _embed_applied_diffs(iterations: list[dict[str, Any]]) -> None:
    """Embed exact accepted diffs so the final copied report is self-contained."""

    for iteration in iterations:
        embedded: list[dict[str, Any]] = []
        for raw in iteration["applied_change"]["diff_artifacts"]:
            path = Path(str(raw["path"]))
            evidence = {
                "artifact_id": raw["artifact_id"],
                "kind": raw["kind"],
                "sha256": raw["sha256"],
                "path": str(path),
            }
            if path.is_symlink() or not path.is_file():
                evidence.update(
                    {
                        "available": False,
                        "error": "recorded diff artifact is missing or is a symlink",
                    }
                )
            elif sha256_file(path) != raw["sha256"]:
                evidence.update(
                    {"available": False, "error": "recorded diff artifact hash drifted"}
                )
            else:
                try:
                    diff_text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as error:
                    evidence.update({"available": False, "error": str(error)})
                else:
                    evidence.update({"available": True, "diff": diff_text})
            embedded.append(evidence)
        iteration["applied_change"]["embedded_diffs"] = embedded


def _runtime_environment_identity(
    data: dict[str, list[dict[str, Any]]], output: Path
) -> dict[str, Any] | None:
    candidates = [
        item for item in data["artifacts"] if item["kind"] == "runtime_environment_identity"
    ]
    destination = output / "environment_identity.json"
    if not candidates:
        destination.unlink(missing_ok=True)
        return None
    digests = {str(item["sha256"]) for item in candidates}
    if len(digests) != 1:
        raise RuntimeError("run has conflicting runtime environment identities")
    source = Path(str(candidates[0]["path"]))
    if source.is_symlink() or not source.is_file() or sha256_file(source) not in digests:
        raise RuntimeError("runtime environment identity artifact is missing or drifted")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"runtime environment identity is invalid: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("runtime environment identity must be a JSON object")
    if payload.get("runtime_kind") == "docker" and not str(
        payload.get("worker_image_digest") or ""
    ).startswith("sha256:"):
        raise RuntimeError("Docker runtime identity has no immutable worker image digest")
    atomic_write_json(destination, payload)
    return {
        "artifact_id": candidates[0]["artifact_id"],
        "source_sha256": candidates[0]["sha256"],
        "report_sha256": sha256_file(destination),
        "identity": payload,
    }


def _validation_results(
    data: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    run = data["runs"][0] if data["runs"] else {}
    champion = (
        run.get("search_champion_experiment_id") or run.get("best_ever_experiment_id") or "baseline"
    )
    baseline_row = data["baseline_gates"][0] if data["baseline_gates"] else None
    selected: dict[str, Any] | None = None
    source = "experiment"
    if champion == "baseline" and baseline_row is not None:
        source = "run_verified_baseline"
        selected = {
            "GAUC": float(baseline_row["gauc"]),
            "nDCG@5": float(baseline_row["ndcg5"]),
            "primary": float(baseline_row["primary_units"]) / 1_000_000.0,
            "split": "valid",
        }
    else:
        eligible = [
            item
            for item in data["metrics"]
            if item["experiment_id"] == champion and item["split"] == "valid"
        ]
        if eligible:
            metric = max(
                eligible,
                key=lambda item: (int(item["primary_units"]), int(item["metric_id"])),
            )
            selected = {
                "GAUC": float(metric["gauc"]),
                "nDCG@5": float(metric["ndcg5"]),
                "primary": float(metric["primary_score"]),
                "split": "valid",
                "fold": metric.get("fold"),
                "seed": metric.get("seed"),
                "evaluator_sha256": metric.get("evaluator_sha256"),
            }
    official = dict(OFFICIAL_KUAIRAND_PURE_VALID)
    delta = (
        None
        if selected is None
        else {
            name: float(selected[name]) - float(official[name])
            for name in ("GAUC", "nDCG@5", "primary")
        }
    )
    baseline = (
        None
        if baseline_row is None
        else {
            "GAUC": float(baseline_row["gauc"]),
            "nDCG@5": float(baseline_row["ndcg5"]),
            "primary": float(baseline_row["primary_units"]) / 1_000_000.0,
            "split": "valid",
        }
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset": "KuaiRand-Pure",
        "validation_best": selected,
        "validation_best_source": source if selected is not None else None,
        "validation_best_experiment_id": champion if selected is not None else None,
        "official_baseline": {
            **official,
            "split": "valid",
            "source": "Starter Kit fm_official.valid",
        },
        "delta_over_official_baseline": delta,
        "run_verified_baseline": baseline,
        "bonus_benchmarks": [],
        "hidden_test_scored_locally": False,
        "notes": [
            "All reported run metrics are validation metrics.",
            "The published hidden-test baseline is intentionally not compared with validation results.",
            "KuaiRand-1k and KuaiRand-27k were not attempted unless listed in bonus_benchmarks.",
        ],
    }


def _artifact_summary(
    data: dict[str, list[dict[str, Any]]], results: dict[str, Any]
) -> dict[str, Any]:
    champion = results.get("validation_best_experiment_id")
    baseline_artifact_ids: set[str] = set()
    if champion == "baseline" and data["baseline_gates"]:
        try:
            raw_ids = json.loads(str(data["baseline_gates"][0]["evidence_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_ids = []
        if isinstance(raw_ids, list):
            baseline_artifact_ids = {str(item) for item in raw_ids}
    champion_artifacts = [
        {
            "artifact_id": item["artifact_id"],
            "kind": item["kind"],
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in data["artifacts"]
        if (champion not in (None, "baseline") and item.get("experiment_id") == champion)
        or item["artifact_id"] in baseline_artifact_ids
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "validation_best_experiment_id": champion,
        "validation_model_artifacts": champion_artifacts,
        "checkpoint_artifacts": [
            item for item in champion_artifacts if item["kind"] in {"checkpoint", "model_bundle"}
        ],
        "validation_prediction_artifacts": [
            item
            for item in champion_artifacts
            if item["kind"] in {"predictions", "valid_predictions"}
        ],
        "final_submission": {
            "status": "created_only_after_the_validation_run_is_complete",
            "authoritative_evidence": (
                "The separately sealed final-submission manifest records submission.csv, "
                "test predictions, the selected checkpoint, both organizer checks, and hashes."
            ),
            "test_scored_locally": False,
        },
    }


def _resource_summary(
    data: dict[str, list[dict[str, Any]]], run: dict[str, Any], manual_count: int
) -> dict[str, Any]:
    input_tokens = sum(int(item["input_tokens"]) for item in data["llm_calls"])
    output_tokens = sum(int(item["output_tokens"]) for item in data["llm_calls"])
    component_wall_seconds = sum(float(item["wall_seconds"]) for item in data["resource_usage"])
    gpu_seconds = sum(float(item["gpu_seconds"]) for item in data["resource_usage"])
    active_session_seconds = sum(
        float(item["monotonic_seconds"]) for item in data["process_sessions"]
    )
    iterations = len(data["experiments"])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "agent_wall_seconds": _elapsed_seconds(run),
        "active_process_session_seconds": active_session_seconds,
        "summed_component_wall_seconds": component_wall_seconds,
        # Backward-compatible name used by the fixture report contract.
        "wall_seconds": component_wall_seconds,
        "cpu_user_seconds": sum(float(item["cpu_user_seconds"]) for item in data["resource_usage"]),
        "cpu_system_seconds": sum(
            float(item["cpu_system_seconds"]) for item in data["resource_usage"]
        ),
        "gpu_seconds": gpu_seconds,
        "gpu_hours": gpu_seconds / 3600.0,
        "llm_input_tokens": input_tokens,
        "llm_output_tokens": output_tokens,
        "llm_total_tokens": input_tokens + output_tokens,
        # Backward-compatible total.
        "llm_tokens": input_tokens + output_tokens,
        "llm_calls": len(data["llm_calls"]),
        "llm_failed_calls": sum(bool(item.get("error")) for item in data["llm_calls"]),
        "resource_usage_llm_tokens": sum(
            int(item["llm_tokens"]) for item in data["resource_usage"]
        ),
        "iterations": iterations,
        "iterations_used": iterations,
        "iteration_cap": ITERATION_CAP,
        "iterations_remaining": max(0, ITERATION_CAP - iterations),
        "manual_interventions": manual_count,
        "run_state": run.get("state"),
        "stop_reason": run.get("stop_reason"),
        "run_started_at": run.get("created_at"),
        "run_last_updated_at": run.get("updated_at"),
    }


def _run_recovery_summary(
    data: dict[str, list[dict[str, Any]]], iterations: list[dict[str, Any]]
) -> dict[str, Any]:
    interrupted_sessions = [
        {
            "session_id": item["session_id"],
            "exit_reason": item.get("exit_reason"),
            "started_at": item.get("started_at"),
            "ended_at": item.get("ended_at"),
            "active_seconds": float(item.get("monotonic_seconds") or 0.0),
        }
        for item in data["process_sessions"]
        if item.get("exit_reason") not in {None, "fixture_complete", "production_complete"}
    ]
    pre_experiment_llm_errors = [
        {
            "call_id": item["call_id"],
            "role": item["role"],
            "provider": item["provider"],
            "error": item["error"],
        }
        for item in data["llm_calls"]
        if not item.get("experiment_id") and item.get("error")
    ]
    iteration_events = [
        {
            "iteration": item["iteration"],
            "experiment_id": item["experiment_id"],
            "events": item["error_recovery_events"],
        }
        for item in iterations
        if item["error_recovery_events"]
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "interrupted_process_sessions": interrupted_sessions,
        "pre_experiment_llm_errors": pre_experiment_llm_errors,
        "iteration_error_recovery_events": iteration_events,
        "interrupted_session_count": len(interrupted_sessions),
        "iterations_with_error_or_recovery": len(iteration_events),
    }


def _markdown_report(
    run_id: str,
    iterations: list[dict[str, Any]],
    *,
    results: dict[str, Any],
    resources: dict[str, Any],
    interventions: dict[str, Any],
    recovery: dict[str, Any],
) -> str:
    lines = [f"# Experiment report: {run_id}", ""]
    best = results.get("validation_best")
    delta = results.get("delta_over_official_baseline")
    lines.extend(
        [
            "## Run summary",
            "",
            f"- Run state: {resources.get('run_state') or 'unknown'}",
            f"- Stop reason: {resources.get('stop_reason') or 'not recorded'}",
            f"- Iterations: {resources['iterations_used']} / {resources['iteration_cap']}",
            f"- Agent wall-clock: {float(resources['agent_wall_seconds']):.3f} seconds",
            f"- LLM tokens: {resources['llm_total_tokens']} total "
            f"({resources['llm_input_tokens']} input + {resources['llm_output_tokens']} output)",
            f"- GPU usage: {float(resources['gpu_hours']):.6f} GPU-hours",
            f"- Manual interventions: {interventions['manual_intervention_count']}",
            "- Local hidden-test scoring: never performed",
            "",
            "## Validation-best results",
            "",
            "| Benchmark | Result | GAUC | nDCG@5 | Primary |",
            "|---|---|---:|---:|---:|",
        ]
    )
    if best is not None:
        lines.append(
            f"| KuaiRand-Pure | Validation best ({results['validation_best_experiment_id']}) | "
            f"{float(best['GAUC']):.6f} | {float(best['nDCG@5']):.6f} | "
            f"{float(best['primary']):.6f} |"
        )
    official = results["official_baseline"]
    lines.append(
        f"| KuaiRand-Pure | Official FM validation baseline | "
        f"{float(official['GAUC']):.6f} | {float(official['nDCG@5']):.6f} | "
        f"{float(official['primary']):.6f} |"
    )
    if delta is not None:
        lines.append(
            f"| KuaiRand-Pure | Absolute delta over official baseline | "
            f"{float(delta['GAUC']):+.6f} | {float(delta['nDCG@5']):+.6f} | "
            f"{float(delta['primary']):+.6f} |"
        )
    if recovery["interrupted_process_sessions"] or recovery["pre_experiment_llm_errors"]:
        lines.extend(["", "## Run-level error and recovery events", ""])
        for session in recovery["interrupted_process_sessions"]:
            lines.append(
                f"- Process session `{session['session_id']}` ended as "
                f"`{session['exit_reason']}` after {session['active_seconds']:.3f} seconds."
            )
        for event in recovery["pre_experiment_llm_errors"]:
            lines.append(f"- Pre-experiment LLM call `{event['call_id']}` failed: {event['error']}")
    lines.extend(["", "## Per-iteration log", ""])
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
                f"- Why: {item['scientific_intent']['why'] or 'not recorded'}",
                f"- Primary change: {item['scientific_intent']['primary_change'] or 'not recorded'}",
                f"- Falsifier: {item['scientific_intent']['falsifier'] or 'not recorded'}",
            ]
        )
        if item["terminal_reason"]:
            lines.append(f"- Terminal reason: {item['terminal_reason']}")
        change = item["applied_change"]
        lines.extend(["", "### Applied change", "", change["explanation"]])
        for embedded in change["embedded_diffs"]:
            lines.extend(
                [
                    "",
                    f"Diff artifact `{embedded['artifact_id']}` (SHA-256 `{embedded['sha256']}`):",
                    "",
                ]
            )
            if embedded["available"]:
                lines.extend(["```diff", embedded["diff"].rstrip(), "```"])
            else:
                lines.append(f"Unavailable: {embedded['error']}")
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
        if item["error_recovery_events"]:
            lines.extend(["", "### Recovery evidence", ""])
            for event in item["error_recovery_events"]:
                if event["kind"] == "attempt_failure":
                    summary = str(event.get("error_summary") or "").replace("\n", " ")
                    lines.append(
                        f"- Attempt `{event['attempt_id']}` ({event['rung']}, repair "
                        f"{event['repair_number']}): {event['status']} — {summary[:300]}"
                    )
                elif event["kind"] == "bounded_repair":
                    lines.append(
                        f"- Bounded repair `{event['repair_id']}` for "
                        f"{event['failure_status']}: "
                        f"{'completed' if event['completed'] else 'incomplete'}"
                    )
                else:
                    lines.append(f"- LLM call `{event['call_id']}` failed: {event['error']}")
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
    """Atomically regenerate the complete judge-facing report from durable state.

    Replaying this function against unchanged database contents produces the
    same report bytes.  Call it once more after the run is COMPLETE and its
    final process session has closed so wall-clock and recovery evidence are
    final before submission discovery fingerprints the directory.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = {table: _rows(database, table, run_id) for table in TABLES}
    if not data["runs"]:
        raise RuntimeError(f"cannot report unknown run: {run_id}")
    forbidden_metrics = [item for item in data["metrics"] if item.get("split") == "test"]
    if forbidden_metrics:
        raise RuntimeError("production report refuses locally scored hidden-test metrics")
    event_path = output / "events.jsonl"
    export_events(database, run_id, event_path, include_exported=True)
    if not verify_event_chain(event_path):
        raise RuntimeError("exported event hash chain is invalid")

    iterations = _iteration_rows(data)
    _embed_applied_diffs(iterations)
    graph = {
        "schema_version": REPORT_SCHEMA_VERSION,
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
    run = data["runs"][0]
    interventions = _intervention_summary(data["interventions"])
    resources = _resource_summary(data, run, int(interventions["manual_intervention_count"]))
    results = _validation_results(data)
    artifacts = _artifact_summary(data, results)
    environment_identity = _runtime_environment_identity(data, output)
    artifacts["runtime_environment_identity"] = environment_identity
    recovery = _run_recovery_summary(data, iterations)
    atomic_write_json(output / "resources.json", resources)
    atomic_write_json(output / "interventions.json", data["interventions"])
    atomic_write_json(output / "manual_interventions.json", interventions)
    _atomic_write_text(
        output / "manual_interventions.md",
        "# Manual intervention summary\n\n"
        + str(interventions["summary"])
        + "\n\n"
        + f"Recorded control/intervention events: {interventions['recorded_interventions_total']}.\n"
        + f"Automated control events excluded: {len(interventions['automated_control_events_excluded'])}.\n",
    )
    atomic_write_json(output / "results.json", results)
    atomic_write_json(output / "artifact_summary.json", artifacts)
    atomic_write_json(output / "recovery_events.json", recovery)
    evidence = data
    evidence_path = output / "evidence_index.json"
    atomic_write_json(evidence_path, evidence)
    iteration_path = output / "iteration_logs.json"
    atomic_write_json(iteration_path, iterations)
    _atomic_write_text(
        output / "experiments.md",
        _markdown_report(
            run_id,
            iterations,
            results=results,
            resources=resources,
            interventions=interventions,
            recovery=recovery,
        ),
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "event_log": artifact_ref(event_path, "event_log").model_dump(mode="json"),
        "evidence_index": artifact_ref(evidence_path, "evidence_index").model_dump(mode="json"),
        "resources": resources,
        "results": results,
        "manual_interventions": interventions,
        "artifact_summary": artifacts,
        "runtime_environment_identity": environment_identity,
        "recovery_events": recovery,
        "experiments": len(data["experiments"]),
        "iteration_logs": artifact_ref(iteration_path, "iteration_logs").model_dump(mode="json"),
    }
