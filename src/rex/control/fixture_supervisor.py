"""Fixture-only autonomous loop used to prove orchestration and recovery.

This module deliberately cannot open the competition data or create a production
submission.  It exercises the same provider, patch, worktree, worker, storage,
and state-machine boundaries with tiny generated arrays and synthetic metrics.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rex.agents.coordinator import PatchTransactionCoordinator, PreparedExperiment
from rex.agents.memory import remember_reflection
from rex.agents.patch_guard import PatchPolicy
from rex.agents.provider import ProviderResponse, StructuredProvider
from rex.agents.recovery import decide_repair
from rex.agents.services import CodingService, DiagnosisService, ProposalService
from rex.agents.workspace import GitWorkspace
from rex.contracts import (
    AttemptStatus,
    ExperimentProposal,
    ExperimentState,
    Metrics,
    RunRequest,
    RunResult,
    RunState,
)
from rex.control.budget import BudgetConfig, deadline_epoch_ms
from rex.data.manifest import repo_root, sha256_file
from rex.execution.artifacts import artifact_ref, atomic_write_json, load_prediction_artifact
from rex.execution.runner import execute_request
from rex.reporting.report import build_report
from rex.store.db import Database
from rex.store.repository import ExperimentRepository, RepositoryError


HASH = "0" * 64


@dataclass(frozen=True)
class FixtureRunConfig:
    source_path: Path
    runs_dir: Path
    budget_config: Path
    protected_paths: Path
    max_hypotheses: int
    attempt_timeout_seconds: int
    wall_clock_seconds: int
    process_stale_after_seconds: int
    full_threshold: float
    inject_worker_nan_once: bool
    inject_worker_nan_always_iteration: int | None
    llm: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "FixtureRunConfig":
        candidate = Path(path).resolve()
        value = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("fixture run configuration must be a YAML mapping")
        if value.get("execution_mode") != "fixture":
            raise RuntimeError(
                "production science is disabled in this phase; use configs/run/fixture.yaml"
            )
        root = repo_root()

        def resolve(raw: str) -> Path:
            item = Path(raw)
            return item if item.is_absolute() else root / item

        inject_worker_nan_once = value.get("inject_worker_nan_once", False)
        if not isinstance(inject_worker_nan_once, bool):
            raise ValueError("inject_worker_nan_once must be a boolean")
        persistent_nan_iteration = value.get("inject_worker_nan_always_iteration")
        if persistent_nan_iteration is not None and (
            isinstance(persistent_nan_iteration, bool)
            or not isinstance(persistent_nan_iteration, int)
            or not 1 <= persistent_nan_iteration <= 50
        ):
            raise ValueError(
                "inject_worker_nan_always_iteration must be null or an integer from 1 to 50"
            )
        config = cls(
            source_path=candidate,
            runs_dir=resolve(str(value.get("runs_dir", "runs"))),
            budget_config=resolve(str(value["budget_config"])),
            protected_paths=resolve(str(value["protected_paths"])),
            max_hypotheses=int(value.get("max_fixture_hypotheses", 3)),
            attempt_timeout_seconds=int(value.get("attempt_timeout_seconds", 30)),
            wall_clock_seconds=int(value.get("fixture_wall_clock_seconds", 900)),
            process_stale_after_seconds=int(value.get("process_stale_after_seconds", 90)),
            full_threshold=float(value.get("fixture_full_threshold", 0.001)),
            inject_worker_nan_once=inject_worker_nan_once,
            inject_worker_nan_always_iteration=persistent_nan_iteration,
            llm=dict(value.get("llm", {})),
        )
        if not 1 <= config.max_hypotheses <= 50:
            raise ValueError("max_fixture_hypotheses must be between 1 and 50")
        if config.attempt_timeout_seconds <= 0 or config.wall_clock_seconds <= 0:
            raise ValueError("fixture timeout values must be positive")
        if config.process_stale_after_seconds <= config.attempt_timeout_seconds:
            raise ValueError(
                "process_stale_after_seconds must exceed attempt_timeout_seconds"
            )
        if config.full_threshold < 0:
            raise ValueError("fixture_full_threshold must be non-negative")
        return config


class FixtureScriptProvider:
    """Deterministic role-aware provider for LLM-free fixture rehearsals."""

    def __init__(self) -> None:
        self.proposal_index = 0

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        context = json.loads(prompt)
        if role == "proposal":
            requested = context.get("fixture_iteration_number")
            number = int(requested) if requested is not None else self.proposal_index + 1
            self.proposal_index = max(self.proposal_index, number)
            experiment_id = f"fixture-{number:03d}"
            value = {
                "experiment_id": experiment_id,
                "parent_id": None,
                "operator": "HYPERPARAMETER",
                "hypothesis": "A fixture-only bias change proves isolated autonomous execution.",
                "mechanism": "The synthetic model adds a declared constant to every fixture score.",
                "primary_change": f"fixture default bias variant {number}",
                "files_to_change": ["src/rex/models/experimental/fixture.py"],
                "expected_metric_effects": {"fixture_primary": "controlled"},
                "falsifier": "The patched worktree does not change the synthetic prediction value.",
                "leakage_analysis": "Generated fixture arrays contain no competition or hidden labels.",
                "estimated_seconds": 10,
                "cheap_rung": {"fixture": "small"},
                "full_rung": {"fixture": "full"},
            }
        elif role == "patch":
            proposal = context["proposal"]
            number = int(str(proposal["experiment_id"]).rsplit("-", 1)[-1])
            biases = {1: 0.010, 2: -0.010, 3: 0.020}
            bias = biases.get(number, 0.001 * number)
            value = {
                "patch": (
                    "--- a/src/rex/models/experimental/fixture.py\n"
                    "+++ b/src/rex/models/experimental/fixture.py\n"
                    "@@ -15,7 +15,7 @@ from rex.data.views import FeatureView, TargetView\n"
                    " \n"
                    " # Fixture-only patches deliberately change this value to prove that code from an\n"
                    " # isolated worktree, rather than the main checkout, is what the worker imports.\n"
                    "-DEFAULT_BIAS = 0.0\n"
                    f"+DEFAULT_BIAS = {bias:.3f}\n"
                    " \n"
                    " \n"
                    " class FixturePlugin:\n"
                ),
                "rationale": "Changes one harmless synthetic constant inside the fixture allowlist.",
                "tests": ["fixture prediction reflects the worktree-specific default bias"],
            }
        elif role == "diagnosis":
            artifact_ids = list(context.get("artifact_ids", []))
            if not artifact_ids:
                raise RuntimeError("fixture diagnosis requires an evidence artifact")
            primary = float(context.get("fixture_primary", 0.0))
            value = {
                "experiment_id": context["experiment_id"],
                "outcome": "supported" if primary > 0.5 else "contradicted",
                "evidence_artifact_ids": artifact_ids,
                "metric_deltas": {"fixture_primary": primary - 0.5},
                "uncertainty": "Synthetic fixture evidence is not scientific model evidence.",
                "next_operator": "ABANDON",
                "reusable_lesson": "The isolated fixture lifecycle completed with durable evidence.",
            }
        else:
            raise RuntimeError(f"unsupported fixture provider role: {role}")
        return ProviderResponse(value=value, provider="fixed", model="fixture-script")


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr[-2000:]}")
    return result.stdout.strip()


def _create_disposable_source(run_dir: Path) -> tuple[Path, str]:
    source = run_dir / "source"
    if (source / ".git").is_dir():
        if _git(source, "status", "--porcelain"):
            raise RuntimeError("disposable fixture source is not clean")
        return source, _git(source, "rev-parse", "HEAD")
    if source.exists():
        shutil.rmtree(source)
    (source / "tests").mkdir(parents=True)
    shutil.copytree(repo_root() / "src", source / "src")
    shutil.copytree(repo_root() / "tests/fixture", source / "tests/fixture")
    # The worker's label firewall reads the frozen inference-column contract
    # from its own committed checkout.  Copy only immutable contracts here;
    # competition data and organizer code remain outside the fixture worktree.
    shutil.copytree(repo_root() / "configs/frozen", source / "configs/frozen")
    (source / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    _git(source, "init")
    _git(source, "config", "user.email", "rex-fixture@example.invalid")
    _git(source, "config", "user.name", "REX Fixture")
    _git(source, "add", "--all")
    _git(source, "commit", "-m", "fixture source snapshot")
    return source, _git(source, "rev-parse", "HEAD")


def _write_fixture_views(run_dir: Path) -> tuple[Path, Path]:
    data = run_dir / "fixture-data"
    data.mkdir(parents=True, exist_ok=True)
    features = data / "features.npz"
    targets = data / "targets.npz"
    if not features.exists():
        np.savez_compressed(
            features,
            row_id=np.arange(8, dtype=np.int64),
            date=np.asarray(
                [20220408, 20220408, 20220409, 20220409, 20220410, 20220410, 20220411, 20220411]
            ),
            user_id=np.asarray(["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"]),
            video_id=np.asarray(["v1", "v2", "v1", "v3", "v2", "v4", "v1", "v4"]),
            author_id=np.asarray(["a1", "a2", "a1", "a3", "a2", "a4", "a1", "a4"]),
            tab=np.asarray(["1"] * 8),
            duration_ms=np.asarray([10, 20, 10, 30, 20, 40, 10, 40], dtype=np.float32),
        )
        np.savez_compressed(
            targets,
            long_view=np.asarray([1, 0, 0, 1, 1, 0, 0, 1], dtype=np.float32),
        )
    return features, targets


def next_fixture_action(state: ExperimentState) -> str:
    """Pure recovery dispatcher, intentionally testable for every lifecycle state."""
    actions = {
        ExperimentState.PROPOSED: "prepare_worktree",
        ExperimentState.WORKTREE_READY: "apply_patch",
        ExperimentState.PATCHED: "static_gate",
        ExperimentState.STATIC_VALID: "fixture_gate",
        ExperimentState.FIXTURE_VALID: "start_cheap",
        ExperimentState.CHEAP_RUNNING: "resume_cheap",
        ExperimentState.CHEAP_COMPLETE: "decide_cheap",
        ExperimentState.FULL_RESERVED: "start_full",
        ExperimentState.FULL_RUNNING: "resume_full",
        ExperimentState.FULL_COMPLETE: "diagnose",
        ExperimentState.DIAGNOSED: "close_fixture",
        ExperimentState.FAILED_REPAIRABLE: "repair_or_fail",
        ExperimentState.REPAIRING: "resume_repair",
    }
    return actions.get(state, "terminal")


class FixtureAutopilot:
    def __init__(self, config: FixtureRunConfig, provider: StructuredProvider):
        self.config = config
        self.provider = provider
        self.budget = BudgetConfig.from_yaml(config.budget_config)

    def run(self, *, run_id: str | None = None) -> dict[str, Any]:
        identifier = run_id or f"fixture-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir = self.config.runs_dir / identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        database = Database(run_dir / "state.sqlite3")
        database.initialize()
        repository = ExperimentRepository(database)
        source, root_commit = _create_disposable_source(run_dir)
        features, targets = _write_fixture_views(run_dir)
        try:
            run = repository.get_run(identifier)
        except RepositoryError:
            repository.create_run(
                run_id=identifier,
                deadline_epoch_ms=deadline_epoch_ms(
                    min(self.config.wall_clock_seconds, self.budget.wall_clock_seconds)
                ),
                root_commit=root_commit,
                environment_sha256=HASH,
                data_manifest_sha256=sha256_file(features),
                evaluator_sha256=HASH,
            )
            repository.transition_run(
                identifier, RunState.INITIALIZING, RunState.BASELINE_VERIFYING
            )
            repository.transition_run(
                identifier, RunState.BASELINE_VERIFYING, RunState.SEARCHING
            )
            run = repository.get_run(identifier)
        if root_commit != run["root_commit"]:
            raise RuntimeError(
                "disposable fixture source commit does not match the durable run root"
            )
        if RunState(run["state"]) == RunState.COMPLETE:
            return {
                **self._result(database, identifier, run_dir),
                "report": build_report(database, identifier, run_dir / "report"),
            }
        resumable_states = {
            RunState.SEARCHING,
            RunState.BUDGET_EXHAUSTED,
            RunState.FINALIZING,
        }
        if RunState(run["state"]) not in resumable_states:
            raise RuntimeError(f"fixture run cannot resume from state {run['state']}")
        session_id = f"session-{uuid.uuid4().hex}"
        session_started = time.monotonic()
        session_closed = False
        repository.open_process_session(
            session_id=session_id,
            run_id=identifier,
            stale_after_seconds=(
                self.config.process_stale_after_seconds if run_id is not None else None
            ),
        )
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(5):
                try:
                    repository.heartbeat_process_session(
                        session_id, time.monotonic() - session_started
                    )
                except RepositoryError:
                    # A transient database lock is retried on the next tick. The
                    # stale threshold is deliberately much longer than the tick.
                    continue

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"rex-heartbeat-{identifier}",
            daemon=True,
        )
        heartbeat_thread.start()

        def finish_fixture_run() -> dict[str, Any]:
            nonlocal session_closed
            build_report(database, identifier, run_dir / "report")
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=10)
            repository.close_process_session(
                session_id,
                exit_reason="fixture_complete",
                monotonic_seconds=time.monotonic() - session_started,
            )
            session_closed = True
            repository.transition_run(identifier, RunState.FINALIZING, RunState.COMPLETE)
            report = build_report(database, identifier, run_dir / "report")
            return {**self._result(database, identifier, run_dir), "report": report}

        try:
            current_state = RunState(repository.get_run(identifier)["state"])
            if current_state == RunState.BUDGET_EXHAUSTED:
                repository.transition_run(
                    identifier, RunState.BUDGET_EXHAUSTED, RunState.FINALIZING
                )
                current_state = RunState.FINALIZING
            if current_state == RunState.FINALIZING:
                return finish_fixture_run()
            with database.connect() as connection:
                active_rows = connection.execute(
                    "SELECT * FROM experiments WHERE run_id=? AND state NOT IN "
                    "('PROMOTED','REJECTED','ABANDONED','FAILED_FINAL') "
                    "ORDER BY iteration_number",
                    (identifier,),
                ).fetchall()
                completed = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM experiments WHERE run_id=?", (identifier,)
                    ).fetchone()[0]
                )
            for row in active_rows:
                if not row["workspace_path"] or not row["branch_name"] or not row["commit_sha"]:
                    self._terminalize_incomplete_preparation(repository, dict(row))
                    continue
                prepared = PreparedExperiment(
                    proposal=ExperimentProposal.model_validate_json(row["proposal_json"]),
                    workspace=GitWorkspace(Path(row["workspace_path"]), row["branch_name"]),
                    commit_sha=row["commit_sha"],
                    log_artifact_ids=(),
                )
                self._execute_candidate(
                    repository, identifier, prepared, features, targets, run_dir
                )
            maximum = min(self.config.max_hypotheses, self.budget.max_hypotheses)
            stop_reason = "fixture_hypothesis_cap"
            for iteration_index in range(completed, maximum):
                current_run = repository.get_run(identifier)
                if current_run["stop_reason"] == "epsilon_plateau":
                    stop_reason = "epsilon_plateau"
                    break
                if int(time.time() * 1000) >= int(current_run["deadline_epoch_ms"]):
                    stop_reason = "fixture_wall_clock_exhausted"
                    repository.transition_run(
                        identifier,
                        RunState.SEARCHING,
                        RunState.BUDGET_EXHAUSTED,
                        stop_reason,
                    )
                    break
                try:
                    prepared = self._prepare(
                        repository,
                        identifier,
                        source,
                        root_commit,
                        run_dir,
                        iteration_number=iteration_index + 1,
                    )
                except RuntimeError:
                    with database.connect() as connection:
                        interrupted = connection.execute(
                            "SELECT * FROM experiments WHERE run_id=? AND iteration_number=?",
                            (identifier, iteration_index + 1),
                        ).fetchone()
                    if interrupted is None:
                        raise
                    self._terminalize_incomplete_preparation(
                        repository, dict(interrupted)
                    )
                    continue
                self._execute_candidate(
                    repository, identifier, prepared, features, targets, run_dir
                )
            current_state = RunState(repository.get_run(identifier)["state"])
            if current_state == RunState.BUDGET_EXHAUSTED:
                repository.transition_run(
                    identifier, RunState.BUDGET_EXHAUSTED, RunState.FINALIZING
                )
            else:
                repository.transition_run(
                    identifier, RunState.SEARCHING, RunState.FINALIZING, stop_reason
                )
            return finish_fixture_run()
        except BaseException:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=10)
            if not session_closed:
                repository.close_process_session(
                    session_id,
                    exit_reason="fixture_interrupted",
                    monotonic_seconds=time.monotonic() - session_started,
                )
            raise

    @staticmethod
    def _terminalize_incomplete_preparation(
        repository: ExperimentRepository,
        row: dict[str, Any],
    ) -> None:
        """Fail closed after a kill in a pre-commit patch-preparation state.

        The hypothesis remains counted and auditable, the partially prepared
        workspace is never executed, and the resumed run may safely continue
        with the next transaction.
        """

        state = ExperimentState(row["state"])
        if state == ExperimentState.PROPOSED:
            target = ExperimentState.ABANDONED
        elif state in {
            ExperimentState.WORKTREE_READY,
            ExperimentState.PATCHED,
            ExperimentState.STATIC_VALID,
            ExperimentState.FIXTURE_VALID,
        }:
            target = ExperimentState.REJECTED
        elif state == ExperimentState.FAILED_REPAIRABLE:
            target = ExperimentState.ABANDONED
        elif state == ExperimentState.REPAIRING:
            target = ExperimentState.FAILED_FINAL
        else:
            raise RuntimeError(
                f"experiment {row['experiment_id']} has incomplete provenance in state {state}"
            )
        repository.transition_experiment(
            str(row["experiment_id"]),
            state,
            target,
            payload={
                "reason": "coordinator interruption before a durable, clean candidate commit"
            },
            idempotency_key=f"{row['experiment_id']}:incomplete-preparation-closed",
        )

    def _prepare(
        self,
        repository: ExperimentRepository,
        run_id: str,
        source: Path,
        parent_commit: str,
        run_dir: Path,
        *,
        iteration_number: int,
    ) -> PreparedExperiment:
        coordinator = PatchTransactionCoordinator(
            repository=repository,
            proposal_service=ProposalService(self.provider),
            coding_service=CodingService(self.provider),
            project_root=source,
            worktree_root=run_dir / "worktrees",
            patch_policy=PatchPolicy.from_yaml(self.config.protected_paths),
            static_command=(sys.executable, "-m", "compileall", "-q", "src"),
            fixture_command=(
                sys.executable,
                "-c",
                "import sys;sys.path.insert(0,'src');"
                "from rex.models.experimental.fixture import FixturePlugin;assert FixturePlugin",
            ),
        )
        return coordinator.prepare(
            run_id=run_id,
            parent_commit=parent_commit,
            proposal_context={
                "fixture_only": True,
                "fixture_iteration_number": iteration_number,
                "artifact_ids": [],
                "warning": "No competition data, scientific claims, confirmation, or submission.",
            },
            coding_context={
                "fixture_only": True,
                "allowed_file": "src/rex/models/experimental/fixture.py",
            },
            max_hypotheses=min(self.config.max_hypotheses, self.budget.max_hypotheses),
        )

    def _request(
        self,
        *,
        run_id: str,
        experiment_id: str,
        commit_sha: str,
        workspace: Path,
        config_path: Path,
        features: Path,
        targets: Path | None,
        output: Path,
        rung: str,
        model_bundle_path: str | None = None,
        run_deadline_epoch_ms: int | None = None,
    ) -> RunRequest:
        values: dict[str, Any] = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "attempt_id": f"{experiment_id}-{rung}-{uuid.uuid4().hex[:8]}",
            "commit_sha": commit_sha,
            "plugin": "rex.models.experimental.fixture:FixturePlugin",
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "seed": 0,
            "rung": "predict" if rung.endswith("predict") else rung,
            "split": "shadow",
            "fold": "fixture",
            "feature_view_path": str(features),
            "target_view_path": str(targets) if targets else None,
            "output_dir": str(output),
            "deadline_epoch_ms": min(
                deadline_epoch_ms(self.config.attempt_timeout_seconds + 10),
                run_deadline_epoch_ms or 2**63 - 1,
            ),
            "timeout_seconds": self.config.attempt_timeout_seconds,
            "max_memory_mb": 512,
            "data_view_sha256": sha256_file(features),
            "environment_sha256": HASH,
            "workspace_path": str(workspace),
            "operation": "predict" if rung.endswith("predict") else "fit",
            "model_bundle_path": model_bundle_path,
        }
        return RunRequest(**values)

    @staticmethod
    def _successful_artifact(
        repository: ExperimentRepository,
        experiment_id: str,
        rung: str,
        kind: str,
    ) -> dict[str, Any] | None:
        with repository.database.connect() as connection:
            row = connection.execute(
                "SELECT link.artifact_id,link.artifact_path,artifact.sha256,attempt.attempt_id "
                "FROM attempts attempt JOIN artifact_links link "
                "ON link.attempt_id=attempt.attempt_id JOIN artifacts artifact "
                "ON artifact.artifact_id=link.artifact_id "
                "WHERE attempt.experiment_id=? AND attempt.rung=? AND attempt.status=? "
                "AND artifact.kind=? ORDER BY attempt.ended_at DESC LIMIT 1",
                (experiment_id, rung, AttemptStatus.SUCCESS, kind),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        path = Path(result["artifact_path"])
        if not path.is_file() or sha256_file(path) != result["sha256"]:
            raise RuntimeError(
                f"durable successful artifact is missing or corrupt: {result['artifact_id']}"
            )
        return result

    def _execute_bounded(
        self,
        repository: ExperimentRepository,
        *,
        experiment_id: str,
        rung: str,
        commit_sha: str,
        request_factory: Callable[[], RunRequest],
        attempt_directory: Path,
        trusted_worktree_root: Path,
    ) -> RunResult:
        with repository.database.connect() as connection:
            prior = connection.execute(
                "SELECT repair_number,status FROM attempts WHERE experiment_id=? AND rung=? "
                "ORDER BY repair_number DESC,ended_at DESC",
                (experiment_id, rung),
            ).fetchall()
        failures = [row for row in prior if row["status"] != AttemptStatus.SUCCESS]
        start_repair = 0 if not failures else max(int(row["repair_number"]) for row in failures) + 1
        maximum = self.budget.max_repairs_per_experiment
        if start_repair > maximum:
            raise RuntimeError(
                f"fixture {rung} repair limit {maximum} was already exhausted"
            )
        last: RunResult | None = None
        for repair_number in range(start_repair, maximum + 1):
            request = request_factory()
            repository.reserve_attempt(
                attempt_id=request.attempt_id,
                experiment_id=experiment_id,
                rung=rung,
                repair_number=repair_number,
                commit_sha=commit_sha,
            )
            last = execute_request(
                request,
                attempt_directory / f"repair-{repair_number}",
                trusted_worktree_root=trusted_worktree_root,
            )
            self._ingest_result(repository, last, rung, repair_number=repair_number)
            if last.status == AttemptStatus.SUCCESS:
                return last
            repair = decide_repair(last.status, repair_number, maximum=maximum)
            if not repair.repair:
                break
        assert last is not None
        raise RuntimeError(
            f"fixture {rung} failed after bounded repairs: "
            f"{last.status}: {last.error_summary}"
        )

    def _run_rung(
        self,
        repository: ExperimentRepository,
        *,
        run_id: str,
        prepared: PreparedExperiment,
        features: Path,
        targets: Path,
        run_dir: Path,
        rung: str,
    ) -> tuple[float, list[str]]:
        experiment_id = prepared.proposal.experiment_id
        with repository.database.connect() as connection:
            existing_metric = connection.execute(
                "SELECT primary_score FROM metrics WHERE experiment_id=? AND split='fixture' "
                "AND fold=? AND seed=0",
                (experiment_id, rung),
            ).fetchone()
            existing_evidence = [
                row["artifact_id"]
                for row in connection.execute(
                    "SELECT DISTINCT artifact_id FROM artifact_links WHERE experiment_id=? "
                    "ORDER BY artifact_id",
                    (experiment_id,),
                )
            ]
        if existing_metric is not None:
            return float(existing_metric["primary_score"]), existing_evidence
        config_path = run_dir / "configs" / f"{experiment_id}.json"
        experiment_config: dict[str, Any] = {"fixture_only": True}
        if (
            self.config.inject_worker_nan_once
            and experiment_id == "fixture-001"
        ):
            experiment_config["raise_floating_point_once_marker"] = str(
                run_dir / "faults" / "fixture-001-cheap-nan-once.marker"
            )
        if (
            self.config.inject_worker_nan_always_iteration is not None
            and experiment_id
            == f"fixture-{self.config.inject_worker_nan_always_iteration:03d}"
            and rung == "cheap"
        ):
            experiment_config["raise_floating_point"] = True
        atomic_write_json(config_path, experiment_config)
        run_deadline = int(repository.get_run(run_id)["deadline_epoch_ms"])
        repository.record_experiment_workspace(
            experiment_id,
            workspace_path=str(prepared.workspace.root),
            branch_name=prepared.workspace.branch,
            commit_sha=prepared.commit_sha,
            config_sha256=sha256_file(config_path),
        )
        checkpoint = self._successful_artifact(
            repository, experiment_id, rung, "model_bundle"
        )
        if checkpoint is None:
            train = self._execute_bounded(
                repository,
                experiment_id=experiment_id,
                rung=rung,
                commit_sha=prepared.commit_sha,
                request_factory=lambda: self._request(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    commit_sha=prepared.commit_sha,
                    workspace=prepared.workspace.root,
                    config_path=config_path,
                    features=features,
                    targets=targets,
                    output=(
                        run_dir
                        / "attempts"
                        / experiment_id
                        / f"{rung}-fit-output-{uuid.uuid4().hex[:8]}"
                    ),
                    rung=rung,
                    run_deadline_epoch_ms=run_deadline,
                ),
                attempt_directory=run_dir / "attempts" / experiment_id / f"{rung}-fit",
                trusted_worktree_root=run_dir / "worktrees",
            )
            checkpoint_ref = next(item for item in train.artifacts if item.kind == "model_bundle")
            checkpoint = {
                "artifact_id": checkpoint_ref.artifact_id,
                "artifact_path": checkpoint_ref.path,
                "attempt_id": train.attempt_id,
            }

        predict_rung = f"{rung}-predict"
        predictions = self._successful_artifact(
            repository, experiment_id, predict_rung, "predictions"
        )
        if predictions is None:
            prediction = self._execute_bounded(
                repository,
                experiment_id=experiment_id,
                rung=predict_rung,
                commit_sha=prepared.commit_sha,
                request_factory=lambda: self._request(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    commit_sha=prepared.commit_sha,
                    workspace=prepared.workspace.root,
                    config_path=config_path,
                    features=features,
                    targets=None,
                    output=(
                        run_dir
                        / "attempts"
                        / experiment_id
                        / f"{rung}-predict-output-{uuid.uuid4().hex[:8]}"
                    ),
                    rung=predict_rung,
                    model_bundle_path=str(checkpoint["artifact_path"]),
                    run_deadline_epoch_ms=run_deadline,
                ),
                attempt_directory=run_dir / "attempts" / experiment_id / predict_rung,
                trusted_worktree_root=run_dir / "worktrees",
            )
            prediction_ref = next(
                item for item in prediction.artifacts if item.kind == "predictions"
            )
            predictions = {
                "artifact_id": prediction_ref.artifact_id,
                "artifact_path": prediction_ref.path,
                "attempt_id": prediction.attempt_id,
            }
        payload = load_prediction_artifact(str(predictions["artifact_path"]), features)
        primary = float(np.mean(payload["score"]))
        primary = min(1.0, max(0.0, primary))
        metrics = Metrics(
            GAUC=primary,
            **{"nDCG@5": primary},
            primary=primary,
            users=4,
            rows=8,
            evaluator_sha256=HASH,
            split="fixture",
            fold=rung,
            seed=0,
        )
        repository.record_metrics(experiment_id, metrics, str(predictions["attempt_id"]))
        metrics_path = run_dir / "evidence" / experiment_id / f"{rung}-metrics.json"
        atomic_write_json(metrics_path, metrics.model_dump(mode="json", by_alias=True))
        metrics_ref = artifact_ref(metrics_path, "fixture_metrics")
        repository.register_artifact(metrics_ref, experiment_id=experiment_id)
        return primary, [str(predictions["artifact_id"]), metrics_ref.artifact_id]

    @staticmethod
    def _ingest_result(
        repository: ExperimentRepository,
        result: RunResult,
        rung: str,
        *,
        repair_number: int = 0,
    ) -> None:
        for ref in result.artifacts:
            repository.register_artifact(
                ref, experiment_id=result.experiment_id, attempt_id=result.attempt_id
            )
        repository.record_attempt(result, rung=rung, repair_number=repair_number)

    def _execute_candidate(
        self,
        repository: ExperimentRepository,
        run_id: str,
        prepared: PreparedExperiment,
        features: Path,
        targets: Path,
        run_dir: Path,
    ) -> None:
        experiment_id = prepared.proposal.experiment_id
        while True:
            state = ExperimentState(repository.get_experiment(experiment_id)["state"])
            if state == ExperimentState.FIXTURE_VALID:
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.CHEAP_RUNNING,
                    idempotency_key=f"{experiment_id}:cheap-running",
                )
                continue
            if state == ExperimentState.CHEAP_RUNNING:
                try:
                    cheap, _ = self._run_rung(
                        repository,
                        run_id=run_id,
                        prepared=prepared,
                        features=features,
                        targets=targets,
                        run_dir=run_dir,
                        rung="cheap",
                    )
                except RuntimeError as error:
                    repository.transition_experiment(
                        experiment_id,
                        state,
                        ExperimentState.FAILED_FINAL,
                        payload={"reason": str(error)[-1000:]},
                        idempotency_key=f"{experiment_id}:cheap-failed-final",
                    )
                    return
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.CHEAP_COMPLETE,
                    payload={"fixture_primary": cheap},
                    idempotency_key=f"{experiment_id}:cheap-complete",
                )
                continue
            if state == ExperimentState.CHEAP_COMPLETE:
                cheap = self._stored_fixture_primary(repository, experiment_id, "cheap")
                if cheap - 0.5 < self.config.full_threshold:
                    repository.reject_non_improving(
                        run_id=run_id,
                        experiment_id=experiment_id,
                        expected_state=state,
                        reason="synthetic cheap gate rejected the fixture variant",
                        patience=self.budget.convergence_patience,
                        idempotency_key=f"{experiment_id}:fixture-rejected",
                    )
                    return
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.FULL_RESERVED,
                    idempotency_key=f"{experiment_id}:full-reserved",
                )
                continue
            if state == ExperimentState.FULL_RESERVED:
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.FULL_RUNNING,
                    idempotency_key=f"{experiment_id}:full-running",
                )
                continue
            if state == ExperimentState.FULL_RUNNING:
                try:
                    full, _ = self._run_rung(
                        repository,
                        run_id=run_id,
                        prepared=prepared,
                        features=features,
                        targets=targets,
                        run_dir=run_dir,
                        rung="full",
                    )
                except RuntimeError as error:
                    repository.transition_experiment(
                        experiment_id,
                        state,
                        ExperimentState.FAILED_FINAL,
                        payload={"reason": str(error)[-1000:]},
                        idempotency_key=f"{experiment_id}:full-failed-final",
                    )
                    return
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.FULL_COMPLETE,
                    payload={"fixture_primary": full},
                    idempotency_key=f"{experiment_id}:full-complete",
                )
                continue
            if state == ExperimentState.FULL_COMPLETE:
                self._diagnose_fixture(repository, run_id, experiment_id, run_dir)
                continue
            if state == ExperimentState.DIAGNOSED:
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.ABANDONED,
                    payload={"reason": "fixture completed; production promotion is disabled"},
                    idempotency_key=f"{experiment_id}:fixture-closed",
                )
                return
            if state in {
                ExperimentState.REJECTED,
                ExperimentState.ABANDONED,
                ExperimentState.FAILED_FINAL,
            }:
                return
            raise RuntimeError(
                f"fixture candidate {experiment_id} cannot resume from {state}: "
                f"next action is {next_fixture_action(state)}"
            )

    @staticmethod
    def _stored_fixture_primary(
        repository: ExperimentRepository, experiment_id: str, fold: str
    ) -> float:
        with repository.database.connect() as connection:
            row = connection.execute(
                "SELECT primary_score FROM metrics WHERE experiment_id=? AND split='fixture' "
                "AND fold=? AND seed=0",
                (experiment_id, fold),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"fixture metric is missing for {experiment_id}/{fold}")
        return float(row["primary_score"])

    def _diagnose_fixture(
        self,
        repository: ExperimentRepository,
        run_id: str,
        experiment_id: str,
        run_dir: Path,
    ) -> None:
        full = self._stored_fixture_primary(repository, experiment_id, "full")
        with repository.database.connect() as connection:
            evidence = [
                row["artifact_id"]
                for row in connection.execute(
                    "SELECT DISTINCT artifact_id FROM artifact_links WHERE experiment_id=? "
                    "ORDER BY artifact_id",
                    (experiment_id,),
                )
            ]
        diagnosis_context = {
            "fixture_only": True,
            "artifact_ids": evidence,
            "fixture_primary": full,
            "warning": "Synthetic fixture evidence only; no model-quality claim.",
        }
        decision = DiagnosisService(self.provider).diagnose(
            experiment_id, diagnosis_context
        )
        diagnosis_request_path = (
            run_dir / "evidence" / experiment_id / "diagnosis-request.json"
        )
        reflection_path = run_dir / "evidence" / experiment_id / "diagnosis.json"
        atomic_write_json(diagnosis_request_path, diagnosis_context)
        atomic_write_json(
            reflection_path,
            {
                "decision": decision.parsed.model_dump(mode="json", by_alias=True),
                "provider": decision.response.provider,
                "model": decision.response.model,
                "request_id": decision.response.request_id,
                "input_tokens": decision.response.input_tokens,
                "output_tokens": decision.response.output_tokens,
                "wall_seconds": decision.response.wall_seconds,
                "attempts": decision.response.attempts,
                "fallback_chain": list(decision.response.fallback_chain),
                "fallback_errors": list(decision.response.fallback_errors),
            },
        )
        diagnosis_request_ref = artifact_ref(diagnosis_request_path, "llm_request")
        reflection_ref = artifact_ref(reflection_path, "fixture_diagnosis")
        repository.register_artifact(diagnosis_request_ref, experiment_id=experiment_id)
        repository.register_artifact(reflection_ref, experiment_id=experiment_id)
        repository.record_llm_call(
            call_id=f"{experiment_id}:diagnosis",
            run_id=run_id,
            experiment_id=experiment_id,
            role="diagnosis",
            provider=decision.response.provider,
            model=decision.response.model,
            request_artifact_id=diagnosis_request_ref.artifact_id,
            response_artifact_id=reflection_ref.artifact_id,
            schema_valid=decision.response.schema_valid,
            input_tokens=decision.response.input_tokens,
            output_tokens=decision.response.output_tokens,
            wall_seconds=decision.response.wall_seconds,
            request_id=decision.response.request_id,
        )
        remember_reflection(repository, run_id, decision.parsed)
        repository.transition_experiment(
            experiment_id,
            ExperimentState.FULL_COMPLETE,
            ExperimentState.DIAGNOSED,
            payload={"diagnosis_artifact_id": reflection_ref.artifact_id},
            idempotency_key=f"{experiment_id}:diagnosed",
        )

    @staticmethod
    def _result(database: Database, run_id: str, run_dir: Path) -> dict[str, Any]:
        with database.connect() as connection:
            run = dict(connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())
            experiments = [
                dict(row)
                for row in connection.execute(
                    "SELECT experiment_id,state,iteration_number FROM experiments "
                    "WHERE run_id=? ORDER BY iteration_number",
                    (run_id,),
                )
            ]
        return {
            "run_id": run_id,
            "execution_mode": "fixture",
            "state": run["state"],
            "stop_reason": run["stop_reason"],
            "hypothesis_count": run["hypothesis_count"],
            "non_improvement_streak": run["non_improvement_streak"],
            "run_dir": str(run_dir),
            "experiments": experiments,
            "production_science_enabled": False,
            "confirmation_enabled": False,
            "final_submission_enabled": False,
        }


def run_fixture_autopilot(
    config_path: str | Path,
    *,
    provider: StructuredProvider | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    config = FixtureRunConfig.load(config_path)
    return FixtureAutopilot(config, provider or FixtureScriptProvider()).run(run_id=run_id)
