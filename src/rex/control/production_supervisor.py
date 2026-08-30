"""Durable, validation-only production autopilot with scientific execution hooks.

The restart-safe lifecycle runs baseline, cheap, full-shadow, and official-valid
gates. It deliberately excludes confirmation sweeps, test prediction, and final
competition submission.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from rex.agents.memory import remember_reflection
from rex.agents.provider import ProviderResponse, StructuredProvider
from rex.agents.recovery import RepairAction, TypedRepairPlan, plan_repair
from rex.agents.search_policy import (
    METHOD_CARD_REFERENCES,
    METHOD_CARD_VERSION,
    ExperimentCard,
    SearchPolicy,
)
from rex.agents.services import DiagnosisService
from rex.contracts import (
    ArtifactRef,
    AttemptStatus,
    ExperimentProposal,
    ExperimentState,
    Metrics,
    RunState,
)
from rex.control.budget import BudgetConfig, deadline_epoch_ms, seconds_remaining, should_finalize
from rex.data.manifest import canonical_json_bytes, repo_root, sha256_file
from rex.execution.artifacts import artifact_ref, atomic_write_json
from rex.reporting.finalizer import create_best_valid_bundle
from rex.reporting.report import build_report
from rex.store.db import Database
from rex.store.repository import ExperimentRepository, RepositoryError


CARD_CODE_PATHS: dict[str, tuple[str, ...]] = {
    "E01": ("src/rex/models/experimental/pair_rank_fm.py",),
    "E02": ("src/rex/models/experimental/tree_history.py",),
    "E03": ("src/rex/models/experimental/tree_history.py",),
    "E04": ("src/rex/models/experimental/pair_rank_fm.py",),
    "E05": ("src/rex/models/experimental/pair_rank_fm.py",),
    "E06": ("src/rex/models/experimental/tree_history.py",),
    "E07": ("src/rex/models/experimental/tree_history.py",),
    "E08": ("src/rex/models/experimental/tree_history.py",),
    "E10": (),
}
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class MethodCardBinding:
    config_path: Path
    feature_recipe: str


@dataclass(frozen=True)
class ProductionRunConfig:
    source_path: Path
    project_root: Path
    runs_dir: Path
    budget_config: Path
    protected_paths: Path
    data_manifest: Path
    evaluator_path: Path
    environment_lock: Path
    scientific_execution_enabled: bool
    process_stale_after_seconds: int
    cleanup_worktrees: bool
    method_cards: dict[str, MethodCardBinding]
    llm: dict[str, Any]
    raw_data_dir: Path | None = None
    scientific_execution: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ProductionRunConfig":
        candidate = Path(path).resolve()
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("production run configuration must be a YAML mapping")
        if raw.get("execution_mode") != "production":
            raise RuntimeError("production configuration requires execution_mode: production")
        root = repo_root()

        def resolve(value: str) -> Path:
            item = Path(value)
            return item.resolve() if item.is_absolute() else (root / item).resolve()

        bindings: dict[str, MethodCardBinding] = {}
        for card_id, value in dict(raw.get("method_cards", {})).items():
            if value is None:
                continue
            if not isinstance(value, dict) or "config" not in value or "feature_recipe" not in value:
                raise ValueError(f"method card {card_id} needs config and feature_recipe")
            supported_cards = {"E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E10"}
            if card_id not in supported_cards:
                raise ValueError(
                    f"method card {card_id} is unsupported or deferred in this implementation phase"
                )
            bindings[str(card_id)] = MethodCardBinding(
                config_path=resolve(str(value["config"])),
                feature_recipe=str(value["feature_recipe"]),
            )
        scientific_execution = raw.get("scientific_execution", {})
        if not isinstance(scientific_execution, dict):
            raise ValueError("scientific_execution must be a mapping")
        config = cls(
            source_path=candidate,
            project_root=resolve(str(raw.get("project_root", "."))),
            runs_dir=resolve(str(raw.get("runs_dir", "runs"))),
            budget_config=resolve(str(raw.get("budget_config", "configs/budget.yaml"))),
            protected_paths=resolve(
                str(raw.get("protected_paths", "configs/security/protected_paths.yaml"))
            ),
            data_manifest=resolve(str(raw.get("data_manifest", "runs/data/manifest.json"))),
            evaluator_path=resolve(
                str(raw.get("evaluator_path", "kuairand-starter-kit/evaluate.py"))
            ),
            environment_lock=resolve(str(raw.get("environment_lock", "requirements-lock.txt"))),
            scientific_execution_enabled=bool(raw.get("scientific_execution_enabled", False)),
            process_stale_after_seconds=int(raw.get("process_stale_after_seconds", 900)),
            cleanup_worktrees=bool(raw.get("cleanup_worktrees", True)),
            method_cards=bindings,
            llm=dict(raw.get("llm", {})),
            raw_data_dir=resolve(
                str(raw.get("raw_data_dir", "data/KuaiRand-Pure/data"))
            ),
            scientific_execution=dict(scientific_execution),
        )
        if config.process_stale_after_seconds <= 0:
            raise ValueError("process_stale_after_seconds must be positive")
        return config


@dataclass(frozen=True)
class ProductionContext:
    run_id: str
    run_dir: Path
    project_root: Path
    root_commit: str
    deadline_epoch_ms: int


@dataclass(frozen=True)
class BaselineGateResult:
    accepted: bool
    metrics: Metrics | None
    artifacts: tuple[ArtifactRef, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class CandidatePreparation:
    proposal: ExperimentProposal
    commit_sha: str
    workspace_path: Path | None = None
    branch_name: str | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    effective_config_path: Path | None = None
    effective_config_sha256: str | None = None


@dataclass(frozen=True)
class ComparisonObservation:
    candidate: Metrics
    reference: Metrics

    @property
    def primary_delta(self) -> float:
        return self.candidate.primary - self.reference.primary

    @property
    def gauc_delta(self) -> float:
        return self.candidate.GAUC - self.reference.GAUC

    @property
    def ndcg_delta(self) -> float:
        return self.candidate.ndcg5 - self.reference.ndcg5


@dataclass(frozen=True)
class ProductionRungResult:
    observations: tuple[ComparisonObservation, ...]
    artifacts: tuple[ArtifactRef, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RungRequest:
    context: ProductionContext
    experiment: dict[str, Any]
    method_card: ExperimentCard
    binding: MethodCardBinding
    rung: str


@dataclass(frozen=True)
class RepairRequest:
    context: ProductionContext
    experiment: dict[str, Any]
    phase: str
    plan: TypedRepairPlan


class ProductionRungFailure(RuntimeError):
    def __init__(self, status: AttemptStatus, message: str):
        super().__init__(message)
        self.status = status


class ProductionPreparationRejected(RuntimeError):
    """A bounded candidate preparation failed safely and the search may continue."""


class ProductionHooks(Protocol):
    """Model/data adapter boundary; implementations must be idempotent by request identity."""

    def verify_baseline(self, context: ProductionContext) -> BaselineGateResult: ...

    def prepare_candidate(
        self,
        context: ProductionContext,
        card: ExperimentCard,
        binding: MethodCardBinding,
        proposal_context: dict[str, object],
        parent_commit: str,
    ) -> CandidatePreparation: ...

    def run_rung(self, request: RungRequest) -> ProductionRungResult: ...

    def repair_candidate(self, request: RepairRequest) -> tuple[ArtifactRef, ...]: ...


class ProductionFixedProvider:
    """Deterministic production decisions for config-backed method cards.

    Config-driven hooks bypass an LLM-authored patch. The coordinator still
    records the versioned card/config decision as evidence. A patch request is
    therefore rejected instead of fabricating code.
    """

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        del system, schema
        context = json.loads(prompt)
        if role == "proposal":
            card = dict(context["method_card"])
            card_id = str(card["card_id"])
            detail = METHOD_CARD_REFERENCES[card_id]
            operator = {
                "E01": "LOSS",
                "E02": "HYPERPARAMETER",
                "E03": "FEATURE",
                "E04": "LOSS",
                "E05": "LOSS",
                "E06": "FEATURE",
                "E07": "FEATURE",
                "E08": "FEATURE",
                "E10": "ENSEMBLE",
            }[card_id]
            value = {
                "experiment_id": str(context.get("experiment_id", f"fixed-{card_id.lower()}")),
                "parent_id": context.get("incumbent", {}).get("experiment_id"),
                "operator": operator,
                "hypothesis": f"{detail['primary_change']} should improve validated ranking.",
                "mechanism": str(card["mechanism"]),
                "primary_change": detail["primary_change"],
                "files_to_change": [str(context["allowed_files"][0])],
                "expected_metric_effects": {"primary": "increase"},
                "falsifier": detail["falsifier"],
                "leakage_analysis": "Only prior-train state and shadow-selected settings are used.",
                "estimated_seconds": 600,
                "cheap_rung": {"fold": "A", "complete_users": True},
                "full_rung": {"folds": ["A", "B", "C"]},
            }
        elif role == "diagnosis":
            artifact_ids = list(context.get("artifact_ids", []))
            if not artifact_ids:
                raise RuntimeError("fixed diagnosis requires durable evidence artifact IDs")
            value = {
                "experiment_id": context["experiment_id"],
                "outcome": "inconclusive",
                "evidence_artifact_ids": artifact_ids,
                "metric_deltas": {
                    "mean_shadow_primary": sum(context.get("fold_metric_deltas", []))
                    / max(1, len(context.get("fold_metric_deltas", [])))
                },
                "uncertainty": "Deterministic diagnosis defers interpretation to trusted gates.",
                "next_operator": "ABANDON",
                "reusable_lesson": "Trusted temporal gates, not the fixed provider, decide promotion.",
            }
        elif role == "patch":
            raise RuntimeError(
                "fixed production mode uses a versioned config transaction and does not author patches"
            )
        else:
            raise RuntimeError(f"unsupported fixed production role: {role}")
        return ProviderResponse(value=value, provider="fixed", model="method-card-queue-v1")


def environment_provenance_sha256(config: ProductionRunConfig) -> str:
    """Hash the lock plus interpreter/platform and packaging metadata."""

    paths = [config.environment_lock, config.project_root / "pyproject.toml"]
    members: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "files": {},
    }
    for path in paths:
        if path.is_file():
            members["files"][str(path.resolve())] = sha256_file(path)
    return hashlib.sha256(canonical_json_bytes(members)).hexdigest()


class ProductionAutopilot:
    def __init__(
        self,
        config: ProductionRunConfig,
        provider: StructuredProvider,
        hooks: ProductionHooks | None = None,
        *,
        search_policy: SearchPolicy | None = None,
        now: Any = time.time,
        lifecycle_checkpoint: Callable[[str, str], None] | None = None,
    ):
        self.config = config
        self.provider = provider
        self.hooks = hooks
        self.search_policy = search_policy or SearchPolicy()
        self.budget = BudgetConfig.from_yaml(config.budget_config)
        self.now = now
        self.lifecycle_checkpoint = lifecycle_checkpoint

    def run(
        self,
        *,
        run_id: str | None = None,
        create_only: bool = False,
        resume_only: bool = False,
        external_deadline_epoch_ms: int | None = None,
    ) -> dict[str, Any]:
        if create_only and resume_only:
            raise ValueError("production run cannot be both create-only and resume-only")
        if not self.config.scientific_execution_enabled:
            return {
                "execution_mode": "production",
                "state": "DEFERRED",
                "stop_reason": "scientific_experiments_deferred_by_configuration",
                "production_control_plane_ready": True,
                "scientific_execution_enabled": False,
                "confirmation_enabled": False,
                "final_submission_enabled": False,
            }
        if self.hooks is None:
            raise RuntimeError("production scientific hooks are not configured")
        root_commit = self._clean_root_commit()
        self._require_input(config_path=self.config.data_manifest, name="data manifest")
        self._require_input(config_path=self.config.evaluator_path, name="evaluator")
        self._require_input(config_path=self.config.environment_lock, name="environment lock")
        identifier = run_id or f"production-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        if not SAFE_RUN_ID.fullmatch(identifier):
            raise RuntimeError("production run ID must be one safe path component")
        runs_root = self.config.runs_dir.resolve()
        run_dir = (runs_root / identifier).resolve()
        try:
            run_dir.relative_to(runs_root)
        except ValueError as error:  # pragma: no cover - regex is the primary guard
            raise RuntimeError("production run directory escapes the configured runs root") from error
        run_dir.mkdir(parents=True, exist_ok=True)
        database = Database(run_dir / "state.sqlite3")
        database.initialize()
        repository = ExperimentRepository(database)
        created = False
        try:
            run = repository.get_run(identifier)
        except RepositoryError:
            if resume_only:
                raise RuntimeError(f"unknown production run: {identifier}") from None
            durable_deadline = self._new_deadline(external_deadline_epoch_ms)
            repository.create_run(
                run_id=identifier,
                deadline_epoch_ms=durable_deadline,
                root_commit=root_commit,
                environment_sha256=environment_provenance_sha256(self.config),
                data_manifest_sha256=sha256_file(self.config.data_manifest),
                evaluator_sha256=sha256_file(self.config.evaluator_path),
            )
            run = repository.get_run(identifier)
            created = True
        if create_only and not created:
            raise RuntimeError(f"production run already exists; use --resume {identifier}")
        if run["root_commit"] != root_commit:
            raise RuntimeError("production run root commit differs from the durable snapshot")
        expected_environment = environment_provenance_sha256(self.config)
        if run["environment_sha256"] != expected_environment:
            raise RuntimeError("production run environment differs from the durable snapshot")
        if run["data_manifest_sha256"] != sha256_file(self.config.data_manifest):
            raise RuntimeError("production run data manifest differs from the durable snapshot")
        if run["evaluator_sha256"] != sha256_file(self.config.evaluator_path):
            raise RuntimeError("production run evaluator differs from the durable snapshot")
        if (
            external_deadline_epoch_ms is not None
            and int(run["deadline_epoch_ms"]) != int(external_deadline_epoch_ms)
        ):
            raise RuntimeError("production run deadline differs from the R3 envelope")
        if RunState(run["state"]) == RunState.COMPLETE:
            return self.status(identifier)

        context = ProductionContext(
            run_id=identifier,
            run_dir=run_dir,
            project_root=self.config.project_root,
            root_commit=root_commit,
            deadline_epoch_ms=int(run["deadline_epoch_ms"]),
        )
        session_id = f"production-session-{uuid.uuid4().hex}"
        session_started = time.monotonic()
        repository.open_process_session(
            session_id=session_id,
            run_id=identifier,
            stale_after_seconds=self.config.process_stale_after_seconds if run_id else None,
        )
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(repository, session_id, session_started, heartbeat_stop),
            daemon=True,
        )
        heartbeat.start()
        closed = False
        try:
            state = RunState(repository.get_run(identifier)["state"])
            if state == RunState.INITIALIZING:
                repository.transition_run(
                    identifier, RunState.INITIALIZING, RunState.BASELINE_VERIFYING
                )
                state = RunState.BASELINE_VERIFYING
            if state == RunState.BASELINE_VERIFYING:
                baseline = self.hooks.verify_baseline(context)
                for ref in baseline.artifacts:
                    repository.register_artifact(ref)
                if not baseline.accepted or baseline.metrics is None:
                    repository.transition_run(
                        identifier,
                        RunState.BASELINE_VERIFYING,
                        RunState.BASELINE_BLOCKED,
                        baseline.reason or "baseline gate rejected",
                    )
                    return self.status(identifier)
                repository.establish_baseline(
                    run_id=identifier,
                    metrics=baseline.metrics,
                    evidence_artifact_ids=[item.artifact_id for item in baseline.artifacts],
                )
                repository.transition_run(
                    identifier, RunState.BASELINE_VERIFYING, RunState.SEARCHING
                )
                state = RunState.SEARCHING
            if state in {RunState.BUDGET_EXHAUSTED, RunState.FINALIZING}:
                if state == RunState.BUDGET_EXHAUSTED:
                    repository.transition_run(
                        identifier, RunState.BUDGET_EXHAUSTED, RunState.FINALIZING
                    )
                return self._finalize(repository, context)
            if state != RunState.SEARCHING:
                return self.status(identifier)

            self._resume_active(repository, context)
            while True:
                run = repository.get_run(identifier)
                stop_reason = self._stop_reason(run)
                if stop_reason is not None:
                    repository.transition_run(
                        identifier, RunState.SEARCHING, RunState.FINALIZING, stop_reason
                    )
                    break
                card = self._next_card(repository, identifier)
                if card is None:
                    repository.transition_run(
                        identifier,
                        RunState.SEARCHING,
                        RunState.FINALIZING,
                        "eligible_method_queue_exhausted",
                    )
                    break
                try:
                    prepared = self._prepare_candidate(repository, context, card)
                except ProductionPreparationRejected:
                    continue
                self._run_candidate(repository, context, prepared.proposal.experiment_id)
            return self._finalize(repository, context)
        except BaseException:
            raise
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=10)
            try:
                repository.close_process_session(
                    session_id,
                    exit_reason=(
                        "production_complete"
                        if RunState(repository.get_run(identifier)["state"]) == RunState.COMPLETE
                        else "production_interrupted"
                    ),
                    monotonic_seconds=time.monotonic() - session_started,
                )
                closed = True
            except RepositoryError:
                if not closed:
                    pass

    def _new_deadline(self, external_deadline_epoch_ms: int | None) -> int:
        internal = deadline_epoch_ms(self.budget.wall_clock_seconds, self.now())
        if external_deadline_epoch_ms is None:
            return internal
        supplied = int(external_deadline_epoch_ms)
        now_ms = int(self.now() * 1000)
        if supplied <= now_ms:
            raise RuntimeError("external R3 deadline has already expired")
        if supplied > internal:
            raise RuntimeError("external R3 deadline exceeds the configured wall ceiling")
        return supplied

    def status(self, run_id: str) -> dict[str, Any]:
        if not SAFE_RUN_ID.fullmatch(run_id):
            raise RuntimeError("production run ID must be one safe path component")
        run_dir = self.config.runs_dir / run_id
        database = Database(run_dir / "state.sqlite3")
        if not database.path.is_file():
            raise RuntimeError(f"unknown production run: {run_id}")
        with database.connect() as connection:
            run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise RuntimeError(f"unknown production run: {run_id}")
            experiments = connection.execute(
                "SELECT experiment_id,iteration_number,method_card_id,state,commit_sha,terminal_reason "
                "FROM experiments WHERE run_id=? ORDER BY iteration_number",
                (run_id,),
            ).fetchall()
            repairs = connection.execute(
                "SELECT repair.experiment_id,repair.repair_number,repair.phase,repair.failure_status,"
                "repair.completed_at FROM experiment_repairs repair JOIN experiments experiment "
                "ON experiment.experiment_id=repair.experiment_id WHERE experiment.run_id=? "
                "ORDER BY repair.experiment_id,repair.repair_number",
                (run_id,),
            ).fetchall()
            sessions = connection.execute(
                "SELECT session_id,pid,host,started_at,ended_at,last_heartbeat,exit_reason "
                "FROM process_sessions WHERE run_id=? ORDER BY started_at",
                (run_id,),
            ).fetchall()
            baseline = connection.execute(
                "SELECT primary_units,gauc,ndcg5,evidence_json,created_at FROM baseline_gates "
                "WHERE run_id=?",
                (run_id,),
            ).fetchone()
            promotions = connection.execute(
                "SELECT previous_experiment_id,experiment_id,primary_units,evidence_json,created_at "
                "FROM search_promotions WHERE run_id=? ORDER BY promotion_id",
                (run_id,),
            ).fetchall()
            convergence = connection.execute(
                "SELECT experiment_id,outcome,delta_units,created_at FROM convergence_transactions "
                "WHERE run_id=? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        return {
            "run_id": run_id,
            "execution_mode": "production",
            "state": run["state"],
            "stop_reason": run["stop_reason"],
            "deadline_epoch_ms": run["deadline_epoch_ms"],
            "hypothesis_count": run["hypothesis_count"],
            "official_evaluation_count": run["official_evaluation_count"],
            "non_improvement_streak": run["non_improvement_streak"],
            "search_champion_experiment_id": run["search_champion_experiment_id"],
            "experiments": [dict(row) for row in experiments],
            "repairs": [dict(row) for row in repairs],
            "sessions": [dict(row) for row in sessions],
            "baseline_gate": None if baseline is None else dict(baseline),
            "search_promotions": [dict(row) for row in promotions],
            "convergence_transactions": [dict(row) for row in convergence],
            "scientific_execution_enabled": self.config.scientific_execution_enabled,
            "confirmation_enabled": False,
            "final_submission_enabled": False,
        }

    def compact_status(self, run_id: str) -> dict[str, Any]:
        """Return a small, read-only snapshot intended for hourly monitoring."""

        status = self.status(run_id)
        experiments = list(status["experiments"])
        active = [
            item
            for item in experiments
            if item["state"]
            not in {
                ExperimentState.PROMOTED,
                ExperimentState.REJECTED,
                ExperimentState.ABANDONED,
                ExperimentState.FAILED_FINAL,
            }
        ]
        sessions = list(status["sessions"])
        last_session = sessions[-1] if sessions else None
        return {
            "run_id": run_id,
            "state": status["state"],
            "stop_reason": status["stop_reason"],
            "hypotheses": status["hypothesis_count"],
            "official_evaluations": status["official_evaluation_count"],
            "non_improvement_streak": status["non_improvement_streak"],
            "champion": status["search_champion_experiment_id"],
            "active_experiment": active[-1] if active else None,
            "latest_session": last_session,
            "deadline_epoch_ms": int(status["deadline_epoch_ms"]),
            "monitoring_mode": "hourly_read_only",
            "test_prediction_created": False,
            "submission_created": False,
        }

    @staticmethod
    def _require_input(*, config_path: Path, name: str) -> None:
        if not config_path.is_file():
            raise RuntimeError(f"production {name} is missing: {config_path}")

    def _clean_root_commit(self) -> str:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=self.config.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0:
            raise RuntimeError("production project root is not a Git repository")
        if status.stdout.strip():
            raise RuntimeError("production search requires a clean committed project root")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.config.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode != 0:
            raise RuntimeError("cannot resolve production root commit")
        return commit.stdout.strip()

    @staticmethod
    def _heartbeat(
        repository: ExperimentRepository,
        session_id: str,
        started: float,
        stop: threading.Event,
    ) -> None:
        while not stop.wait(5):
            try:
                repository.heartbeat_process_session(session_id, time.monotonic() - started)
            except RepositoryError:
                continue

    def _stop_reason(self, run: dict[str, Any]) -> str | None:
        if run["stop_reason"] == "epsilon_plateau":
            return "epsilon_plateau"
        if int(run["hypothesis_count"]) >= self.budget.max_hypotheses:
            return "hypothesis_cap"
        if int(run["official_evaluation_count"]) >= self.budget.max_official_evaluations:
            return "official_evaluation_cap"
        if should_finalize(
            int(run["deadline_epoch_ms"]),
            self.budget.finalization_reserve_seconds,
            self.now(),
        ):
            return "finalization_reserve_reached"
        return None

    def _next_card(
        self, repository: ExperimentRepository, run_id: str
    ) -> ExperimentCard | None:
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT method_card_id,state FROM experiments WHERE run_id=?", (run_id,)
            ).fetchall()
            supported_rows = connection.execute(
                "SELECT DISTINCT experiment.method_card_id FROM experiments experiment "
                "JOIN transitions transition ON transition.experiment_id=experiment.experiment_id "
                "WHERE experiment.run_id=? AND transition.to_state=?",
                (run_id, ExperimentState.DIAGNOSED),
            ).fetchall()
        attempted = {str(row["method_card_id"]) for row in rows if row["method_card_id"]}
        supported = {
            str(row["method_card_id"])
            for row in supported_rows
            if row["method_card_id"]
        }
        unsupported = {
            card.card_id
            for card in self.search_policy.cards
            if card.stage == "search" and card.card_id not in self.config.method_cards
        }
        return self.search_policy.next_card(
            attempted | unsupported,
            self.search_policy.evidence_flags(supported),
        )

    def _proposal_context(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        card: ExperimentCard,
        *,
        exclude_experiment_id: str | None = None,
    ) -> dict[str, object]:
        run = repository.get_run(context.run_id)
        with repository.database.connect() as connection:
            evidence = [
                str(row["artifact_id"])
                for row in connection.execute(
                    "SELECT DISTINCT link.artifact_id FROM artifact_links link JOIN experiments experiment "
                    "ON experiment.experiment_id=link.experiment_id WHERE experiment.run_id=? AND "
                    "experiment.state IN ('PROMOTED','DIAGNOSED','REJECTED','FAILED_FINAL') AND "
                    "(? IS NULL OR experiment.experiment_id<>?) ORDER BY link.artifact_id",
                    (context.run_id, exclude_experiment_id, exclude_experiment_id),
                )
            ]
        binding = self.config.method_cards[card.card_id]
        proposal = self.search_policy.proposal_context(
            card,
            evidence_artifact_ids=evidence,
            incumbent_experiment_id=run["search_champion_experiment_id"],
            incumbent_primary_units=run["best_primary_units"],
            hypotheses_remaining=self.budget.max_hypotheses - int(run["hypothesis_count"]),
            seconds_remaining=seconds_remaining(int(run["deadline_epoch_ms"]), self.now()),
        )
        proposal["method_card_binding"] = {
            "experiment_config": str(binding.config_path),
            "feature_recipe": binding.feature_recipe,
        }
        research = self._research_context_summary(
            repository,
            context,
            card,
            exclude_experiment_id=exclude_experiment_id,
        )
        summarized_evidence = list(research.pop("_evidence_artifact_ids"))
        all_evidence = sorted(set(evidence).union(summarized_evidence))
        proposal["evidence_artifact_ids"] = all_evidence
        proposal["artifact_ids"] = all_evidence
        proposal.update(research)
        return proposal

    def _research_context_summary(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        card: ExperimentCard,
        *,
        exclude_experiment_id: str | None,
    ) -> dict[str, Any]:
        """Return bounded, label-free evidence summaries for researcher decisions."""

        with repository.database.connect() as connection:
            experiment_rows = connection.execute(
                "SELECT experiment_id,method_card_id,state,terminal_reason,proposal_json "
                "FROM experiments WHERE run_id=? AND (? IS NULL OR experiment_id<>?) "
                "ORDER BY iteration_number DESC LIMIT 20",
                (context.run_id, exclude_experiment_id, exclude_experiment_id),
            ).fetchall()
            attempt_rows = connection.execute(
                "SELECT attempt.experiment_id,experiment.method_card_id,attempt.rung,attempt.status,"
                "attempt.error_type FROM attempts attempt JOIN experiments experiment ON "
                "experiment.experiment_id=attempt.experiment_id WHERE experiment.run_id=? AND "
                "attempt.status<>'success' AND (? IS NULL OR attempt.experiment_id<>?) "
                "ORDER BY attempt.started_at DESC LIMIT 30",
                (context.run_id, exclude_experiment_id, exclude_experiment_id),
            ).fetchall()
            repair_rows = connection.execute(
                "SELECT repair.experiment_id,experiment.method_card_id,repair.repair_number,repair.phase,"
                "repair.failure_status,repair.plan_json,repair.evidence_json FROM experiment_repairs repair "
                "JOIN experiments experiment ON experiment.experiment_id=repair.experiment_id "
                "WHERE experiment.run_id=? AND (? IS NULL OR repair.experiment_id<>?) "
                "ORDER BY repair.created_at DESC LIMIT 30",
                (context.run_id, exclude_experiment_id, exclude_experiment_id),
            ).fetchall()
            resource = connection.execute(
                "SELECT COALESCE(SUM(wall_seconds),0) AS wall_seconds,"
                "COALESCE(MAX(peak_rss_bytes),0) AS peak_rss_bytes,"
                "COALESCE(SUM(gpu_seconds),0) AS gpu_seconds FROM resource_usage WHERE run_id=?",
                (context.run_id,),
            ).fetchone()
            baseline = connection.execute(
                "SELECT evidence_json FROM baseline_gates WHERE run_id=?", (context.run_id,)
            ).fetchone()

        prior_experiments: list[dict[str, Any]] = []
        segment_diagnostics: list[dict[str, Any]] = []
        prediction_correlations: list[dict[str, Any]] = []
        evidence_ids: set[str] = set(json.loads(baseline["evidence_json"])) if baseline else set()
        for row in reversed(experiment_rows):
            proposal = ExperimentProposal.model_validate_json(row["proposal_json"])
            rung_summaries = self._rung_artifact_summaries(repository, str(row["experiment_id"]))
            for rung in rung_summaries:
                evidence_ids.add(str(rung["artifact_id"]))
                diagnostics = dict(rung.get("diagnostics", {}))
                partitions = diagnostics.get("by_partition", {})
                diagnostic_parts = (
                    list(partitions.items())
                    if isinstance(partitions, dict)
                    else [(None, diagnostics)]
                )
                if not diagnostic_parts:
                    diagnostic_parts = [(None, diagnostics)]
                for partition, detail in diagnostic_parts:
                    if not isinstance(detail, dict):
                        continue
                    segment = {
                        key: detail[key]
                        for key in (
                            "segment_primary_deltas",
                            "segment_wins",
                            "segment_regressions",
                        )
                        if key in detail
                    }
                    if segment:
                        segment_diagnostics.append(
                            {
                                "experiment_id": row["experiment_id"],
                                "rung": rung["rung"],
                                "partition": partition,
                                "artifact_id": rung["artifact_id"],
                                **segment,
                            }
                        )
                    if "prediction_correlation" in detail:
                        prediction_correlations.append(
                            {
                                "experiment_id": row["experiment_id"],
                                "rung": rung["rung"],
                                "partition": partition,
                                "artifact_id": rung["artifact_id"],
                                "prediction_correlation": detail["prediction_correlation"],
                            }
                        )
            prior_experiments.append(
                {
                    "experiment_id": row["experiment_id"],
                    "method_card_id": row["method_card_id"],
                    "state": row["state"],
                    "terminal_reason": row["terminal_reason"],
                    "hypothesis": proposal.hypothesis,
                    "primary_change": proposal.primary_change,
                    "falsifier": proposal.falsifier,
                    "rungs": rung_summaries,
                }
            )

        failures: list[dict[str, Any]] = [
            {
                "experiment_id": row["experiment_id"],
                "method_card_id": row["method_card_id"],
                "phase": row["rung"],
                "status": row["status"],
                "error_type": row["error_type"],
            }
            for row in attempt_rows
        ]
        for row in repair_rows:
            plan = json.loads(row["plan_json"])
            repair_evidence = json.loads(row["evidence_json"]) if row["evidence_json"] else []
            evidence_ids.update(str(item) for item in repair_evidence)
            failures.append(
                {
                    "experiment_id": row["experiment_id"],
                    "method_card_id": row["method_card_id"],
                    "phase": row["phase"],
                    "status": row["failure_status"],
                    "repair_number": row["repair_number"],
                    "repair_action": plan.get("action"),
                    "evidence_artifact_ids": repair_evidence,
                }
            )

        try:
            binding_path = str(
                self.config.method_cards[card.card_id].config_path.relative_to(
                    self.config.project_root
                )
            )
        except ValueError:
            binding_path = str(self.config.method_cards[card.card_id].config_path)
        allowed_files = [binding_path, *CARD_CODE_PATHS.get(card.card_id, ())]
        timeout = self.budget.default_attempt_timeout_seconds
        return {
            "prior_experiments": prior_experiments,
            "segment_diagnostics": segment_diagnostics[-30:],
            "prediction_correlations": prediction_correlations[-30:],
            "failure_and_repair_history": failures[:30],
            "allowed_files": allowed_files,
            "resource_estimate": {
                "cheap_seconds": timeout,
                "full_seconds": timeout * 3,
                "official_valid_seconds": timeout,
                "maximum_candidate_seconds": timeout * 5,
                "observed_run_wall_seconds": float(resource["wall_seconds"]),
                "observed_peak_rss_bytes": int(resource["peak_rss_bytes"]),
                "observed_gpu_seconds": float(resource["gpu_seconds"]),
            },
            "falsification_criteria": {
                "method_card_falsifier": METHOD_CARD_REFERENCES[card.card_id]["falsifier"],
                "cheap_min_primary_delta": 0.001,
                "full_min_positive_temporal_folds": 2,
                "component_regression_limit": 0.002,
                "convergence_epsilon": self.budget.convergence_epsilon_units / 1_000_000_000,
            },
            "_evidence_artifact_ids": sorted(evidence_ids),
        }

    def _rung_artifact_summaries(
        self,
        repository: ExperimentRepository,
        experiment_id: str,
    ) -> list[dict[str, Any]]:
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT artifact.artifact_id,artifact.kind,artifact.sha256,link.artifact_path "
                "FROM artifact_links link JOIN artifacts artifact ON artifact.artifact_id=link.artifact_id "
                "WHERE link.experiment_id=? AND artifact.kind LIKE 'production_%_result' "
                "ORDER BY artifact.created_at",
                (experiment_id,),
            ).fetchall()
        summaries: list[dict[str, Any]] = []
        for row in rows:
            path = Path(str(row["artifact_path"]))
            if not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            if sha256_file(path) != row["sha256"]:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                observations = list(payload["observations"])
                deltas = [
                    {
                        "primary": float(item["candidate"]["primary"])
                        - float(item["reference"]["primary"]),
                        "GAUC": float(item["candidate"]["GAUC"])
                        - float(item["reference"]["GAUC"]),
                        "nDCG@5": float(item["candidate"]["nDCG@5"])
                        - float(item["reference"]["nDCG@5"]),
                        "fold": item["candidate"].get("fold"),
                    }
                    for item in observations[:10]
                ]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            summaries.append(
                {
                    "artifact_id": row["artifact_id"],
                    "rung": payload.get("rung", str(row["kind"])[11:-7]),
                    "metric_component_deltas": deltas,
                    "diagnostics": self._safe_diagnostics(payload.get("diagnostics", {})),
                }
            )
        return summaries

    @staticmethod
    def _safe_diagnostics(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        allowed = {
            "candidate",
            "reference",
            "delta",
            "candidate_primary_ci",
            "primary_delta_ci",
            "prediction_correlation",
            "segment_primary_deltas",
            "segment_wins",
            "segment_regressions",
        }

        def bounded(item: Any, depth: int = 0) -> Any:
            if depth > 2:
                return None
            if item is None or isinstance(item, (bool, int, float, str)):
                return item
            if isinstance(item, list):
                return [bounded(child, depth + 1) for child in item[:100]]
            if isinstance(item, dict):
                return {
                    str(key)[:120]: bounded(child, depth + 1)
                    for key, child in list(sorted(item.items(), key=lambda pair: str(pair[0])))[:100]
                }
            return str(item)[:200]

        direct = {key: bounded(value[key]) for key in sorted(allowed.intersection(value))}
        if direct:
            return direct
        partitions: dict[str, Any] = {}
        for key, detail in list(sorted(value.items(), key=lambda pair: str(pair[0])))[:10]:
            if not isinstance(detail, dict):
                continue
            safe = {name: bounded(detail[name]) for name in sorted(allowed.intersection(detail))}
            if safe:
                partitions[str(key)[:120]] = safe
        return {"by_partition": partitions} if partitions else {}

    def _effective_config(
        self,
        prepared: CandidatePreparation,
        binding: MethodCardBinding,
    ) -> tuple[Path, str]:
        path = (prepared.effective_config_path or binding.config_path).resolve()
        self._require_input(config_path=path, name="effective experiment config")
        binding_path = binding.config_path.resolve()
        if path != binding_path:
            allowed_roots = [self.config.runs_dir.resolve()]
            if prepared.workspace_path is not None:
                allowed_roots.append(prepared.workspace_path.resolve())
            if not any(self._is_relative_to(path, root) for root in allowed_roots):
                raise RuntimeError(
                    "effective candidate config is outside the candidate worktree and durable run evidence"
                )
        digest = sha256_file(path)
        if prepared.effective_config_sha256 not in {None, digest}:
            raise RuntimeError("effective candidate config hash does not match its contents")
        return path, digest

    @staticmethod
    def _snapshot_effective_config(
        context: ProductionContext,
        experiment_id: str,
        source: Path,
        expected_sha256: str,
    ) -> Path:
        suffix = source.suffix if source.suffix else ".yaml"
        destination = (
            context.run_dir / "evidence" / experiment_id / f"effective-config{suffix}"
        ).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination:
            temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
            with source.open("rb") as reader, temporary.open("wb") as writer:
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, destination)
        if sha256_file(destination) != expected_sha256:
            raise RuntimeError("durable effective config snapshot failed hash verification")
        return destination

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    def _write_preparation_evidence(
        self,
        context: ProductionContext,
        card: ExperimentCard,
        binding: MethodCardBinding,
        proposal_context: dict[str, object],
        effective_config: Path,
        effective_config_sha256: str,
    ) -> tuple[ArtifactRef, ArtifactRef]:
        directory = context.run_dir / "evidence" / str(proposal_context.get("resume_experiment_id") or "")
        if not directory.name:
            raise RuntimeError("preparation evidence needs an experiment identity")
        decision_path = directory / "method-card-decision.json"
        atomic_write_json(
            decision_path,
            {
                "provider": "fixed_config"
                if self._fixed_mode()
                else "live_researcher",
                "method_card_version": METHOD_CARD_VERSION,
                "citation_id": f"method-card:{METHOD_CARD_VERSION}:{card.card_id}",
                "card_id": card.card_id,
                "primary_change": METHOD_CARD_REFERENCES[card.card_id]["primary_change"],
                "falsifier": METHOD_CARD_REFERENCES[card.card_id]["falsifier"],
                "published_method_sources": proposal_context["method_sources"],
                "binding_config_path": str(binding.config_path),
                "effective_config_path": str(effective_config),
                "effective_config_sha256": effective_config_sha256,
                "feature_recipe": binding.feature_recipe,
                "allowed_files": proposal_context["allowed_files"],
                "patch_authored": not self._fixed_mode(),
            },
        )
        stored_context = dict(proposal_context)
        stored_context.pop("resume_experiment_id", None)
        stored_context.pop("durable_proposal", None)
        proposal_context_path = directory / "proposal-context.json"
        atomic_write_json(proposal_context_path, stored_context)
        return (
            artifact_ref(decision_path, "method_card_decision"),
            artifact_ref(proposal_context_path, "proposal_context"),
        )

    def _fixed_mode(self) -> bool:
        return self.config.llm.get("mode") == "fixed" or isinstance(
            self.provider, ProductionFixedProvider
        )

    def _stored_proposal_context(
        self,
        repository: ExperimentRepository,
        experiment_id: str,
    ) -> dict[str, object] | None:
        with repository.database.connect() as connection:
            row = connection.execute(
                "SELECT link.artifact_path,artifact.sha256 FROM artifact_links link JOIN artifacts artifact "
                "ON artifact.artifact_id=link.artifact_id WHERE link.experiment_id=? AND "
                "artifact.kind='proposal_context' ORDER BY artifact.created_at DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        path = Path(str(row["artifact_path"]))
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError("durable proposal context is missing or corrupt")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("durable proposal context is malformed")
        return value

    def _prepare_candidate(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        card: ExperimentCard,
    ) -> CandidatePreparation:
        assert self.hooks is not None
        binding = self.config.method_cards[card.card_id]
        self._require_input(config_path=binding.config_path, name=f"{card.card_id} config")
        run = repository.get_run(context.run_id)
        parent_id = run["search_champion_experiment_id"]
        parent_commit = context.root_commit
        if parent_id and parent_id != "baseline":
            parent = repository.get_experiment(str(parent_id))
            parent_commit = str(parent["commit_sha"] or context.root_commit)
        proposal_context = self._proposal_context(repository, context, card)
        prepared = self.hooks.prepare_candidate(
            context,
            card,
            binding,
            proposal_context,
            parent_commit,
        )
        unexpected_files = sorted(
            set(prepared.proposal.files_to_change).difference(proposal_context["allowed_files"])
        )
        if unexpected_files:
            raise RuntimeError(
                "candidate proposal requested files outside the card allowlist: "
                + ", ".join(unexpected_files)
            )
        if prepared.proposal.parent_id not in {None, parent_id}:
            raise RuntimeError("candidate proposal changed the coordinator-selected parent")
        if prepared.proposal.parent_id == "baseline":
            prepared = CandidatePreparation(
                proposal=prepared.proposal.model_copy(update={"parent_id": None}),
                commit_sha=prepared.commit_sha,
                workspace_path=prepared.workspace_path,
                branch_name=prepared.branch_name,
                artifacts=prepared.artifacts,
                effective_config_path=prepared.effective_config_path,
                effective_config_sha256=prepared.effective_config_sha256,
            )
        effective_config, effective_config_sha256 = self._effective_config(prepared, binding)
        effective_config = self._snapshot_effective_config(
            context,
            prepared.proposal.experiment_id,
            effective_config,
            effective_config_sha256,
        )
        repository.create_experiment(
            context.run_id,
            prepared.proposal,
            parent_commit,
            max_hypotheses=self.budget.max_hypotheses,
            workspace_path=str(prepared.workspace_path) if prepared.workspace_path else None,
            branch_name=prepared.branch_name,
            commit_sha=prepared.commit_sha,
            config_sha256=effective_config_sha256,
            method_card_id=card.card_id,
            experiment_kind="production_search",
        )
        proposal_context["resume_experiment_id"] = prepared.proposal.experiment_id
        decision_ref, proposal_context_ref = self._write_preparation_evidence(
            context,
            card,
            binding,
            proposal_context,
            effective_config,
            effective_config_sha256,
        )
        config_ref = artifact_ref(effective_config, "experiment_config")
        for ref in (*prepared.artifacts, decision_ref, proposal_context_ref, config_ref):
            repository.register_artifact(ref, experiment_id=prepared.proposal.experiment_id)
        transitions = (
            (ExperimentState.PROPOSED, ExperimentState.WORKTREE_READY, "worktree-ready"),
            (ExperimentState.WORKTREE_READY, ExperimentState.PATCHED, "patched"),
            (ExperimentState.PATCHED, ExperimentState.STATIC_VALID, "static-valid"),
            (ExperimentState.STATIC_VALID, ExperimentState.FIXTURE_VALID, "fixture-valid"),
        )
        for current, target, suffix in transitions:
            repository.transition_experiment(
                prepared.proposal.experiment_id,
                current,
                target,
                payload={
                    "method_card_id": card.card_id,
                    "evidence_artifact_ids": [item.artifact_id for item in prepared.artifacts],
                },
                idempotency_key=f"{prepared.proposal.experiment_id}:{suffix}",
            )
        return prepared

    def _resume_active(self, repository: ExperimentRepository, context: ProductionContext) -> None:
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT experiment_id,state,workspace_path,commit_sha FROM experiments WHERE run_id=? "
                "AND state NOT IN ('PROMOTED','REJECTED','ABANDONED') "
                "ORDER BY iteration_number",
                (context.run_id,),
            ).fetchall()
        for row in rows:
            state = ExperimentState(row["state"])
            if state in {
                ExperimentState.PROPOSED,
                ExperimentState.WORKTREE_READY,
                ExperimentState.PATCHED,
                ExperimentState.STATIC_VALID,
            }:
                try:
                    self._resume_preparation(repository, context, str(row["experiment_id"]))
                except ProductionPreparationRejected:
                    continue
            self._run_candidate(repository, context, str(row["experiment_id"]))

    def _resume_preparation(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        experiment_id: str,
    ) -> None:
        """Idempotently rebuild and finish safety gates for the same method card."""

        assert self.hooks is not None
        experiment = repository.get_experiment(experiment_id)
        card = next(
            item for item in self.search_policy.cards if item.card_id == experiment["method_card_id"]
        )
        binding = self.config.method_cards[card.card_id]
        parent_commit = str(experiment["parent_commit"] or context.root_commit)
        proposal_context = self._stored_proposal_context(repository, experiment_id)
        if proposal_context is None:
            proposal_context = self._proposal_context(
                repository,
                context,
                card,
                exclude_experiment_id=experiment_id,
            )
        proposal_context = {
            **proposal_context,
            "resume_experiment_id": experiment_id,
            "durable_proposal": json.loads(experiment["proposal_json"]),
        }
        prepared = self.hooks.prepare_candidate(
            context,
            card,
            binding,
            proposal_context,
            parent_commit,
        )
        if prepared.proposal.experiment_id != experiment_id:
            raise RuntimeError("preparation resume returned a different experiment ID")
        if prepared.proposal.model_dump(mode="json") != json.loads(experiment["proposal_json"]):
            raise RuntimeError("preparation resume changed the durable experiment proposal")
        allowed_files = proposal_context["allowed_files"]
        unexpected_files = sorted(set(prepared.proposal.files_to_change).difference(allowed_files))
        if unexpected_files:
            raise RuntimeError(
                "resumed candidate requested files outside the card allowlist: "
                + ", ".join(unexpected_files)
            )
        effective_config, effective_config_sha256 = self._effective_config(prepared, binding)
        effective_config = self._snapshot_effective_config(
            context,
            experiment_id,
            effective_config,
            effective_config_sha256,
        )
        if prepared.workspace_path and prepared.branch_name:
            repository.record_experiment_workspace(
                experiment_id,
                workspace_path=str(prepared.workspace_path),
                branch_name=prepared.branch_name,
                commit_sha=prepared.commit_sha,
                config_sha256=effective_config_sha256,
            )
        decision_ref, proposal_context_ref = self._write_preparation_evidence(
            context,
            card,
            binding,
            proposal_context,
            effective_config,
            effective_config_sha256,
        )
        config_ref = artifact_ref(effective_config, "experiment_config")
        for ref in (*prepared.artifacts, decision_ref, proposal_context_ref, config_ref):
            repository.register_artifact(ref, experiment_id=experiment_id)
        chain = {
            ExperimentState.PROPOSED: (
                ExperimentState.WORKTREE_READY,
                "worktree-ready",
            ),
            ExperimentState.WORKTREE_READY: (ExperimentState.PATCHED, "patched"),
            ExperimentState.PATCHED: (ExperimentState.STATIC_VALID, "static-valid"),
            ExperimentState.STATIC_VALID: (ExperimentState.FIXTURE_VALID, "fixture-valid"),
        }
        while True:
            current = ExperimentState(repository.get_experiment(experiment_id)["state"])
            if current == ExperimentState.FIXTURE_VALID:
                return
            target, suffix = chain[current]
            repository.transition_experiment(
                experiment_id,
                current,
                target,
                payload={
                    "resumed": True,
                    "method_card_id": card.card_id,
                    "evidence_artifact_ids": [item.artifact_id for item in prepared.artifacts],
                },
                idempotency_key=f"{experiment_id}:{suffix}",
            )

    def _run_candidate(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        experiment_id: str,
    ) -> None:
        while True:
            experiment = repository.get_experiment(experiment_id)
            state = ExperimentState(experiment["state"])
            if state == ExperimentState.FIXTURE_VALID:
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.CHEAP_RUNNING,
                    idempotency_key=f"{experiment_id}:cheap-running",
                )
                continue
            if state == ExperimentState.CHEAP_RUNNING:
                if not self._complete_rung(repository, context, experiment, "cheap"):
                    return
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.CHEAP_COMPLETE,
                    idempotency_key=f"{experiment_id}:cheap-complete",
                )
                continue
            if state == ExperimentState.CHEAP_COMPLETE:
                result = self._stored_rung(repository, experiment_id, "cheap")
                if not self._cheap_gate(experiment, result):
                    repository.reject_candidate(
                        run_id=context.run_id,
                        experiment_id=experiment_id,
                        expected_state=state,
                        reason="cheap evidence gate rejected the candidate",
                        patience=self.budget.convergence_patience,
                        idempotency_key=f"{experiment_id}:cheap-rejected",
                    )
                    self._cleanup_worktree(repository, context, experiment_id)
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
                if not self._complete_rung(repository, context, experiment, "full"):
                    return
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.FULL_COMPLETE,
                    idempotency_key=f"{experiment_id}:full-complete",
                )
                continue
            if state == ExperimentState.FULL_COMPLETE:
                result = self._stored_rung(repository, experiment_id, "full")
                self._diagnose(repository, context, experiment_id, result)
                continue
            if state == ExperimentState.DIAGNOSED:
                result = self._stored_rung(repository, experiment_id, "full")
                if not self._full_gate(experiment, result):
                    repository.reject_candidate(
                        run_id=context.run_id,
                        experiment_id=experiment_id,
                        expected_state=state,
                        reason="full temporal evidence gate rejected the candidate",
                        patience=self.budget.convergence_patience,
                        idempotency_key=f"{experiment_id}:full-rejected",
                    )
                    self._cleanup_worktree(repository, context, experiment_id)
                    return
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.OFFICIAL_VALID_RUNNING,
                    idempotency_key=f"{experiment_id}:official-valid-running",
                )
                continue
            if state == ExperimentState.OFFICIAL_VALID_RUNNING:
                if not self._complete_rung(repository, context, experiment, "official_valid"):
                    return
                repository.transition_experiment(
                    experiment_id,
                    state,
                    ExperimentState.OFFICIAL_VALID_COMPLETE,
                    idempotency_key=f"{experiment_id}:official-valid-complete",
                )
                continue
            if state == ExperimentState.OFFICIAL_VALID_COMPLETE:
                result = self._stored_rung(repository, experiment_id, "official_valid")
                observation = result.observations[0]
                rule = ExperimentProposal.model_validate_json(
                    experiment["proposal_json"]
                ).promotion_rule
                evidence = self._experiment_evidence(repository, experiment_id)
                if (
                    observation.primary_delta <= 0
                    or observation.gauc_delta < -rule.max_gauc_regression
                    or observation.ndcg_delta < -rule.max_ndcg5_regression
                ):
                    repository.reject_non_improving(
                        run_id=context.run_id,
                        experiment_id=experiment_id,
                        expected_state=state,
                        reason="official validation did not safely improve the incumbent",
                        patience=self.budget.convergence_patience,
                        idempotency_key=f"{experiment_id}:official-valid-rejected",
                    )
                else:
                    repository.promote_search_candidate(
                        run_id=context.run_id,
                        experiment_id=experiment_id,
                        primary=observation.candidate.primary,
                        evidence_artifact_ids=evidence,
                        epsilon=self.budget.convergence_epsilon_units / 1_000_000_000,
                        patience=self.budget.convergence_patience,
                        idempotency_key=f"{experiment_id}:search-promotion",
                    )
                self._cleanup_worktree(repository, context, experiment_id)
                return
            if state in {
                ExperimentState.REJECTED,
                ExperimentState.PROMOTED,
                ExperimentState.ABANDONED,
                ExperimentState.FAILED_FINAL,
            }:
                if state == ExperimentState.FAILED_FINAL:
                    self._count_failed_transaction(repository, context, experiment_id)
                self._cleanup_worktree(repository, context, experiment_id)
                return
            if state in {ExperimentState.FAILED_REPAIRABLE, ExperimentState.REPAIRING}:
                if not self._resume_repair(repository, context, experiment):
                    return
                continue
            raise RuntimeError(f"cannot resume production candidate {experiment_id} from {state}")

    def _complete_rung(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        experiment: dict[str, Any],
        rung: str,
    ) -> bool:
        experiment_id = str(experiment["experiment_id"])
        if self._stored_rung(repository, experiment_id, rung, required=False) is not None:
            return True
        assert self.hooks is not None
        card = next(
            card for card in self.search_policy.cards if card.card_id == experiment["method_card_id"]
        )
        try:
            result = self.hooks.run_rung(
                RungRequest(
                    context=context,
                    experiment=experiment,
                    method_card=card,
                    binding=self.config.method_cards[card.card_id],
                    rung=rung,
                )
            )
        except ProductionRungFailure as failure:
            recovered = self._handle_failure(repository, context, experiment, rung, failure)
            if not recovered:
                return False
            return self._complete_rung(
                repository,
                context,
                repository.get_experiment(experiment_id),
                rung,
            )
        if not result.observations:
            raise RuntimeError(f"production {rung} returned no comparison observations")
        if rung in {"cheap", "official_valid"} and len(result.observations) != 1:
            raise RuntimeError(f"production {rung} requires exactly one observation")
        for ref in result.artifacts:
            repository.register_artifact(ref, experiment_id=experiment_id)
        for observation in result.observations:
            repository.record_metrics(
                experiment_id,
                observation.candidate,
                max_official_evaluations=self.budget.max_official_evaluations,
            )
        path = context.run_dir / "evidence" / experiment_id / f"{rung}-result.json"
        atomic_write_json(
            path,
            {
                "rung": rung,
                "observations": [
                    {
                        "candidate": item.candidate.model_dump(mode="json", by_alias=True),
                        "reference": item.reference.model_dump(mode="json", by_alias=True),
                    }
                    for item in result.observations
                ],
                "artifact_ids": [item.artifact_id for item in result.artifacts],
                "diagnostics": self._safe_diagnostics(result.diagnostics),
            },
        )
        ref = artifact_ref(path, f"production_{rung}_result")
        repository.register_artifact(ref, experiment_id=experiment_id)
        return True

    def _stored_rung(
        self,
        repository: ExperimentRepository,
        experiment_id: str,
        rung: str,
        *,
        required: bool = True,
    ) -> ProductionRungResult | None:
        with repository.database.connect() as connection:
            row = connection.execute(
                "SELECT link.artifact_path,artifact.sha256 FROM artifact_links link JOIN artifacts artifact "
                "ON artifact.artifact_id=link.artifact_id WHERE link.experiment_id=? AND "
                "artifact.kind=? ORDER BY artifact.created_at DESC LIMIT 1",
                (experiment_id, f"production_{rung}_result"),
            ).fetchone()
        if row is None:
            if required:
                raise RuntimeError(f"missing durable {rung} result for {experiment_id}")
            return None
        path = Path(row["artifact_path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"durable {rung} result is missing or corrupt")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ProductionRungResult(
            observations=tuple(
                ComparisonObservation(
                    candidate=Metrics.model_validate(item["candidate"]),
                    reference=Metrics.model_validate(item["reference"]),
                )
                for item in payload["observations"]
            ),
            diagnostics=self._safe_diagnostics(payload.get("diagnostics", {})),
        )

    @staticmethod
    def _cheap_gate(experiment: dict[str, Any], result: ProductionRungResult) -> bool:
        rule = ExperimentProposal.model_validate_json(experiment["proposal_json"]).promotion_rule
        observation = result.observations[0]
        return (
            observation.primary_delta >= rule.min_primary_delta
            and observation.gauc_delta >= -rule.max_gauc_regression
            and observation.ndcg_delta >= -rule.max_ndcg5_regression
        )

    @staticmethod
    def _full_gate(experiment: dict[str, Any], result: ProductionRungResult) -> bool:
        rule = ExperimentProposal.model_validate_json(experiment["proposal_json"]).promotion_rule
        observations = result.observations
        positive = sum(item.primary_delta > 0 for item in observations)
        mean_primary = sum(item.primary_delta for item in observations) / len(observations)
        mean_gauc = sum(item.gauc_delta for item in observations) / len(observations)
        mean_ndcg = sum(item.ndcg_delta for item in observations) / len(observations)
        return (
            positive >= rule.min_positive_shadow_folds
            and mean_primary > 0
            and mean_gauc >= -rule.max_gauc_regression
            and mean_ndcg >= -rule.max_ndcg5_regression
        )

    def _diagnose(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        experiment_id: str,
        result: ProductionRungResult,
    ) -> None:
        evidence = self._experiment_evidence(repository, experiment_id)
        experiment = repository.get_experiment(experiment_id)
        card = next(
            item for item in self.search_policy.cards if item.card_id == experiment["method_card_id"]
        )
        research = self._research_context_summary(
            repository,
            context,
            card,
            exclude_experiment_id=None,
        )
        current_summary = next(
            (
                item
                for item in research["prior_experiments"]
                if item["experiment_id"] == experiment_id
            ),
            None,
        )
        diagnosis_context = {
            "artifact_ids": evidence,
            "method_card_id": experiment["method_card_id"],
            "fold_metric_deltas": [item.primary_delta for item in result.observations],
            "metric_component_summary": None if current_summary is None else current_summary["rungs"],
            "segment_diagnostics": [
                item
                for item in research["segment_diagnostics"]
                if item["experiment_id"] == experiment_id
            ],
            "prediction_correlations": [
                item
                for item in research["prediction_correlations"]
                if item["experiment_id"] == experiment_id
            ],
            "failure_and_repair_history": [
                item
                for item in research["failure_and_repair_history"]
                if item["experiment_id"] == experiment_id
            ],
            "resource_summary": research["resource_estimate"],
            "falsification_criteria": research["falsification_criteria"],
            "evidence_binding_required": True,
            "confirmation_deferred": True,
            "test_submission_deferred": True,
        }
        decision = DiagnosisService(self.provider).diagnose(experiment_id, diagnosis_context)
        directory = context.run_dir / "evidence" / experiment_id
        request_path = directory / "diagnosis-request.json"
        response_path = directory / "diagnosis.json"
        atomic_write_json(request_path, diagnosis_context)
        atomic_write_json(
            response_path,
            {
                "decision": decision.parsed.model_dump(mode="json", by_alias=True),
                "provider": decision.response.provider,
                "model": decision.response.model,
                "request_id": decision.response.request_id,
            },
        )
        request_ref = artifact_ref(request_path, "llm_request")
        response_ref = artifact_ref(response_path, "diagnosis")
        repository.register_artifact(request_ref, experiment_id=experiment_id)
        repository.register_artifact(response_ref, experiment_id=experiment_id)
        repository.record_llm_call(
            call_id=f"{experiment_id}:diagnosis",
            run_id=context.run_id,
            experiment_id=experiment_id,
            role="diagnosis",
            provider=decision.response.provider,
            model=decision.response.model,
            request_artifact_id=request_ref.artifact_id,
            response_artifact_id=response_ref.artifact_id,
            schema_valid=decision.response.schema_valid,
            input_tokens=decision.response.input_tokens,
            output_tokens=decision.response.output_tokens,
            wall_seconds=decision.response.wall_seconds,
            request_id=decision.response.request_id,
        )
        remember_reflection(repository, context.run_id, decision.parsed)
        repository.transition_experiment(
            experiment_id,
            ExperimentState.FULL_COMPLETE,
            ExperimentState.DIAGNOSED,
            payload={"diagnosis_artifact_id": response_ref.artifact_id},
            idempotency_key=f"{experiment_id}:diagnosed",
        )

    def _handle_failure(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        experiment: dict[str, Any],
        phase: str,
        failure: ProductionRungFailure,
    ) -> bool:
        experiment_id = str(experiment["experiment_id"])
        state = ExperimentState(experiment["state"])
        used = repository.experiment_repairs_used(experiment_id)
        plan = plan_repair(
            failure.status,
            used,
            phase=phase,
            maximum=self.budget.max_repairs_per_experiment,
        )
        if plan.action == RepairAction.RESUME:
            raise failure
        repository.transition_experiment(
            experiment_id,
            state,
            ExperimentState.FAILED_REPAIRABLE,
            payload={
                "phase": phase,
                "status": failure.status,
                "reason": str(failure)[-1000:],
            },
            idempotency_key=f"{experiment_id}:{phase}:failure:{used}",
        )
        if not plan.repair:
            repository.transition_experiment(
                experiment_id,
                ExperimentState.FAILED_REPAIRABLE,
                ExperimentState.FAILED_FINAL,
                payload={"reason": plan.reason},
                idempotency_key=f"{experiment_id}:{phase}:failed-final:{used}",
            )
            self._count_failed_transaction(repository, context, experiment_id)
            self._cleanup_worktree(repository, context, experiment_id)
            return False
        reservation = repository.reserve_experiment_repair(
            experiment_id=experiment_id,
            phase=phase,
            failure_status=failure.status,
            plan={
                "action": plan.action,
                "reason": plan.reason,
                "overrides": plan.overrides,
            },
            maximum=self.budget.max_repairs_per_experiment,
        )
        repository.transition_experiment(
            experiment_id,
            ExperimentState.FAILED_REPAIRABLE,
            ExperimentState.REPAIRING,
            payload={"repair_id": reservation["repair_id"], "phase": phase},
            idempotency_key=f"{reservation['repair_id']}:repairing",
        )
        return self._execute_repair(repository, context, experiment, plan, reservation)

    def _resume_repair(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        experiment: dict[str, Any],
    ) -> bool:
        with repository.database.connect() as connection:
            transition = connection.execute(
                "SELECT payload_json FROM transitions WHERE experiment_id=? ORDER BY transition_id DESC "
                "LIMIT 1",
                (experiment["experiment_id"],),
            ).fetchone()
            repair = connection.execute(
                "SELECT * FROM experiment_repairs WHERE experiment_id=? ORDER BY repair_number DESC LIMIT 1",
                (experiment["experiment_id"],),
            ).fetchone()
        payload = json.loads(transition["payload_json"]) if transition else {}
        phase = str(payload.get("phase") or (repair["phase"] if repair else "cheap"))
        if repair is None:
            status = AttemptStatus(str(payload.get("status", AttemptStatus.CRASH)))
            used = repository.experiment_repairs_used(str(experiment["experiment_id"]))
            plan = plan_repair(
                status,
                used,
                phase=phase,
                maximum=self.budget.max_repairs_per_experiment,
            )
            if not plan.repair:
                repository.transition_experiment(
                    str(experiment["experiment_id"]),
                    ExperimentState.FAILED_REPAIRABLE,
                    ExperimentState.FAILED_FINAL,
                    payload={"reason": plan.reason},
                    idempotency_key=f"{experiment['experiment_id']}:{phase}:failed-final:{used}",
                )
                self._count_failed_transaction(
                    repository, context, str(experiment["experiment_id"])
                )
                return False
            reservation = repository.reserve_experiment_repair(
                experiment_id=str(experiment["experiment_id"]),
                phase=phase,
                failure_status=status,
                plan={
                    "action": plan.action,
                    "reason": plan.reason,
                    "overrides": plan.overrides,
                },
                maximum=self.budget.max_repairs_per_experiment,
            )
            repository.transition_experiment(
                str(experiment["experiment_id"]),
                ExperimentState.FAILED_REPAIRABLE,
                ExperimentState.REPAIRING,
                payload={"repair_id": reservation["repair_id"], "phase": phase},
                idempotency_key=f"{reservation['repair_id']}:repairing",
            )
            return self._execute_repair(repository, context, experiment, plan, reservation)
        plan_payload = json.loads(repair["plan_json"])
        plan = TypedRepairPlan(
            True,
            True,
            int(repair["repair_number"]),
            RepairAction(plan_payload["action"]),
            str(plan_payload["reason"]),
            dict(plan_payload.get("overrides", {})),
        )
        reservation = {
            "repair_id": repair["repair_id"],
            "repair_number": repair["repair_number"],
        }
        if ExperimentState(experiment["state"]) == ExperimentState.FAILED_REPAIRABLE:
            repository.transition_experiment(
                experiment["experiment_id"],
                ExperimentState.FAILED_REPAIRABLE,
                ExperimentState.REPAIRING,
                payload={"repair_id": repair["repair_id"], "phase": phase},
                idempotency_key=f"{repair['repair_id']}:repairing",
            )
        return self._execute_repair(repository, context, experiment, plan, reservation)

    def _execute_repair(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        experiment: dict[str, Any],
        plan: TypedRepairPlan,
        reservation: dict[str, Any],
    ) -> bool:
        assert self.hooks is not None
        phase = self._repair_phase(repository, str(reservation["repair_id"]))
        try:
            experiment_id = str(experiment["experiment_id"])
            evidence = self._recover_pending_repair_evidence(
                context,
                experiment_id,
                phase,
                int(reservation["repair_number"]),
            )
            if evidence is None:
                evidence = self.hooks.repair_candidate(
                    RepairRequest(
                        context=context,
                        experiment=repository.get_experiment(experiment_id),
                        phase=phase,
                        plan=plan,
                    )
                )
            for ref in evidence:
                repository.register_artifact(
                    ref, experiment_id=experiment_id
                )
            if self.lifecycle_checkpoint is not None:
                self.lifecycle_checkpoint("repair_evidence_registered", experiment_id)
            self._apply_repair_revision(
                repository,
                context,
                experiment_id,
                str(reservation["repair_id"]),
                int(reservation["repair_number"]),
                evidence,
            )
            repository.complete_experiment_repair(
                str(reservation["repair_id"]),
                evidence_artifact_ids=[item.artifact_id for item in evidence],
            )
        except Exception as error:
            repository.transition_experiment(
                str(experiment["experiment_id"]),
                ExperimentState.REPAIRING,
                ExperimentState.FAILED_FINAL,
                payload={"reason": f"repair failed: {type(error).__name__}: {str(error)[-500:]}"},
                idempotency_key=f"{reservation['repair_id']}:failed-final",
            )
            self._count_failed_transaction(
                repository, context, str(experiment["experiment_id"])
            )
            return False
        target = {
            "cheap": ExperimentState.CHEAP_RUNNING,
            "full": ExperimentState.FULL_RUNNING,
            "official_valid": ExperimentState.OFFICIAL_VALID_RUNNING,
        }[phase]
        repository.transition_experiment(
            str(experiment["experiment_id"]),
            ExperimentState.REPAIRING,
            target,
            payload={"repair_id": reservation["repair_id"], "phase": phase},
            idempotency_key=f"{reservation['repair_id']}:completed",
        )
        return True

    @staticmethod
    def _recover_pending_repair_evidence(
        context: ProductionContext,
        experiment_id: str,
        phase: str,
        repair_number: int,
    ) -> tuple[ArtifactRef, ...] | None:
        directory = context.run_dir / "evidence" / experiment_id / "repairs"
        update_path = directory / f"repair-{repair_number}-candidate-update.json"
        override_path = directory / f"repair-{repair_number}-{phase}.json"
        marker = update_path if update_path.is_file() else override_path
        if not marker.is_file():
            return None
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if (
                str(payload.get("experiment_id", experiment_id)) != experiment_id
                or int(payload["repair_number"]) != repair_number
            ):
                return None
            if marker == override_path and str(payload["phase"]) != phase:
                return None
            config_value = payload.get("config_path") or payload.get("repaired_config_path")
            config_sha256 = payload.get("config_sha256") or payload.get(
                "repaired_config_sha256"
            )
            refs: list[ArtifactRef] = []
            if config_value:
                config_path = Path(str(config_value)).resolve()
                if (
                    not config_path.is_file()
                    or sha256_file(config_path) != config_sha256
                    or not ProductionAutopilot._is_relative_to(
                        config_path, context.run_dir.resolve()
                    )
                ):
                    return None
                refs.append(artifact_ref(config_path, "repaired_experiment_config"))
            refs.append(
                artifact_ref(
                    marker,
                    "repair_candidate_update" if marker == update_path else "repair_override",
                )
            )
            return tuple(refs)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return None

    def _apply_repair_revision(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        experiment_id: str,
        repair_id: str,
        repair_number: int,
        evidence: tuple[ArtifactRef, ...],
    ) -> None:
        config_refs = [item for item in evidence if item.kind == "repaired_experiment_config"]
        update_refs = [item for item in evidence if item.kind == "repair_candidate_update"]
        if len(config_refs) > 1 or len(update_refs) > 1:
            raise RuntimeError("repair returned conflicting provenance revisions")
        if update_refs and not config_refs:
            raise RuntimeError("live repair update omitted its effective repaired config")
        if not config_refs:
            return
        config_ref = config_refs[0]
        config_path = Path(config_ref.path).resolve()
        if not self._is_relative_to(config_path, context.run_dir.resolve()):
            raise RuntimeError("repaired config must be durable evidence inside the run directory")
        if not config_path.is_file() or sha256_file(config_path) != config_ref.sha256:
            raise RuntimeError("repaired config is missing or has drifted")

        experiment = repository.get_experiment(experiment_id)
        repaired_commit = str(experiment["commit_sha"])
        if update_refs:
            update_ref = update_refs[0]
            update_path = Path(update_ref.path).resolve()
            if (
                not self._is_relative_to(update_path, context.run_dir.resolve())
                or not update_path.is_file()
                or sha256_file(update_path) != update_ref.sha256
            ):
                raise RuntimeError("repair candidate update is missing or has drifted")
            update = json.loads(update_path.read_text(encoding="utf-8"))
            required = {
                "experiment_id",
                "repair_number",
                "commit_sha",
                "workspace_path",
                "config_path",
                "config_sha256",
            }
            if not required.issubset(update):
                raise RuntimeError("repair candidate update is incomplete")
            if (
                update["experiment_id"] != experiment_id
                or int(update["repair_number"]) != repair_number
                or Path(str(update["config_path"])).resolve() != config_path
                or str(update["config_sha256"]) != config_ref.sha256
            ):
                raise RuntimeError("repair candidate update conflicts with durable provenance")
            workspace = Path(str(update["workspace_path"])).resolve()
            expected_workspace = Path(str(experiment["workspace_path"])).resolve()
            if workspace != expected_workspace or not self._is_relative_to(
                workspace, (context.run_dir / "worktrees").resolve()
            ):
                raise RuntimeError("repair candidate update names an unexpected worktree")
            repaired_commit = str(update["commit_sha"])
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
            clean = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
            if (
                head.returncode != 0
                or head.stdout.strip() != repaired_commit
                or clean.returncode != 0
                or clean.stdout.strip()
            ):
                raise RuntimeError("repaired candidate worktree commit is missing or dirty")
        repository.apply_experiment_repair_revision(
            repair_id,
            repaired_commit_sha=repaired_commit,
            effective_config_artifact_id=config_ref.artifact_id,
        )

    def _count_failed_transaction(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        experiment_id: str,
    ) -> None:
        experiment = repository.get_experiment(experiment_id)
        repository.count_failed_transaction(
            run_id=context.run_id,
            experiment_id=experiment_id,
            reason=str(experiment.get("terminal_reason") or "production candidate failed"),
            patience=self.budget.convergence_patience,
        )

    @staticmethod
    def _repair_rows(
        repository: ExperimentRepository, experiment_id: str
    ) -> list[dict[str, Any]]:
        with repository.database.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM experiment_repairs WHERE experiment_id=?", (experiment_id,)
                )
            ]

    @staticmethod
    def _repair_phase(repository: ExperimentRepository, repair_id: str) -> str:
        with repository.database.connect() as connection:
            row = connection.execute(
                "SELECT phase FROM experiment_repairs WHERE repair_id=?", (repair_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError(f"missing repair reservation {repair_id}")
        return str(row["phase"])

    @staticmethod
    def _experiment_evidence(
        repository: ExperimentRepository, experiment_id: str
    ) -> list[str]:
        with repository.database.connect() as connection:
            return [
                str(row["artifact_id"])
                for row in connection.execute(
                    "SELECT DISTINCT artifact_id FROM artifact_links WHERE experiment_id=? "
                    "ORDER BY artifact_id",
                    (experiment_id,),
                )
            ]

    def _cleanup_worktree(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        experiment_id: str,
    ) -> None:
        if not self.config.cleanup_worktrees:
            return
        experiment = repository.get_experiment(experiment_id)
        workspace_raw = experiment.get("workspace_path")
        if not workspace_raw:
            return
        workspace = Path(str(workspace_raw)).resolve()
        root = (context.run_dir / "worktrees").resolve()
        try:
            workspace.relative_to(root)
        except ValueError as error:
            raise RuntimeError("refusing to clean a worktree outside the run worktree root") from error
        with repository.database.connect() as connection:
            prior = connection.execute(
                "SELECT 1 FROM artifact_links link JOIN artifacts artifact "
                "ON artifact.artifact_id=link.artifact_id WHERE link.experiment_id=? "
                "AND artifact.kind='worktree_cleanup' LIMIT 1",
                (experiment_id,),
            ).fetchone()
        if prior is not None:
            return
        removed = not workspace.exists()
        stderr = ""
        if not removed:
            result = subprocess.run(
                ["git", "worktree", "remove", "--force", str(workspace)],
                cwd=context.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            removed = result.returncode == 0 and not workspace.exists()
            stderr = result.stderr[-1000:]
        evidence_path = context.run_dir / "evidence" / experiment_id / "worktree-cleanup.json"
        atomic_write_json(
            evidence_path,
            {
                "workspace_path": str(workspace),
                "branch_name": experiment.get("branch_name"),
                "commit_sha": experiment.get("commit_sha"),
                "removed": removed,
                "stderr": stderr,
            },
        )
        ref = artifact_ref(evidence_path, "worktree_cleanup")
        repository.register_artifact(ref, experiment_id=experiment_id)

    def _finalize(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
    ) -> dict[str, Any]:
        run = repository.get_run(context.run_id)
        if RunState(run["state"]) != RunState.FINALIZING:
            raise RuntimeError("production finalization requires FINALIZING state")
        with repository.database.connect() as connection:
            experiment_ids = [
                str(row["experiment_id"])
                for row in connection.execute(
                    "SELECT experiment_id FROM experiments WHERE run_id=?", (context.run_id,)
                )
            ]
        for experiment_id in experiment_ids:
            self._cleanup_worktree(repository, context, experiment_id)
        report = build_report(repository.database, context.run_id, context.run_dir / "report")
        best_valid = self._create_best_valid(repository, context, report)
        repository.transition_run(
            context.run_id, RunState.FINALIZING, RunState.COMPLETE, run["stop_reason"]
        )
        return {**self.status(context.run_id), "report": report, "best_valid_bundle": best_valid}

    def _create_best_valid(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        report: dict[str, Any],
    ) -> dict[str, Any] | None:
        run = repository.get_run(context.run_id)
        champion = run["search_champion_experiment_id"]
        if not champion:
            return None
        evidence_index = ArtifactRef.model_validate(report["evidence_index"])
        if champion == "baseline":
            return self._create_baseline_best_valid(repository, context, evidence_index)
        experiment = repository.get_experiment(str(champion))
        refs = self._artifact_refs(repository, str(champion))
        model = next((item for item in refs if item.kind == "model_bundle"), None)
        predictions = next(
            (item for item in refs if item.kind in {"valid_predictions", "predictions"}), None
        )
        config_ref = self._effective_config_artifact(repository, experiment, refs)
        if model is None or predictions is None or config_ref is None:
            return {
                "deferred": "validation-best artifacts are incomplete",
                "experiment_id": champion,
                "artifact_ids": [item.artifact_id for item in refs],
            }
        with repository.database.connect() as connection:
            metric_row = connection.execute(
                "SELECT * FROM metrics WHERE experiment_id=? AND split='valid' "
                "ORDER BY metric_id DESC LIMIT 1",
                (champion,),
            ).fetchone()
        if metric_row is None:
            return None
        metrics = Metrics(
            GAUC=metric_row["gauc"],
            **{"nDCG@5": metric_row["ndcg5"]},
            primary=metric_row["primary_score"],
            users=metric_row["users"],
            rows=metric_row["rows"],
            evaluator_sha256=metric_row["evaluator_sha256"],
            split=metric_row["split"],
            fold=metric_row["fold"],
            seed=metric_row["seed"],
        )
        manifest = create_best_valid_bundle(
            context.run_dir / "best-valid",
            run_id=context.run_id,
            experiment_id=str(champion),
            model_bundle=model,
            valid_predictions=predictions,
            evidence_index=evidence_index,
            metrics=metrics,
            commit_sha=str(experiment["commit_sha"]),
            config_path=config_ref.path,
            config_sha256=str(experiment["config_sha256"]),
            additional_evidence=[
                item
                for item in refs
                if item.artifact_id
                not in {model.artifact_id, predictions.artifact_id, config_ref.artifact_id}
            ],
        )
        repository.register_artifact(manifest, experiment_id=str(champion))
        return manifest.model_dump(mode="json")

    @staticmethod
    def _effective_config_artifact(
        repository: ExperimentRepository,
        experiment: dict[str, Any],
        refs: list[ArtifactRef],
    ) -> ArtifactRef | None:
        with repository.database.connect() as connection:
            repair = connection.execute(
                "SELECT effective_config_artifact_id FROM experiment_repairs "
                "WHERE experiment_id=? AND effective_config_artifact_id IS NOT NULL "
                "ORDER BY repair_number DESC LIMIT 1",
                (experiment["experiment_id"],),
            ).fetchone()
        if repair is not None:
            selected = next(
                (
                    item
                    for item in refs
                    if item.artifact_id == repair["effective_config_artifact_id"]
                ),
                None,
            )
            if selected is not None and selected.sha256 == experiment["config_sha256"]:
                return selected
            return None
        return next(
            (
                item
                for item in refs
                if item.kind == "experiment_config"
                and item.sha256 == experiment["config_sha256"]
            ),
            None,
        )

    def _create_baseline_best_valid(
        self,
        repository: ExperimentRepository,
        context: ProductionContext,
        evidence_index: ArtifactRef,
    ) -> dict[str, Any] | None:
        """Freeze the selected verified baseline without inventing an experiment FK."""

        baseline = repository.get_baseline(context.run_id)
        if baseline is None:
            return None
        evidence_ids = list(json.loads(baseline["evidence_json"]))
        with repository.database.connect() as connection:
            rows = connection.execute(
                f"SELECT artifact_id,kind,path,sha256,size_bytes,schema_version FROM artifacts "
                f"WHERE artifact_id IN ({','.join('?' for _ in evidence_ids)})",
                evidence_ids,
            ).fetchall()
        refs = [ArtifactRef.model_validate(dict(row)) for row in rows]
        gate_ref = next((item for item in refs if item.kind == "baseline_gate"), None)
        if gate_ref is None or not Path(gate_ref.path).is_file():
            return {
                "deferred": "validated baseline gate artifact is incomplete",
                "experiment_id": "baseline",
                "artifact_ids": evidence_ids,
            }
        if sha256_file(gate_ref.path) != gate_ref.sha256:
            raise RuntimeError("baseline gate drifted before finalization")
        gate = json.loads(Path(gate_ref.path).read_text(encoding="utf-8"))
        selected = dict(gate.get("selected_seed", {}))
        required = {
            "prediction_artifact_id",
            "prediction_path",
            "prediction_sha256",
            "model_bundle_artifact_id",
            "model_bundle_path",
            "model_bundle_sha256",
            "config_artifact_id",
            "config_path",
            "config_sha256",
            "metrics",
        }
        if not required.issubset(selected):
            return {
                "deferred": "validated baseline selection is incomplete",
                "experiment_id": "baseline",
                "artifact_ids": evidence_ids,
            }
        metrics = Metrics.model_validate(selected["metrics"])
        baseline_root = (context.run_dir / "baseline").resolve()
        for field_name in ("prediction_path", "model_bundle_path", "config_path"):
            if not self._is_relative_to(
                Path(str(selected[field_name])).resolve(), baseline_root
            ):
                raise RuntimeError("baseline selection references evidence outside the run")

        by_id = {item.artifact_id: item for item in refs}
        try:
            model_bundle = by_id[str(selected["model_bundle_artifact_id"])]
            predictions = by_id[str(selected["prediction_artifact_id"])]
            config_ref = by_id[str(selected["config_artifact_id"])]
        except KeyError as error:
            raise RuntimeError("baseline gate references an unregistered selected artifact") from error
        for ref, expected_path, expected_sha256 in (
            (model_bundle, selected["model_bundle_path"], selected["model_bundle_sha256"]),
            (predictions, selected["prediction_path"], selected["prediction_sha256"]),
            (config_ref, selected["config_path"], selected["config_sha256"]),
        ):
            if Path(ref.path).resolve() != Path(str(expected_path)).resolve():
                raise RuntimeError("baseline gate selected-artifact path differs from the registry")
            if ref.sha256 != str(expected_sha256):
                raise RuntimeError("baseline gate selected-artifact hash differs from the registry")
        manifest = create_best_valid_bundle(
            context.run_dir / "best-valid",
            run_id=context.run_id,
            experiment_id="baseline",
            model_bundle=model_bundle,
            valid_predictions=predictions,
            evidence_index=evidence_index,
            metrics=metrics,
            commit_sha=context.root_commit,
            config_path=config_ref.path,
            config_sha256=str(selected["config_sha256"]),
            additional_evidence=(gate_ref,),
        )
        repository.register_artifact(manifest)
        return manifest.model_dump(mode="json")

    @staticmethod
    def _artifact_refs(
        repository: ExperimentRepository, experiment_id: str
    ) -> list[ArtifactRef]:
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT artifact.artifact_id,artifact.kind,link.artifact_path AS path,"
                "artifact.sha256,artifact.size_bytes,artifact.schema_version FROM artifact_links link "
                "JOIN artifacts artifact ON artifact.artifact_id=link.artifact_id "
                "WHERE link.experiment_id=? ORDER BY artifact.created_at",
                (experiment_id,),
            ).fetchall()
        return [ArtifactRef.model_validate(dict(row)) for row in rows]


def run_production_autopilot(
    config_path: str | Path,
    *,
    provider: StructuredProvider,
    hooks: ProductionHooks | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    config = ProductionRunConfig.load(config_path)
    return ProductionAutopilot(config, provider, hooks).run(run_id=run_id)
