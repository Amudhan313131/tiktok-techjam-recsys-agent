"""Short deterministic rehearsals for contracts, persistence, and recovery plumbing."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from rex.agents.patch_guard import PatchPolicy, PatchRejected, validate_patch
from rex.agents.provider import (
    ProviderResponse,
    ProviderRouter,
    ProviderSchemaError,
    ProviderTimeoutError,
    StructuredProvider,
)
from rex.contracts import (
    AttemptStatus,
    ExperimentProposal,
    ExperimentState,
    Metrics,
    Operator,
    RunState,
)
from rex.control.production_supervisor import (
    BaselineGateResult,
    CandidatePreparation,
    ComparisonObservation,
    ProductionAutopilot,
    ProductionContext,
    ProductionFixedProvider,
    ProductionRungFailure,
    ProductionRungResult,
    ProductionRunConfig,
    RepairRequest,
    RungRequest,
)
from rex.control.budget import deadline_epoch_ms
from rex.control.fixture_supervisor import FixtureAutopilot, FixtureRunConfig, FixtureScriptProvider
from rex.data.manifest import verify_starter_manifest
from rex.execution.artifacts import artifact_ref
from rex.reporting.report import build_report
from rex.store.db import Database
from rex.store.event_log import export_events, verify_event_chain
from rex.store.repository import ExperimentRepository


def _proposal(experiment_id: str) -> ExperimentProposal:
    return ExperimentProposal(
        experiment_id=experiment_id,
        parent_id=None,
        operator=Operator.LOSS,
        hypothesis="Same-user ranking should improve ordering over pointwise loss.",
        mechanism="Contrasts positive and negative impressions belonging to one user.",
        primary_change="pairwise fixture loss",
        files_to_change=["src/rex/losses/experimental/fixture.py"],
        expected_metric_effects={"primary": "positive"},
        falsifier="No cheap-fold improvement above 0.001.",
        leakage_analysis="Uses train labels only within complete user groups.",
        estimated_seconds=10,
        cheap_rung={"fold": "A"},
        full_rung={"folds": ["A", "B", "C"]},
    )


def run_r0(output_dir: str | Path) -> dict[str, object]:
    """Offline fixture rehearsal: restart-safe transitions and event integrity."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    database = Database(output / "state.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    run_id = f"r0-{uuid.uuid4().hex[:10]}"
    starter = verify_starter_manifest()
    repository.create_run(
        run_id=run_id,
        deadline_epoch_ms=deadline_epoch_ms(900),
        root_commit="fixture",
        environment_sha256="0" * 64,
        data_manifest_sha256="1" * 64,
        evaluator_sha256=starter.hashes["evaluate.py"],
    )
    repository.transition_run(run_id, RunState.INITIALIZING, RunState.BASELINE_VERIFYING)
    repository.transition_run(run_id, RunState.BASELINE_VERIFYING, RunState.SEARCHING)
    experiment_id = f"fixture-{uuid.uuid4().hex[:10]}"
    repository.create_experiment(run_id, _proposal(experiment_id), "fixture")
    transitions = (
        (ExperimentState.PROPOSED, ExperimentState.WORKTREE_READY),
        (ExperimentState.WORKTREE_READY, ExperimentState.PATCHED),
        (ExperimentState.PATCHED, ExperimentState.STATIC_VALID),
        (ExperimentState.STATIC_VALID, ExperimentState.FIXTURE_VALID),
        (ExperimentState.FIXTURE_VALID, ExperimentState.CHEAP_RUNNING),
        (ExperimentState.CHEAP_RUNNING, ExperimentState.CHEAP_COMPLETE),
        (ExperimentState.CHEAP_COMPLETE, ExperimentState.REJECTED),
    )
    for index, (current, target) in enumerate(transitions):
        repository.transition_experiment(
            experiment_id,
            current,
            target,
            idempotency_key=f"r0:{experiment_id}:{index}",
        )
    # Replay the last transition to prove exactly-once behavior.
    repository.transition_experiment(
        experiment_id,
        ExperimentState.CHEAP_COMPLETE,
        ExperimentState.REJECTED,
        idempotency_key=f"r0:{experiment_id}:{len(transitions) - 1}",
    )
    event_path = output / "events.jsonl"
    exported = export_events(database, run_id, event_path)
    return {
        "level": "R0",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "state": repository.get_experiment(experiment_id)["state"],
        "events": exported,
        "event_chain_valid": verify_event_chain(event_path),
    }


def rehearsal_requirements(level: str) -> dict[str, object]:
    normalized = level.upper()
    if normalized == "R0":
        return {"data": False, "live_llm": False, "target_minutes": 15}
    if normalized == "FIXTURE":
        return {
            "data": False,
            "live_llm": False,
            "target_minutes": 15,
            "production_science": False,
            "final_submission": False,
        }
    if normalized == "R1":
        return {
            "data": False,
            "live_llm": False,
            "target_minutes": 15,
            "production_control_plane": True,
            "final_submission": False,
        }
    if normalized == "R2":
        return {
            "data": False,
            "live_llm": False,
            "target_minutes": 20,
            "production_control_plane": True,
            "provider_faults": True,
            "final_submission": False,
        }
    raise ValueError(f"unsupported short rehearsal: {level}")


class _InterruptOnceProvider:
    """Inject exactly one retryable LLM interruption, then delegate normally."""

    def __init__(self, delegate: StructuredProvider):
        self.delegate = delegate
        self.calls = 0
        self.interruptions = 0

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        self.calls += 1
        if self.interruptions == 0:
            self.interruptions += 1
            raise ProviderTimeoutError("controlled fixture provider interruption")
        return self.delegate.generate(role=role, system=system, prompt=prompt, schema=schema)


def run_fixture_rehearsal(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run the connected autopilot on generated fixtures with controlled faults.

    This is intentionally not the deferred six-hour dress rehearsal. It never
    opens competition data, confirms a model, promotes an incumbent, or creates
    a submission.
    """

    config = FixtureRunConfig.load(config_path)
    config = replace(
        config,
        runs_dir=Path(output_dir).resolve(),
        max_hypotheses=4,
        inject_worker_nan_once=True,
        inject_worker_nan_always_iteration=4,
    )
    interrupted = _InterruptOnceProvider(FixtureScriptProvider())
    provider = ProviderRouter(
        {"fixed": interrupted},
        mode="fixed",
        retries=2,
    )
    started = time.monotonic()
    result = FixtureAutopilot(config, provider).run()
    event_path = Path(str(result["run_dir"])) / "report" / "events.jsonl"

    protected_patch = (
        "--- a/src/rex/control/fixture_supervisor.py\n"
        "+++ b/src/rex/control/fixture_supervisor.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-protected\n"
        "+modified\n"
    )
    protected_patch_rejected = False
    try:
        validate_patch(protected_patch, PatchPolicy.from_yaml(config.protected_paths))
    except PatchRejected:
        protected_patch_rejected = True

    experiment_states = [str(item["state"]) for item in result["experiments"]]
    database = Database(Path(str(result["run_dir"])) / "state.sqlite3")
    repository = ExperimentRepository(database)
    with database.connect() as connection:
        worker_attempts = [
            {"repair_number": row["repair_number"], "status": row["status"]}
            for row in connection.execute(
                "SELECT repair_number,status FROM attempts WHERE experiment_id='fixture-001' "
                "AND rung='cheap' ORDER BY repair_number"
            )
        ]
        exhausted_worker_attempts = [
            {"repair_number": row["repair_number"], "status": row["status"]}
            for row in connection.execute(
                "SELECT repair_number,status FROM attempts WHERE experiment_id='fixture-004' "
                "AND rung='cheap' ORDER BY repair_number"
            )
        ]
        exhausted_worker = connection.execute(
            "SELECT state FROM experiments WHERE experiment_id='fixture-004'"
        ).fetchone()
    worker_nan_recovered = worker_attempts == [
        {"repair_number": 0, "status": AttemptStatus.NAN},
        {"repair_number": 1, "status": AttemptStatus.SUCCESS},
    ]
    worker_repair_limit_enforced = exhausted_worker_attempts == [
        {"repair_number": 0, "status": AttemptStatus.NAN},
        {"repair_number": 1, "status": AttemptStatus.NAN},
        {"repair_number": 2, "status": AttemptStatus.NAN},
    ] and exhausted_worker is not None and exhausted_worker["state"] == "FAILED_FINAL"
    repository.record_intervention(
        intervention_id=f"{result['run_id']}:provider-interruption",
        run_id=str(result["run_id"]),
        actor="fixture_rehearsal",
        action="controlled_provider_interruption",
        reason="prove bounded retry and continuation without a second hypothesis",
        evidence={"interruptions": interrupted.interruptions, "provider_calls": interrupted.calls},
    )
    repository.record_intervention(
        intervention_id=f"{result['run_id']}:protected-patch",
        run_id=str(result["run_id"]),
        actor="fixture_rehearsal",
        action="protected_patch_probe",
        reason="prove protected control-plane edits fail closed",
        evidence={"rejected": protected_patch_rejected},
    )
    repository.record_intervention(
        intervention_id=f"{result['run_id']}:worker-nan",
        run_id=str(result["run_id"]),
        actor="fixture_rehearsal",
        action="controlled_worker_nan",
        reason="prove a typed worker failure is retried once and the candidate continues",
        evidence={"recovered": worker_nan_recovered, "attempts": worker_attempts},
    )
    repository.record_intervention(
        intervention_id=f"{result['run_id']}:worker-repair-exhausted",
        run_id=str(result["run_id"]),
        actor="fixture_rehearsal",
        action="controlled_worker_repair_exhausted",
        reason="prove persistent failure stops after two repairs and preserves the run",
        evidence={
            "enforced": worker_repair_limit_enforced,
            "attempts": exhausted_worker_attempts,
            "final_state": None if exhausted_worker is None else exhausted_worker["state"],
        },
    )
    result["report"] = build_report(
        database, str(result["run_id"]), Path(str(result["run_dir"])) / "report"
    )
    event_chain_valid = verify_event_chain(event_path)
    return {
        **result,
        "level": "FIXTURE",
        "elapsed_seconds": time.monotonic() - started,
        "event_chain_valid": event_chain_valid,
        "provider_interruption_recovered": interrupted.interruptions == 1,
        "provider_calls": interrupted.calls,
        "worker_nan_recovered": worker_nan_recovered,
        "worker_repair_limit_enforced": worker_repair_limit_enforced,
        "protected_patch_rejected": protected_patch_rejected,
        "production_promotion_blocked": "PROMOTED" not in experiment_states,
        "scope_note": "Short generated-fixture rehearsal; not the deferred six-hour run.",
    }


class RehearsalInterruption(RuntimeError):
    """Controlled coordinator interruption used only by the offline rehearsal."""


def _metric(primary: float, *, split: str, fold: str | None) -> Metrics:
    return Metrics(
        GAUC=primary,
        **{"nDCG@5": primary},
        primary=primary,
        users=4,
        rows=8,
        evaluator_sha256="0" * 64,
        split=split,
        fold=fold,
        seed=0,
    )


class _ControlledDiagnosisProvider:
    """Diagnosis provider that can be interrupted or return one malformed response."""

    def __init__(self, fault: str | None = None):
        self.fault = fault
        self.calls = 0
        self.faults_injected = 0

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        del system, schema
        if role != "diagnosis":
            return ProductionFixedProvider().generate(
                role=role,
                system="",
                prompt=prompt,
                schema={},
            )
        self.calls += 1
        context = json.loads(prompt)
        if self.fault is not None and self.faults_injected == 0:
            self.faults_injected += 1
            if self.fault == "timeout":
                raise ProviderTimeoutError("controlled production diagnosis interruption")
            if self.fault == "schema":
                raise ProviderSchemaError("controlled production diagnosis schema failure")
            raise RuntimeError(f"unsupported diagnosis rehearsal fault: {self.fault}")
        return ProviderResponse(
            value={
                "experiment_id": context["experiment_id"],
                "outcome": "supported",
                "evidence_artifact_ids": context["artifact_ids"],
                "metric_deltas": {"primary": 0.004},
                "uncertainty": "Controlled offline evidence is not model-quality evidence.",
                "next_operator": "ABANDON",
                "reusable_lesson": "Durable evidence lets diagnosis resume after interruption.",
            },
            provider="rehearsal",
            model="controlled-diagnosis-v1",
        )


class _ControlledProductionHooks:
    """Deterministic production adapter with typed, one-shot fault injection."""

    def __init__(
        self,
        *,
        rung_faults: dict[tuple[str, str], list[AttemptStatus | str]] | None = None,
        preparation_interrupt: str | None = None,
    ):
        self.rung_faults = {
            key: list(values) for key, values in (rung_faults or {}).items()
        }
        self.preparation_interrupt = preparation_interrupt
        self.preparation_interruptions = 0
        self.rung_calls: dict[tuple[str, str], int] = {}
        self.repairs: list[RepairRequest] = []

    @staticmethod
    def _artifact(
        context: ProductionContext,
        name: str,
        content: str,
        *,
        kind: str = "rehearsal_evidence",
    ):
        path = context.run_dir / "rehearsal-evidence" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return artifact_ref(path, kind)

    def verify_baseline(self, context: ProductionContext) -> BaselineGateResult:
        evidence = self._artifact(context, "baseline.json", '{"primary":0.5}')
        return BaselineGateResult(
            accepted=True,
            metrics=_metric(0.5, split="valid", fold=None),
            artifacts=(evidence,),
        )

    def prepare_candidate(
        self,
        context: ProductionContext,
        card: Any,
        binding: Any,
        proposal_context: dict[str, object],
        parent_commit: str,
    ) -> CandidatePreparation:
        del binding
        if (
            self.preparation_interrupt == card.card_id
            and self.preparation_interruptions == 0
        ):
            self.preparation_interruptions += 1
            raise RehearsalInterruption(
                f"controlled interruption while preparing {card.card_id}"
            )
        incumbent = dict(proposal_context["incumbent"])
        allowed_files = list(proposal_context["allowed_files"])
        proposal = ExperimentProposal(
            experiment_id=f"rehearsal-{card.card_id.lower()}",
            parent_id=incumbent.get("experiment_id"),
            operator=Operator.LOSS if card.card_id == "E01" else Operator.MODEL_BLOCK,
            hypothesis=f"Controlled {card.card_id} evidence should improve the current incumbent.",
            mechanism="A deterministic hook changes exactly one controlled experiment variable.",
            primary_change=f"controlled {card.card_id} method-card change",
            files_to_change=[str(allowed_files[0])],
            expected_metric_effects={"primary": "increase"},
            falsifier="The trusted temporal evidence gate rejects the candidate.",
            leakage_analysis="Only generated offline rehearsal values are available to this hook.",
            estimated_seconds=1,
            cheap_rung={"fold": "A", "complete_users": True},
            full_rung={"folds": ["A", "B", "C"]},
        )
        evidence = self._artifact(
            context,
            f"{card.card_id.lower()}-preparation.json",
            json.dumps(
                {
                    "card": card.card_id,
                    "parent_commit": parent_commit,
                    "offline_rehearsal": True,
                },
                sort_keys=True,
            ),
        )
        return CandidatePreparation(proposal, parent_commit, artifacts=(evidence,))

    def run_rung(self, request: RungRequest) -> ProductionRungResult:
        card_id = request.method_card.card_id
        key = (card_id, request.rung)
        self.rung_calls[key] = self.rung_calls.get(key, 0) + 1
        queue = self.rung_faults.get(key, [])
        if queue:
            fault = queue.pop(0)
            if fault == "interrupt":
                raise RehearsalInterruption(
                    f"controlled coordinator interruption during {card_id}/{request.rung}"
                )
            status = AttemptStatus(fault)
            raise ProductionRungFailure(
                status,
                f"controlled {status} during {card_id}/{request.rung}",
            )
        if request.rung == "cheap":
            observations = (
                ComparisonObservation(
                    _metric(0.503, split="shadow", fold="cheap"),
                    _metric(0.5, split="shadow", fold="cheap"),
                ),
            )
        elif request.rung == "full":
            observations = tuple(
                ComparisonObservation(
                    _metric(0.503 + index * 0.0001, split="shadow", fold=fold),
                    _metric(0.5, split="shadow", fold=fold),
                )
                for index, fold in enumerate(("A", "B", "C"))
            )
        else:
            observations = (
                ComparisonObservation(
                    _metric(0.504, split="valid", fold=None),
                    _metric(0.5, split="valid", fold=None),
                ),
            )
        evidence = self._artifact(
            request.context,
            f"{card_id.lower()}-{request.rung}-{self.rung_calls[key]}.json",
            json.dumps(
                {
                    "card": card_id,
                    "rung": request.rung,
                    "controlled": True,
                },
                sort_keys=True,
            ),
        )
        return ProductionRungResult(observations=observations, artifacts=(evidence,))

    def repair_candidate(self, request: RepairRequest):
        self.repairs.append(request)
        evidence = self._artifact(
            request.context,
            f"{request.experiment['experiment_id']}-repair-{request.plan.repair_number}.json",
            json.dumps(
                {
                    "phase": request.phase,
                    "action": request.plan.action,
                    "repair_number": request.plan.repair_number,
                    "overrides": request.plan.overrides,
                },
                sort_keys=True,
            ),
            kind="repair_evidence",
        )
        return (evidence,)


class _FinalizationInterruptAutopilot(ProductionAutopilot):
    def __init__(self, *args: Any, interrupt_finalization: bool, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.interrupt_finalization = interrupt_finalization
        self.finalization_interruptions = 0

    def _finalize(self, repository: ExperimentRepository, context: ProductionContext):
        if self.interrupt_finalization and self.finalization_interruptions == 0:
            self.finalization_interruptions += 1
            raise RehearsalInterruption("controlled interruption before final bundle publication")
        return super()._finalize(repository, context)


def _git(project: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"rehearsal git command failed: {completed.stderr[-1000:]}")
    return completed.stdout.strip()


def _prepare_production_rehearsal(output: Path) -> ProductionRunConfig:
    """Create a committed, generated production-control fixture outside the real project."""

    project = output / "control-project"
    project.mkdir(parents=True, exist_ok=True)
    files = {
        "requirements-lock.txt": "rehearsal-lock==1\n",
        "pyproject.toml": "[project]\nname='rex-rehearsal'\nversion='0.0.0'\n",
        "data-manifest.json": '{"fixture":true}\n',
        "evaluate.py": "# controlled evaluator identity only\n",
        "protected.yaml": (
            "allow:\n  - configs/experiments/**\n"
            "deny:\n  - src/rex/control/**\n  - src/rex/store/**\n"
        ),
        "e01.yaml": "plugin: controlled-pairwise\n",
        "e02.yaml": "plugin: controlled-tree\n",
    }
    for name, value in files.items():
        (project / name).write_text(value, encoding="utf-8")
    if not (project / ".git").is_dir():
        _git(project, "init")
        _git(project, "config", "user.email", "rex-rehearsal@example.invalid")
        _git(project, "config", "user.name", "REX Rehearsal")
        _git(project, "add", "--all")
        _git(project, "commit", "-m", "production control fixture")

    budget_path = output / "budget.yaml"
    budget_path.write_text(
        yaml.safe_dump(
            {
                "max_hypotheses": 2,
                "max_official_evaluations": 2,
                "wall_clock_seconds": 300,
                "finalization_reserve_seconds": 0,
                "convergence_epsilon": 0.002,
                "convergence_patience": 3,
                "max_repairs_per_experiment": 2,
                "default_attempt_timeout_seconds": 30,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = output / "production-rehearsal.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "execution_mode": "production",
                "scientific_execution_enabled": True,
                "project_root": str(project),
                "runs_dir": str(output / "runs"),
                "budget_config": str(budget_path),
                "protected_paths": str(project / "protected.yaml"),
                "data_manifest": str(project / "data-manifest.json"),
                "evaluator_path": str(project / "evaluate.py"),
                "environment_lock": str(project / "requirements-lock.txt"),
                "cleanup_worktrees": False,
                "process_stale_after_seconds": 1,
                "method_cards": {
                    "E01": {
                        "config": str(project / "e01.yaml"),
                        "feature_recipe": "control",
                    },
                    "E02": {
                        "config": str(project / "e02.yaml"),
                        "feature_recipe": "control",
                    },
                },
                "llm": {"mode": "controlled"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return ProductionRunConfig.load(config_path)


def _case_policy(card_ids: tuple[str, ...]):
    from rex.agents.search_policy import ExperimentCard, SearchPolicy

    mechanisms = {
        "E01": "controlled pairwise production-control rehearsal",
        "E02": "controlled tree production-control rehearsal",
    }
    return SearchPolicy(
        tuple(ExperimentCard(card_id, mechanisms[card_id]) for card_id in card_ids)
    )


def _database_snapshot(config: ProductionRunConfig, run_id: str) -> dict[str, Any]:
    database = Database(config.runs_dir / run_id / "state.sqlite3")
    with database.connect() as connection:
        run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        experiments = connection.execute(
            "SELECT experiment_id,state FROM experiments WHERE run_id=? ORDER BY iteration_number",
            (run_id,),
        ).fetchall()
        repairs = connection.execute(
            "SELECT repair.experiment_id,repair.repair_number,repair.failure_status "
            "FROM experiment_repairs repair JOIN experiments experiment "
            "ON experiment.experiment_id=repair.experiment_id WHERE experiment.run_id=? "
            "ORDER BY repair.experiment_id,repair.repair_number",
            (run_id,),
        ).fetchall()
        sessions = connection.execute(
            "SELECT session_id,exit_reason FROM process_sessions WHERE run_id=? ORDER BY started_at",
            (run_id,),
        ).fetchall()
        promotions = connection.execute(
            "SELECT previous_experiment_id,experiment_id FROM search_promotions "
            "WHERE run_id=? ORDER BY created_at",
            (run_id,),
        ).fetchall()
    return {
        "state": None if run is None else run["state"],
        "champion": None if run is None else run["search_champion_experiment_id"],
        "experiments": [dict(row) for row in experiments],
        "repairs": [dict(row) for row in repairs],
        "sessions": [dict(row) for row in sessions],
        "promotions": [dict(row) for row in promotions],
    }


def _assert_case_acceptance(
    name: str,
    result: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    expected_champion: str,
) -> dict[str, Any]:
    repair_numbers = [int(row["repair_number"]) for row in snapshot["repairs"]]
    max_repairs = max(repair_numbers, default=0)
    accepted = {
        "run_completed": result["state"] == RunState.COMPLETE,
        "incumbent_preserved_or_validly_improved": snapshot["champion"]
        == expected_champion,
        "repair_limit_respected": max_repairs <= 2,
        "confirmation_disabled": result["confirmation_enabled"] is False,
        "final_submission_disabled": result["final_submission_enabled"] is False,
    }
    return {
        "name": name,
        "passed": all(accepted.values()),
        "acceptance": accepted,
        "max_repair_number": max_repairs,
        "snapshot": snapshot,
    }


def _run_fault_case(
    config: ProductionRunConfig,
    name: str,
    hooks: _ControlledProductionHooks,
    *,
    card_ids: tuple[str, ...] = ("E01",),
    provider: _ControlledDiagnosisProvider | None = None,
    expect_interruption: bool = False,
    interrupt_finalization: bool = False,
    stale_takeover: bool = False,
    expected_champion: str = "rehearsal-e01",
) -> dict[str, Any]:
    run_id = f"rehearsal-{name.replace('_', '-')}"
    provider = provider or _ControlledDiagnosisProvider()
    policy = _case_policy(card_ids)
    autopilot: ProductionAutopilot
    if interrupt_finalization:
        autopilot = _FinalizationInterruptAutopilot(
            config,
            provider,
            hooks,
            search_policy=policy,
            interrupt_finalization=True,
        )
    else:
        autopilot = ProductionAutopilot(config, provider, hooks, search_policy=policy)
    interrupted = False
    interrupted_snapshot: dict[str, Any] | None = None
    try:
        result = autopilot.run(run_id=run_id)
    except (RehearsalInterruption, ProviderTimeoutError, ProviderSchemaError):
        if not expect_interruption:
            raise
        interrupted = True
        interrupted_snapshot = _database_snapshot(config, run_id)
        if stale_takeover:
            database = Database(config.runs_dir / run_id / "state.sqlite3")
            repository = ExperimentRepository(database)
            stale_at = datetime.now(timezone.utc) - timedelta(seconds=10)
            repository.open_process_session(
                session_id=f"{run_id}-orphaned-session",
                run_id=run_id,
                pid=os.getpid(),
                host="controlled-rehearsal",
                now=stale_at,
            )
        result = ProductionAutopilot(
            config,
            provider,
            hooks,
            search_policy=policy,
        ).run(run_id=run_id)
    if expect_interruption and not interrupted:
        raise AssertionError(f"fault case {name} did not inject its interruption")
    snapshot = _database_snapshot(config, run_id)
    case = _assert_case_acceptance(
        name,
        result,
        snapshot,
        expected_champion=expected_champion,
    )
    case.update(
        {
            "interruption_recovered": interrupted if expect_interruption else None,
            "interrupted_snapshot": interrupted_snapshot,
            "repair_actions": [
                {
                    "phase": request.phase,
                    "repair_number": request.plan.repair_number,
                    "action": request.plan.action,
                }
                for request in hooks.repairs
            ],
            "stale_takeover_recorded": (
                any(
                    row["exit_reason"] == "stale_takeover"
                    for row in snapshot["sessions"]
                )
                if stale_takeover
                else None
            ),
        }
    )
    if stale_takeover and not case["stale_takeover_recorded"]:
        case["passed"] = False
    return case


def _database_rollback_probe(config: ProductionRunConfig, run_id: str) -> bool:
    database = Database(config.runs_dir / run_id / "state.sqlite3")
    before = _database_snapshot(config, run_id)["champion"]
    try:
        with database.transaction() as connection:
            connection.execute(
                "UPDATE runs SET search_champion_experiment_id='corrupt-interrupted-value' "
                "WHERE run_id=?",
                (run_id,),
            )
            raise RehearsalInterruption("controlled database interruption")
    except RehearsalInterruption:
        pass
    after = _database_snapshot(config, run_id)["champion"]
    return before == after


def _security_probes(config: ProductionRunConfig) -> dict[str, bool]:
    from rex.execution.gate import GateExecutionError, execute_gate
    from rex.execution.sandbox import SandboxMode

    policy = PatchPolicy.from_yaml(config.protected_paths)
    protected_patch = (
        "--- a/src/rex/control/production_supervisor.py\n"
        "+++ b/src/rex/control/production_supervisor.py\n"
        "@@ -1,1 +1,1 @@\n-old\n+unsafe\n"
    )
    traversal_patch = (
        "--- a/configs/experiments/e01.yaml\n"
        "+++ b/../../outside.txt\n"
        "@@ -1,1 +1,1 @@\n-old\n+escape\n"
    )

    def rejected(patch: str) -> bool:
        try:
            validate_patch(patch, policy)
        except PatchRejected:
            return True
        return False

    sandbox_escape_rejected = False
    try:
        execute_gate(
            name="workspace-escape",
            command=("python", "-c", "print('must not execute')"),
            workspace=config.project_root,
            artifact_dir=config.runs_dir.parent / "security-probe",
            timeout_seconds=2,
            sandbox_mode=SandboxMode.PRODUCTION,
            trusted_worktree_root=config.project_root / "trusted-worktrees",
            trusted_output_root=config.runs_dir.parent,
        )
    except GateExecutionError:
        sandbox_escape_rejected = True

    return {
        "protected_patch_rejected": rejected(protected_patch),
        "patch_path_traversal_rejected": rejected(traversal_patch),
        "sandbox_workspace_escape_rejected": sandbox_escape_rejected,
    }


def _suite_result(level: str, cases: list[dict[str, Any]], started: float) -> dict[str, Any]:
    max_repairs = max((int(case["max_repair_number"]) for case in cases), default=0)
    return {
        "level": level,
        "scope": "offline production-control fault rehearsal",
        "cases": cases,
        "case_count": len(cases),
        "all_cases_passed": all(bool(case["passed"]) for case in cases),
        "max_repair_number_observed": max_repairs,
        "repair_limit_respected": max_repairs <= 2,
        "scientific_model_claim": False,
        "real_data_accessed": False,
        "confirmation_enabled": False,
        "six_hour_dress_rehearsal": False,
        "test_prediction_created": False,
        "final_submission_created": False,
        "elapsed_seconds": time.monotonic() - started,
    }


def run_r1(
    data_dir: str | Path | None,
    output_dir: str | Path,
    *,
    timeout_seconds: int = 1200,
) -> dict[str, object]:
    """Run offline production-control R1 with typed execution and restart faults.

    ``data_dir`` and ``timeout_seconds`` remain accepted for CLI compatibility,
    but R1 intentionally never opens real competition data or trains a model.
    """

    del data_dir, timeout_seconds
    started = time.monotonic()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _prepare_production_rehearsal(output)
    cases = [
        _run_fault_case(
            config,
            "preparation_interrupt",
            _ControlledProductionHooks(preparation_interrupt="E01"),
            expect_interruption=True,
        ),
        _run_fault_case(
            config,
            "training_timeout_descendants",
            _ControlledProductionHooks(
                rung_faults={("E01", "cheap"): [AttemptStatus.TIMEOUT]}
            ),
        ),
        _run_fault_case(
            config,
            "out_of_memory",
            _ControlledProductionHooks(
                rung_faults={("E01", "cheap"): [AttemptStatus.OOM]}
            ),
        ),
        _run_fault_case(
            config,
            "nan_predictions",
            _ControlledProductionHooks(
                rung_faults={("E01", "official_valid"): [AttemptStatus.NAN]}
            ),
        ),
        _run_fault_case(
            config,
            "corrupt_model_bundle",
            _ControlledProductionHooks(
                rung_faults={("E01", "full"): [AttemptStatus.INVALID_ARTIFACT]}
            ),
        ),
        _run_fault_case(
            config,
            "evaluator_failure",
            _ControlledProductionHooks(
                rung_faults={("E01", "official_valid"): [AttemptStatus.INVALID_ARTIFACT]}
            ),
        ),
        _run_fault_case(
            config,
            "misaligned_predictions",
            _ControlledProductionHooks(
                rung_faults={("E01", "official_valid"): [AttemptStatus.CONTRACT]}
            ),
        ),
    ]
    database_hooks = _ControlledProductionHooks(
        rung_faults={("E01", "cheap"): ["interrupt"]}
    )
    database_case = _run_fault_case(
        config,
        "database_interrupt_stale_takeover",
        database_hooks,
        expect_interruption=True,
        stale_takeover=True,
    )
    database_case["transaction_rollback_preserved_incumbent"] = _database_rollback_probe(
        config,
        "rehearsal-database-interrupt-stale-takeover",
    )
    database_case["passed"] = bool(database_case["passed"]) and bool(
        database_case["transaction_rollback_preserved_incumbent"]
    )
    cases.append(database_case)
    cases.append(
        _run_fault_case(
            config,
            "finalization_interrupt",
            _ControlledProductionHooks(),
            expect_interruption=True,
            interrupt_finalization=True,
        )
    )
    cases.append(
        _run_fault_case(
            config,
            "persistent_candidate_then_continue",
            _ControlledProductionHooks(
                rung_faults={
                    ("E01", "cheap"): [
                        AttemptStatus.NAN,
                        AttemptStatus.NAN,
                        AttemptStatus.NAN,
                    ]
                }
            ),
            card_ids=("E01", "E02"),
            expected_champion="rehearsal-e02",
        )
    )
    return _suite_result("R1", cases, started)


def run_r2(
    data_dir: str | Path | None,
    output_dir: str | Path,
    *,
    timeout_seconds: int = 1200,
) -> dict[str, object]:
    """Run R1 plus restart-safe provider and constrained-authority faults."""

    started = time.monotonic()
    output = Path(output_dir).resolve()
    integration = run_r1(data_dir, output / "r1", timeout_seconds=timeout_seconds)
    config = _prepare_production_rehearsal(output / "r2")
    provider_timeout = _ControlledDiagnosisProvider("timeout")
    provider_schema = _ControlledDiagnosisProvider("schema")
    cases = [
        _run_fault_case(
            config,
            "llm_interrupt",
            _ControlledProductionHooks(),
            provider=provider_timeout,
            expect_interruption=True,
        ),
        _run_fault_case(
            config,
            "llm_schema_failure",
            _ControlledProductionHooks(),
            provider=provider_schema,
            expect_interruption=True,
        ),
    ]
    security = _security_probes(config)
    security_case = {
        "name": "protected_patch_and_workspace_escape",
        "passed": all(security.values()),
        "acceptance": security,
        "max_repair_number": 0,
        "snapshot": {},
    }
    cases.append(security_case)
    result = _suite_result("R2", cases, started)
    result.update(
        {
            "r1": integration,
            "r1_passed": integration["all_cases_passed"],
            "all_cases_passed": bool(integration["all_cases_passed"])
            and bool(result["all_cases_passed"]),
            "provider_timeout_injected_once": provider_timeout.faults_injected == 1,
            "provider_schema_failure_injected_once": provider_schema.faults_injected == 1,
        }
    )
    return result


def run_production_rehearsal(
    level: str,
    output_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    timeout_seconds: int = 1200,
) -> dict[str, object]:
    """Dispatch the bounded offline production-control rehearsal levels."""

    normalized = level.upper()
    if normalized == "R1":
        return run_r1(data_dir, output_dir, timeout_seconds=timeout_seconds)
    if normalized == "R2":
        return run_r2(data_dir, output_dir, timeout_seconds=timeout_seconds)
    raise ValueError(f"production rehearsal level must be R1 or R2, got {level!r}")
