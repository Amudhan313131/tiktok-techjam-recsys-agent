"""Short deterministic rehearsals for contracts, persistence, and recovery plumbing."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from rex.agents.patch_guard import PatchPolicy, PatchRejected, validate_patch
from rex.agents.provider import (
    FakeProvider,
    ProviderResponse,
    ProviderRouter,
    ProviderTimeoutError,
    StructuredProvider,
)
from rex.agents.services import ProposalService
from rex.contracts import (
    AttemptStatus,
    ExperimentProposal,
    ExperimentState,
    Operator,
    RunRequest,
    RunState,
)
from rex.control.budget import deadline_epoch_ms
from rex.control.fixture_supervisor import FixtureAutopilot, FixtureRunConfig, FixtureScriptProvider
from rex.data.bootstrap import bootstrap_views
from rex.data.manifest import sha256_file, verify_starter_manifest
from rex.evaluation.official_adapter import evaluate_predictions
from rex.evaluation.submission import build_submission, validate_submission
from rex.execution.runner import execute_request
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
        return {"data": True, "live_llm": False, "target_minutes": 60}
    if normalized == "R2":
        return {"data": True, "live_llm": True, "target_minutes": 90}
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


def _request(
    *,
    config_path: Path,
    feature_path: Path,
    target_path: Path | None,
    output_dir: Path,
    rung: str,
    split: str,
    timeout_seconds: int,
) -> RunRequest:
    return RunRequest(
        run_id="rehearsal",
        experiment_id="r1-pair-loss",
        attempt_id=f"r1-{rung}-{uuid.uuid4().hex[:8]}",
        commit_sha="rehearsal",
        plugin="rex.models.rank_fm:RankFMPlugin",
        config_path=str(config_path),
        config_sha256=sha256_file(config_path),
        seed=0,
        rung=rung,
        split=split,
        feature_view_path=str(feature_path),
        target_view_path=str(target_path) if target_path else None,
        output_dir=str(output_dir),
        deadline_epoch_ms=int((time.time() + timeout_seconds + 60) * 1000),
        timeout_seconds=timeout_seconds,
        data_view_sha256=sha256_file(feature_path),
        environment_sha256="0" * 64,
    )


def run_r1(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    timeout_seconds: int = 1200,
) -> dict[str, object]:
    """Real-data integration rehearsal with a bounded pairwise candidate and CSV fault."""
    raise RuntimeError(
        "R1 real-data model experimentation is explicitly deferred in the current phase"
    )
    output = Path(output_dir)
    views = output / "views"
    manifest = bootstrap_views(data_dir, views)
    train_config = output / "pair-train.json"
    train_config.write_text(
        json.dumps(
            {
                "k": 16,
                "lr": 0.001,
                "epochs": 1,
                "pair_batch_size": 4096,
                "negatives_per_positive": 2,
                "max_pairs": 50_000,
                "bce_weight": 0.02,
            }
        ),
        encoding="utf-8",
    )
    train_result = execute_request(
        _request(
            config_path=train_config,
            feature_path=views / "train_features.npz",
            target_path=views / "label_vault/train_targets.npz",
            output_dir=output / "train",
            rung="cheap",
            split="train",
            timeout_seconds=timeout_seconds,
        ),
        output / "attempt-train",
    )
    if train_result.status != AttemptStatus.SUCCESS:
        raise RuntimeError(f"R1 pairwise training failed: {train_result.error_summary}")
    checkpoint = next(item for item in train_result.artifacts if item.kind == "checkpoint")
    predict_config = output / "pair-predict.json"
    predict_config.write_text(
        json.dumps({"model_artifact_path": checkpoint.path}), encoding="utf-8"
    )
    predict_result = execute_request(
        _request(
            config_path=predict_config,
            feature_path=views / "valid_features.npz",
            target_path=None,
            output_dir=output / "predict",
            rung="predict",
            split="valid",
            timeout_seconds=timeout_seconds,
        ),
        output / "attempt-predict",
    )
    if predict_result.status != AttemptStatus.SUCCESS:
        raise RuntimeError(f"R1 prediction failed: {predict_result.error_summary}")
    predictions = next(item for item in predict_result.artifacts if item.kind == "predictions")
    metrics = evaluate_predictions(
        views / "valid_features.npz",
        views / "label_vault/valid_targets.npz",
        predictions.path,
        split="valid",
        seed=0,
    )
    submission = build_submission(predictions.path, output / "valid_submission.csv")
    valid_check = validate_submission(submission, data_dir=data_dir, split="valid")
    invalid = output / "invalid_submission.csv"
    lines = submission.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[1] = "definitely-misaligned-user"
    lines[1] = ",".join(fields)
    invalid.write_text("\n".join(lines) + "\n", encoding="utf-8")
    invalid_check = validate_submission(invalid, data_dir=data_dir, split="valid")
    resume = run_r0(output / "resume")
    return {
        "level": "R1",
        "data_manifest_sha256": manifest["manifest_sha256"],
        "train_status": train_result.status,
        "predict_status": predict_result.status,
        "metrics": metrics.model_dump(mode="json", by_alias=True),
        "valid_submission_accepted": valid_check.valid,
        "misaligned_submission_rejected": not invalid_check.valid,
        "resume_event_chain_valid": resume["event_chain_valid"],
    }


def run_r2(data_dir: str | Path, output_dir: str | Path, *, timeout_seconds: int = 1200) -> dict[str, object]:
    """Autonomy rehearsal: R1 plus schema-constrained offline proposal replay."""
    raise RuntimeError(
        "R2 scientific autonomy experimentation is explicitly deferred in the current phase"
    )
    integration = run_r1(data_dir, output_dir, timeout_seconds=timeout_seconds)
    payload = {
        "experiment_id": "r2-agent-patch",
        "parent_id": "r1-pair-loss",
        "operator": "LOSS",
        "hypothesis": "Delta nDCG weighting should focus pair gradients near rank five.",
        "mechanism": "Pair weights approximate the official cutoff-specific swap delta.",
        "primary_change": "delta nDCG pair weighting",
        "files_to_change": ["src/rex/losses/experimental/delta_weight.py"],
        "expected_metric_effects": {"nDCG@5": "increase"},
        "falsifier": "No positive delta on two of three shadow folds.",
        "leakage_analysis": "Weights use train labels and current train predictions only.",
        "estimated_seconds": 120,
        "cheap_rung": {"fold": "A"},
        "full_rung": {"folds": ["A", "B", "C"]},
    }
    decision = ProposalService(FakeProvider([payload])).propose(
        {"artifact_ids": [], "prior_metrics": integration["metrics"]}
    )
    return {
        **integration,
        "level": "R2",
        "offline_proposal_valid": decision.parsed.experiment_id == "r2-agent-patch",
        "accepted_patch_required_for_public_claim": True,
    }
