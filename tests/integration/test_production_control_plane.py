from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rex.agents.provider import ProviderResponse
from rex.agents.search_policy import METHOD_CARD_VERSION
from rex.contracts import ArtifactRef, AttemptStatus, ExperimentProposal, Metrics
from rex.control.production_supervisor import (
    BaselineGateResult,
    CandidatePreparation,
    ComparisonObservation,
    ProductionAutopilot,
    ProductionContext,
    ProductionFixedProvider,
    ProductionRungFailure,
    ProductionRungResult,
    RepairRequest,
    RungRequest,
    ProductionRunConfig,
    environment_provenance_sha256,
)
from rex.control.budget import deadline_epoch_ms
from rex.contracts import ExperimentState, RunState
from rex.data.manifest import sha256_file
from rex.execution.artifacts import artifact_ref
from rex.store.db import Database
from rex.store.repository import ExperimentRepository


HASH = "0" * 64


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _metric(primary: float, *, split: str, fold: str | None) -> Metrics:
    return Metrics(
        GAUC=primary,
        **{"nDCG@5": primary},
        primary=primary,
        users=4,
        rows=8,
        evaluator_sha256=HASH,
        split=split,
        fold=fold,
        seed=0,
    )


def _proposal(experiment_id: str) -> ExperimentProposal:
    return ExperimentProposal(
        experiment_id=experiment_id,
        parent_id=None,
        operator="FEATURE",
        hypothesis="Durable full-fold evidence supports a controlled mechanism dependency.",
        mechanism="The prerequisite derives from completed temporal gates, not global promotion.",
        primary_change="controlled mechanism",
        files_to_change=["e01.yaml"],
        expected_metric_effects={"primary": "increase"},
        falsifier="The full temporal gate rejects this branch.",
        leakage_analysis="Only prior training state is available.",
        estimated_seconds=1,
        cheap_rung={"fold": "A"},
        full_rung={"folds": ["A", "B", "C"]},
    )


class DiagnosisProvider:
    def generate(self, *, role: str, system: str, prompt: str, schema: dict[str, Any]):
        del system, schema
        assert role == "diagnosis"
        context = json.loads(prompt)
        assert context["evidence_binding_required"] is True
        assert context["metric_component_summary"]
        assert context["segment_diagnostics"][0]["artifact_id"] in context["artifact_ids"]
        assert context["prediction_correlations"][0]["prediction_correlation"] == 0.75
        assert context["falsification_criteria"]["full_min_positive_temporal_folds"] == 2
        return ProviderResponse(
            value={
                "experiment_id": context["experiment_id"],
                "outcome": "supported",
                "evidence_artifact_ids": context["artifact_ids"],
                "metric_deltas": {"primary": 0.003},
                "uncertainty": "fixture adapter only",
                "next_operator": "ABANDON",
                "reusable_lesson": "The trusted temporal gate accepted the controlled evidence.",
            },
            provider="fixed",
            model="production-control-test",
        )


class ScriptedHooks:
    def __init__(self, failures: dict[str, int] | None = None):
        self.failures = dict(failures or {})
        self.repairs: list[RepairRequest] = []
        self.proposal_contexts: list[dict[str, Any]] = []

    @staticmethod
    def _artifact(context: ProductionContext, name: str, content: str) -> ArtifactRef:
        path = context.run_dir / "hook-evidence" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return artifact_ref(path, "hook_evidence")

    def verify_baseline(self, context: ProductionContext) -> BaselineGateResult:
        ref = self._artifact(context, "baseline.json", "baseline")
        return BaselineGateResult(True, _metric(0.5, split="valid", fold=None), (ref,))

    def prepare_candidate(self, context, card, binding, proposal_context, parent_commit):
        assert proposal_context["method_card"]["citation_id"] == (
            f"method-card:{METHOD_CARD_VERSION}:E01"
        )
        assert proposal_context["allowed_files"] == [
            "e01.yaml",
            "src/rex/models/experimental/pair_rank_fm.py",
        ]
        assert "src/rex/models/rank_fm.py" in proposal_context["read_only_context_files"]
        assert "src/rex/data/views.py" in proposal_context["read_only_context_files"]
        assert proposal_context["method_sources"]
        assert proposal_context["falsification_criteria"]["cheap_min_primary_delta"] == 0.001
        assert proposal_context["resource_estimate"]["maximum_candidate_seconds"] == 150
        self.proposal_contexts.append(proposal_context)
        durable = proposal_context.get("durable_proposal")
        if durable is not None:
            proposal = ExperimentProposal.model_validate(durable)
        else:
            proposal = ExperimentProposal(
                experiment_id="production-e01",
                parent_id="baseline",
                operator="LOSS",
                hypothesis="Pairwise fixture evidence should improve controlled ranking.",
                mechanism="Same-user pairs align the controlled gradients with ranking.",
                primary_change="pairwise loss",
                files_to_change=["e01.yaml"],
                expected_metric_effects={"primary": "increase"},
                falsifier="Cheap primary delta is below one thousandth.",
                leakage_analysis="Only generated prior-training labels are used.",
                estimated_seconds=1,
                cheap_rung={"fold": "A"},
                full_rung={"folds": ["A", "B", "C"]},
            )
        ref = self._artifact(context, "patch.diff", "controlled patch evidence")
        return CandidatePreparation(proposal, parent_commit, artifacts=(ref,))

    def run_rung(self, request: RungRequest) -> ProductionRungResult:
        remaining = self.failures.get(request.rung, 0)
        if remaining:
            self.failures[request.rung] = remaining - 1
            raise ProductionRungFailure(AttemptStatus.NAN, f"controlled {request.rung} NaN")
        if request.rung == "cheap":
            observations = (
                ComparisonObservation(
                    _metric(0.502, split="shadow", fold="cheap"),
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
        ref = self._artifact(
            request.context,
            f"{request.rung}-{len(self.repairs)}.json",
            request.rung,
        )
        return ProductionRungResult(
            observations,
            (ref,),
            diagnostics={
                "prediction_correlation": 0.75,
                "segment_primary_deltas": {"user:warm": 0.003},
                "segment_wins": ["user:warm"],
                "segment_regressions": [],
            },
        )

    def repair_candidate(self, request: RepairRequest) -> tuple[ArtifactRef, ...]:
        self.repairs.append(request)
        return (
            self._artifact(
                request.context,
                f"repair-{request.plan.repair_number}.json",
                request.plan.action,
            ),
        )


class WeakFullHooks(ScriptedHooks):
    def run_rung(self, request: RungRequest) -> ProductionRungResult:
        result = super().run_rung(request)
        if request.rung != "full":
            return result
        observations = tuple(
            ComparisonObservation(
                _metric(0.499 - index * 0.0001, split="shadow", fold=fold),
                _metric(0.5, split="shadow", fold=fold),
            )
            for index, fold in enumerate(("A", "B", "C"))
        )
        return ProductionRungResult(observations, result.artifacts, result.diagnostics)


class RepairRevisionHooks(ScriptedHooks):
    def repair_candidate(self, request: RepairRequest) -> tuple[ArtifactRef, ...]:
        self.repairs.append(request)
        directory = (
            request.context.run_dir
            / "evidence"
            / str(request.experiment["experiment_id"])
            / "repairs"
        )
        directory.mkdir(parents=True, exist_ok=True)
        repaired = directory / f"effective-config-repair-{request.plan.repair_number}.yaml"
        repaired.write_text("plugin: fixture\nbatch_size: 512\n", encoding="utf-8")
        summary = directory / f"repair-{request.plan.repair_number}-{request.phase}.json"
        summary.write_text(
            json.dumps(
                {
                    "repair_number": request.plan.repair_number,
                    "phase": request.phase,
                    "repaired_config_path": str(repaired.resolve()),
                    "repaired_config_sha256": sha256_file(repaired),
                }
            ),
            encoding="utf-8",
        )
        return (
            artifact_ref(repaired, "repaired_experiment_config"),
            artifact_ref(summary, "repair_override"),
        )


class InjectedLifecycleKill(BaseException):
    pass


class BaselineBestHooks(ScriptedHooks):
    def verify_baseline(self, context: ProductionContext) -> BaselineGateResult:
        directory = context.run_dir / "baseline/evidence/seed-0"
        directory.mkdir(parents=True, exist_ok=True)
        model = directory / "model.npz"
        model.write_bytes(b"validated baseline checkpoint")
        encoder = directory / "encoder.json"
        encoder.write_text('{"version": 1}\n', encoding="utf-8")
        selected_config = directory / "config.json"
        selected_config.write_text('{"model": "fm", "seed": 0}\n', encoding="utf-8")
        bundle = directory / "model_bundle.json"
        bundle.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "plugin": "rex.models.official_fm:OfficialFMPlugin",
                    "seed": 0,
                    "commit_sha": context.root_commit,
                    "config_sha256": sha256_file(selected_config),
                    "data_view_sha256": "a" * 64,
                    "primary_member": "model.npz",
                    "feature_schema": {},
                    "members": [
                        {
                            "name": "model.npz",
                            "kind": "checkpoint",
                            "sha256": sha256_file(model),
                            "size_bytes": model.stat().st_size,
                        },
                        {
                            "name": "encoder.json",
                            "kind": "checkpoint_sidecar",
                            "sha256": sha256_file(encoder),
                            "size_bytes": encoder.stat().st_size,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        predictions = directory / "predictions.npz"
        predictions.write_bytes(b"validated baseline predictions")
        model_ref = artifact_ref(bundle, "baseline_incumbent_model_bundle")
        prediction_ref = artifact_ref(predictions, "baseline_incumbent_predictions")
        config_ref = artifact_ref(selected_config, "baseline_incumbent_config")
        metrics = _metric(0.5, split="valid", fold=None)
        gate = context.run_dir / "baseline/gate.json"
        gate.parent.mkdir(parents=True, exist_ok=True)
        gate.write_text(
            json.dumps(
                {
                    "accepted": True,
                    "test_scored": False,
                    "selected_seed": {
                        "seed": 0,
                        "metrics": metrics.model_dump(mode="json", by_alias=True),
                        "prediction_artifact_id": prediction_ref.artifact_id,
                        "prediction_path": str(predictions.resolve()),
                        "prediction_sha256": sha256_file(predictions),
                        "model_bundle_artifact_id": model_ref.artifact_id,
                        "model_bundle_path": str(bundle.resolve()),
                        "model_bundle_sha256": sha256_file(bundle),
                        "config_artifact_id": config_ref.artifact_id,
                        "config_path": str(selected_config.resolve()),
                        "config_sha256": sha256_file(selected_config),
                    },
                }
            ),
            encoding="utf-8",
        )
        return BaselineGateResult(
            True,
            metrics,
            (
                artifact_ref(model, "baseline_incumbent_checkpoint"),
                model_ref,
                prediction_ref,
                config_ref,
                artifact_ref(gate, "baseline_gate"),
            ),
        )

    def run_rung(self, request: RungRequest) -> ProductionRungResult:
        assert request.rung == "cheap"
        metrics = _metric(0.5, split="shadow", fold="cheap")
        return ProductionRungResult(
            (ComparisonObservation(metrics, metrics),),
            (self._artifact(request.context, "baseline-losing-cheap.json", "rejected"),),
        )


def _config(tmp_path: Path) -> ProductionRunConfig:
    project = tmp_path / "project"
    project.mkdir()
    files = {
        "requirements-lock.txt": "locked\n",
        "pyproject.toml": "[project]\nname='control-test'\nversion='0.0.0'\n",
        "data-manifest.json": "{}\n",
        "evaluate.py": "# fixture evaluator\n",
        "e01.yaml": "plugin: fixture\n",
        "e10.yaml": "plugin: fixture-ensemble\n",
    }
    for name, value in files.items():
        (project / name).write_text(value, encoding="utf-8")
    _git(project, "init")
    _git(project, "config", "user.email", "rex@example.invalid")
    _git(project, "config", "user.name", "REX Production Test")
    _git(project, "add", "--all")
    _git(project, "commit", "-m", "control plane fixture")
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        "max_hypotheses: 1\n"
        "max_official_evaluations: 50\n"
        "wall_clock_seconds: 300\n"
        "finalization_reserve_seconds: 0\n"
        "convergence_epsilon: 0.002\n"
        "convergence_patience: 3\n"
        "max_repairs_per_experiment: 2\n"
        "default_attempt_timeout_seconds: 30\n",
        encoding="utf-8",
    )
    config = tmp_path / "production.yaml"
    config.write_text(
        "execution_mode: production\n"
        "scientific_execution_enabled: true\n"
        f"project_root: {project}\n"
        f"runs_dir: {tmp_path / 'runs'}\n"
        f"budget_config: {budget}\n"
        f"protected_paths: {project / 'protected.yaml'}\n"
        f"data_manifest: {project / 'data-manifest.json'}\n"
        f"evaluator_path: {project / 'evaluate.py'}\n"
        f"environment_lock: {project / 'requirements-lock.txt'}\n"
        "cleanup_worktrees: true\n"
        "process_stale_after_seconds: 60\n"
        "method_cards:\n"
        "  E01:\n"
        f"    config: {project / 'e01.yaml'}\n"
        "    feature_recipe: control\n"
        "  E10:\n"
        f"    config: {project / 'e10.yaml'}\n"
        "    feature_recipe: control\n"
        "llm:\n"
        "  mode: fixed\n",
        encoding="utf-8",
    )
    return ProductionRunConfig.load(config)


def test_production_control_plane_promotes_validation_best_without_submission(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = ProductionAutopilot(config, DiagnosisProvider(), ScriptedHooks()).run(
        run_id="production-happy"
    )
    assert result["state"] == "COMPLETE"
    assert result["search_champion_experiment_id"] == "production-e01"
    assert result["official_evaluation_count"] == 1
    assert result["confirmation_enabled"] is False
    assert result["final_submission_enabled"] is False
    assert result["best_valid_bundle"]["deferred"]
    assert result["experiments"][0]["state"] == "PROMOTED"
    snapshot = config.runs_dir / "production-happy/evidence/production-e01/effective-config.yaml"
    assert snapshot.read_text(encoding="utf-8") == "plugin: fixture\n"


def test_full_gate_rejection_is_diagnosed_before_terminal_rejection(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = ProductionAutopilot(config, DiagnosisProvider(), WeakFullHooks()).run(
        run_id="production-full-rejected"
    )

    assert result["state"] == "COMPLETE"
    assert result["search_champion_experiment_id"] == "baseline"
    assert result["official_evaluation_count"] == 0
    assert result["experiments"][0]["state"] == "REJECTED"
    diagnosis = config.runs_dir / "production-full-rejected/evidence/production-e01/diagnosis.json"
    assert diagnosis.is_file()


def test_global_repair_limit_stops_third_failure_and_preserves_baseline(tmp_path: Path) -> None:
    config = _config(tmp_path)
    hooks = ScriptedHooks({"cheap": 1, "full": 1, "official_valid": 1})
    result = ProductionAutopilot(config, DiagnosisProvider(), hooks).run(
        run_id="production-repair-cap"
    )
    assert result["state"] == "COMPLETE"
    assert result["search_champion_experiment_id"] == "baseline"
    assert result["experiments"][0]["state"] == "FAILED_FINAL"
    assert result["non_improvement_streak"] == 0
    assert [item["repair_number"] for item in result["repairs"]] == [1, 2]
    assert [item.plan.repair_number for item in hooks.repairs] == [1, 2]


def test_resume_after_registered_repair_evidence_applies_one_exact_revision(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    hooks = RepairRevisionHooks({"cheap": 1})
    killed = False

    def checkpoint(stage: str, experiment_id: str) -> None:
        nonlocal killed
        assert experiment_id == "production-e01"
        if stage == "repair_evidence_registered" and not killed:
            killed = True
            raise InjectedLifecycleKill("kill between repair evidence and revision")

    with pytest.raises(InjectedLifecycleKill):
        ProductionAutopilot(
            config,
            DiagnosisProvider(),
            hooks,
            lifecycle_checkpoint=checkpoint,
        ).run(run_id="repair-revision-resume")

    database = Database(config.runs_dir / "repair-revision-resume/state.sqlite3")
    repository = ExperimentRepository(database)
    interrupted = repository.get_experiment("production-e01")
    original_config_sha256 = sha256_file(config.method_cards["E01"].config_path)
    assert interrupted["state"] == ExperimentState.REPAIRING
    assert interrupted["config_sha256"] == original_config_sha256
    assert len(hooks.repairs) == 1

    result = ProductionAutopilot(config, DiagnosisProvider(), hooks).run(
        run_id="repair-revision-resume"
    )

    repaired_path = (
        config.runs_dir
        / "repair-revision-resume/evidence/production-e01/repairs/effective-config-repair-1.yaml"
    )
    experiment = repository.get_experiment("production-e01")
    assert result["experiments"][0]["state"] == "PROMOTED"
    assert experiment["commit_sha"] == interrupted["commit_sha"]
    assert experiment["config_sha256"] == sha256_file(repaired_path)
    assert len(hooks.repairs) == 1
    assert result["hypothesis_count"] == 1
    with database.connect() as connection:
        revisions = connection.execute(
            "SELECT previous_commit_sha,repaired_commit_sha,previous_config_sha256,"
            "repaired_config_sha256,effective_config_artifact_id FROM experiment_repairs"
        ).fetchall()
        revision_events = connection.execute(
            "SELECT COUNT(*) FROM event_outbox WHERE event_type='repair.revision_applied'"
        ).fetchone()[0]
    assert len(revisions) == 1
    assert revisions[0]["previous_commit_sha"] == interrupted["commit_sha"]
    assert revisions[0]["repaired_commit_sha"] == interrupted["commit_sha"]
    assert revisions[0]["previous_config_sha256"] == original_config_sha256
    assert revisions[0]["repaired_config_sha256"] == sha256_file(repaired_path)
    assert revisions[0]["effective_config_artifact_id"]
    assert revision_events == 1


def test_baseline_champion_is_frozen_as_best_valid_without_experiment_link(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = ProductionAutopilot(
        config,
        DiagnosisProvider(),
        BaselineBestHooks(),
    ).run(run_id="baseline-best-valid")

    manifest_path = Path(result["best_valid_bundle"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["search_champion_experiment_id"] == "baseline"
    assert manifest["incumbent_experiment_id"] == "baseline"
    assert manifest["commit_sha"] == _git(config.project_root, "rev-parse", "HEAD")
    assert manifest["test_prediction_created"] is False
    assert (manifest_path.parent / "model/model_bundle.json").is_file()
    assert (manifest_path.parent / "model/model.npz").is_file()
    assert (manifest_path.parent / "valid_predictions.npz").is_file()
    assert (manifest_path.parent / "config.json").is_file()
    database = Database(config.runs_dir / "baseline-best-valid/state.sqlite3")
    with database.connect() as connection:
        linked = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE kind='best_valid_manifest'"
        ).fetchone()[0]
        experiment_links = connection.execute(
            "SELECT COUNT(*) FROM artifact_links link JOIN artifacts artifact "
            "ON artifact.artifact_id=link.artifact_id "
            "WHERE artifact.kind='best_valid_manifest' AND link.experiment_id IS NOT NULL"
        ).fetchone()[0]
    assert linked == 1
    assert experiment_links == 0


def test_resume_rebuilds_same_durable_proposal_instead_of_consuming_card(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_id = "production-resume-preparation"
    run_dir = config.runs_dir / run_id
    database = Database(run_dir / "state.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    root_commit = _git(config.project_root, "rev-parse", "HEAD")
    repository.create_run(
        run_id=run_id,
        deadline_epoch_ms=deadline_epoch_ms(300),
        root_commit=root_commit,
        environment_sha256=environment_provenance_sha256(config),
        data_manifest_sha256=sha256_file(config.data_manifest),
        evaluator_sha256=sha256_file(config.evaluator_path),
    )
    repository.transition_run(run_id, RunState.INITIALIZING, RunState.BASELINE_VERIFYING)
    repository.establish_baseline(
        run_id=run_id,
        metrics=_metric(0.5, split="valid", fold=None),
        evidence_artifact_ids=[],
    )
    repository.transition_run(run_id, RunState.BASELINE_VERIFYING, RunState.SEARCHING)
    proposal = (
        ScriptedHooks()
        .prepare_candidate(
            ProductionContext(
                run_id, run_dir, config.project_root, root_commit, deadline_epoch_ms(300)
            ),
            next(
                card
                for card in ProductionAutopilot(config, DiagnosisProvider()).search_policy.cards
                if card.card_id == "E01"
            ),
            config.method_cards["E01"],
            {
                "method_card": {"citation_id": f"method-card:{METHOD_CARD_VERSION}:E01"},
                "allowed_files": ["e01.yaml", "src/rex/models/experimental/pair_rank_fm.py"],
                "read_only_context_files": [
                    "src/rex/models/rank_fm.py",
                    "src/rex/data/views.py",
                ],
                "method_sources": [{"source_id": "ranknet"}],
                "falsification_criteria": {"cheap_min_primary_delta": 0.001},
                "resource_estimate": {"maximum_candidate_seconds": 150},
            },
            root_commit,
        )
        .proposal.model_copy(update={"parent_id": None})
    )
    repository.create_experiment(
        run_id,
        proposal,
        root_commit,
        workspace_path=None,
        branch_name=None,
        commit_sha=root_commit,
        config_sha256=sha256_file(config.method_cards["E01"].config_path),
        method_card_id="E01",
        experiment_kind="production_search",
    )
    repository.transition_experiment(
        proposal.experiment_id,
        ExperimentState.PROPOSED,
        ExperimentState.WORKTREE_READY,
        idempotency_key=f"{proposal.experiment_id}:worktree-ready",
    )

    hooks = ScriptedHooks()
    result = ProductionAutopilot(config, DiagnosisProvider(), hooks).run(run_id=run_id)

    assert result["experiments"][0]["state"] == "PROMOTED"
    assert result["hypothesis_count"] == 1
    assert hooks.proposal_contexts[0]["resume_experiment_id"] == proposal.experiment_id
    assert hooks.proposal_contexts[0]["durable_proposal"]["experiment_id"] == proposal.experiment_id


def test_method_dependencies_use_supported_full_evidence_not_only_global_promotions(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    database = Database(tmp_path / "dependency.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    repository.create_run(
        run_id="dependencies",
        deadline_epoch_ms=deadline_epoch_ms(300),
        root_commit="root",
        environment_sha256=HASH,
        data_manifest_sha256=HASH,
        evaluator_sha256=HASH,
    )
    diagnosed_chain = (
        ExperimentState.WORKTREE_READY,
        ExperimentState.PATCHED,
        ExperimentState.STATIC_VALID,
        ExperimentState.FIXTURE_VALID,
        ExperimentState.CHEAP_RUNNING,
        ExperimentState.CHEAP_COMPLETE,
        ExperimentState.FULL_RESERVED,
        ExperimentState.FULL_RUNNING,
        ExperimentState.FULL_COMPLETE,
        ExperimentState.DIAGNOSED,
        ExperimentState.REJECTED,
    )
    for card_id in ("E01", "E02"):
        experiment_id = f"supported-{card_id.lower()}"
        repository.create_experiment(
            "dependencies",
            _proposal(experiment_id),
            "root",
            method_card_id=card_id,
        )
        current = ExperimentState.PROPOSED
        for index, target in enumerate(diagnosed_chain):
            repository.transition_experiment(
                experiment_id,
                current,
                target,
                idempotency_key=f"{experiment_id}:{index}",
            )
            current = target
    for card_id in ("E03", "E04", "E05", "E06", "E07", "E08"):
        experiment_id = f"attempted-{card_id.lower()}"
        repository.create_experiment(
            "dependencies",
            _proposal(experiment_id),
            "root",
            method_card_id=card_id,
        )
        repository.transition_experiment(
            experiment_id,
            ExperimentState.PROPOSED,
            ExperimentState.REJECTED,
            idempotency_key=f"{experiment_id}:rejected",
        )

    card = ProductionAutopilot(config, DiagnosisProvider())._next_card(repository, "dependencies")

    assert card is not None
    assert card.card_id == "E10"


def test_fixed_provider_uses_truthful_method_card_operators() -> None:
    provider = ProductionFixedProvider()
    expected = {
        "E01": "LOSS",
        "E03": "FEATURE",
        "E10": "ENSEMBLE",
        "E15": "ENSEMBLE",
    }
    for card_id, operator in expected.items():
        response = provider.generate(
            role="proposal",
            system="fixture",
            prompt=json.dumps(
                {
                    "experiment_id": f"fixed-{card_id.lower()}",
                    "method_card": {
                        "card_id": card_id,
                        "mechanism": "controlled fixture mechanism",
                    },
                    "allowed_files": [f"{card_id.lower()}.yaml"],
                    "incumbent": {"experiment_id": "baseline"},
                }
            ),
            schema={},
        )
        assert response.value["operator"] == operator
