from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rex.agents.search_policy import SearchPolicy
from rex.agents.recovery import RepairAction, TypedRepairPlan
from rex.contracts import ArtifactRef, AttemptStatus, ExperimentProposal, Metrics, RunResult
from rex.control.production_supervisor import (
    MethodCardBinding,
    ProductionContext,
    ProductionFixedProvider,
    ProductionRungFailure,
    ProductionRunConfig,
    RepairRequest,
    RungRequest,
)
from rex.control.scientific_hooks import ProductionScientificHooks, ScientificExecutionSettings
from rex.data.manifest import sha256_file
from rex.data.shadow_views import materialize_shadow_folds
from rex.data.views import FeatureView, TargetView, load_feature_view
from rex.execution.artifacts import artifact_ref, write_prediction_artifact
from rex.evaluation.baseline import (
    BaselineAcceptance,
    BaselineBundle,
    BaselineEvidence,
    BaselineSeedResult,
)
from rex.models.bundle import create_model_bundle, validate_model_bundle
from rex.models.shadow_blend import ShadowBlendPlugin
from rex.store.db import Database
from rex.store.repository import ExperimentRepository


HASH = "0" * 64


def test_tree_coder_receives_real_plugin_and_feature_interfaces(tmp_path: Path) -> None:
    del tmp_path
    root = Path(__file__).resolve().parents[2]
    config = ProductionRunConfig.load(root / "configs/run/production.yaml")
    hooks = ProductionScientificHooks(config, ProductionFixedProvider())
    snapshots = hooks._read_only_context_snapshots(
        {
            "read_only_context_files": [
                "src/rex/models/tree_ranker.py",
                "src/rex/features/recipes.py",
                "src/rex/data/views.py",
            ]
        }
    )

    assert "train_features: FeatureView" in snapshots["src/rex/models/tree_ranker.py"]
    assert "materialize_feature_recipe" in snapshots["src/rex/features/recipes.py"]
    assert "class FeatureView" in snapshots["src/rex/data/views.py"]


def _write_view(path: Path, dates: list[int]) -> tuple[Path, Path]:
    rows: list[tuple[int, str, str]] = []
    for date in dates:
        for user in ("u0", "u1", "u2", "u3"):
            rows.extend(((date, user, "negative"), (date, user, "positive")))
    features = path / "features.npz"
    targets = path / "targets.npz"
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        features,
        row_id=np.arange(len(rows), dtype=np.int64),
        date=np.asarray([row[0] for row in rows], dtype=np.int32),
        user_id=np.asarray([row[1] for row in rows]),
        video_id=np.asarray([row[2] for row in rows]),
        author_id=np.asarray(["author" for _ in rows]),
        tab=np.asarray(["1" for _ in rows]),
        duration_ms=np.asarray([1000.0 for _ in rows], dtype=np.float32),
    )
    np.savez_compressed(
        targets,
        long_view=np.asarray([row[2] == "positive" for row in rows], dtype=np.float32),
    )
    return features, targets


def _config(tmp_path: Path) -> tuple[ProductionRunConfig, MethodCardBinding, Path, Path]:
    project = tmp_path / "project"
    config_root = project / "configs/experiments"
    config_root.mkdir(parents=True)
    candidate = config_root / "e01_pair_rank_fm.yaml"
    reference = config_root / "e00_official_fm.yaml"
    pointwise_control = config_root / "e01_pointwise_control.yaml"
    candidate.write_text(
        "plugin: fake:Plugin\nepochs: 1\npair_batch_size: 4096\nmax_pairs: 500000\n",
        encoding="utf-8",
    )
    reference.write_text("plugin: fake:Plugin\nepochs: 1\n", encoding="utf-8")
    pointwise_control.write_text(
        "plugin: fake:Plugin\nepochs: 1\nbatch_size: 4096\n",
        encoding="utf-8",
    )
    budget = project / "budget.yaml"
    budget.write_text(
        "max_hypotheses: 50\n"
        "max_official_evaluations: 50\n"
        "wall_clock_seconds: 1000\n"
        "finalization_reserve_seconds: 0\n"
        "convergence_epsilon: 0.002\n"
        "convergence_patience: 3\n"
        "max_repairs_per_experiment: 2\n"
        "default_attempt_timeout_seconds: 30\n",
        encoding="utf-8",
    )
    train_features, train_targets = _write_view(
        tmp_path / "views/train", list(range(20220408, 20220422))
    )
    valid_features, valid_targets = _write_view(tmp_path / "views/valid", [20220422, 20220423])
    test_features, _ = _write_view(tmp_path / "views/test-source", [20220501])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "raw_dataset_identity_sha256": HASH,
                "splits": {
                    "train": {
                        "feature_path": str(train_features),
                        "feature_sha256": sha256_file(train_features),
                        "target_path": str(train_targets),
                        "target_sha256": sha256_file(train_targets),
                    },
                    "valid": {
                        "feature_path": str(valid_features),
                        "feature_sha256": sha256_file(valid_features),
                        "target_path": str(valid_targets),
                        "target_sha256": sha256_file(valid_targets),
                    },
                    "test": {
                        "feature_path": str(test_features),
                        "feature_sha256": sha256_file(test_features),
                        "target_path": None,
                        "target_sha256": None,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    evaluator = project / "evaluate.py"
    evaluator.write_text("# protected\n", encoding="utf-8")
    lock = project / "requirements-lock.txt"
    lock.write_text("numpy\n", encoding="utf-8")
    protected = project / "protected.yaml"
    protected.write_text("allow: []\ndeny: []\n", encoding="utf-8")
    production = project / "production.yaml"
    production.write_text("execution_mode: production\n", encoding="utf-8")
    binding = MethodCardBinding(candidate, "control")
    config = ProductionRunConfig(
        source_path=production,
        project_root=project,
        runs_dir=tmp_path / "runs",
        budget_config=budget,
        protected_paths=protected,
        data_manifest=manifest,
        evaluator_path=evaluator,
        environment_lock=lock,
        scientific_execution_enabled=True,
        process_stale_after_seconds=60,
        cleanup_worktrees=False,
        method_cards={"E01": binding},
        llm={"mode": "fixed"},
    )
    return config, binding, train_features, valid_features


class _FakeWorker:
    def __call__(self, request, attempt_dir, **kwargs):
        del attempt_dir, kwargs
        output = Path(request.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        artifacts: list[ArtifactRef]
        if request.effective_operation == "fit":
            primary = output / "model.bin"
            primary.write_bytes(request.attempt_id.encode("utf-8"))
            bundle = create_model_bundle(
                output,
                primary,
                plugin=request.plugin,
                seed=request.seed,
                commit_sha=request.commit_sha,
                config_sha256=request.config_sha256,
                data_view_sha256=request.data_view_sha256,
                features=load_feature_view(request.feature_view_path),
            )
            artifacts = [artifact_ref(bundle, "model_bundle")]
        else:
            view = load_feature_view(request.feature_view_path)
            if "candidate" in request.attempt_id:
                scores = np.asarray(
                    [1.0 if str(value) == "positive" else 0.0 for value in view.arrays["video_id"]]
                )
            else:
                scores = np.zeros(view.rows, dtype=np.float64)
            prediction = write_prediction_artifact(output / "predictions.npz", view, scores)
            artifacts = [artifact_ref(prediction, "predictions")]
        return RunResult(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            attempt_id=request.attempt_id,
            status=AttemptStatus.SUCCESS,
            exit_code=0,
            command_sha256=HASH,
            commit_sha=request.commit_sha,
            config_sha256=request.config_sha256,
            data_view_sha256=request.data_view_sha256,
            environment_sha256=request.environment_sha256,
            artifacts=artifacts,
            wall_seconds=0.01,
        )


class _CountingFakeWorker(_FakeWorker):
    def __init__(self):
        self.calls = []

    def __call__(self, request, attempt_dir, **kwargs):
        self.calls.append(request)
        return super().__call__(request, attempt_dir, **kwargs)


class _ConcurrentFakeWorker(_FakeWorker):
    def __init__(self, delay_seconds: float = 0.03):
        self.delay_seconds = delay_seconds
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.completed: list[str] = []

    def __call__(self, request, attempt_dir, **kwargs):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            delay = self.delay_seconds
            if "full-A" in request.attempt_id:
                delay *= 2
            time.sleep(delay)
            return super().__call__(request, attempt_dir, **kwargs)
        finally:
            with self.lock:
                self.completed.append(request.attempt_id)
                self.active -= 1


class _FailingConcurrentWorker(_ConcurrentFakeWorker):
    def __call__(self, request, attempt_dir, **kwargs):
        if "full-A-candidate" not in request.attempt_id or request.effective_operation != "fit":
            if "full-B" in request.attempt_id:
                time.sleep(0.08)
            return super().__call__(request, attempt_dir, **kwargs)
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.005)
            return RunResult(
                run_id=request.run_id,
                experiment_id=request.experiment_id,
                attempt_id=request.attempt_id,
                status=AttemptStatus.CRASH,
                exit_code=1,
                error_type="ControlledFailure",
                error_summary="controlled parallel failure",
                command_sha256=HASH,
                commit_sha=request.commit_sha,
                config_sha256=request.config_sha256,
                data_view_sha256=request.data_view_sha256,
                environment_sha256=request.environment_sha256,
                artifacts=[],
                wall_seconds=0.005,
            )
        finally:
            with self.lock:
                self.completed.append(request.attempt_id)
                self.active -= 1


def _request(tmp_path: Path):
    config, binding, _, _ = _config(tmp_path)
    context = ProductionContext(
        "scientific-unit",
        config.runs_dir / "scientific-unit",
        config.project_root,
        "candidate-commit",
        4_102_444_800_000,
    )
    database = Database(context.run_dir / "state.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    repository.create_run(
        run_id=context.run_id,
        deadline_epoch_ms=context.deadline_epoch_ms,
        root_commit=context.root_commit,
        environment_sha256=HASH,
        data_manifest_sha256=sha256_file(config.data_manifest),
        evaluator_sha256=sha256_file(config.evaluator_path),
    )
    proposal = ExperimentProposal(
        experiment_id="scientific-e01",
        parent_id=None,
        operator="LOSS",
        hypothesis="Pairwise loss should improve within-user recommendation ranking.",
        mechanism="Same-user positive and negative pairs directly optimize ordering.",
        primary_change="pairwise loss",
        files_to_change=["configs/experiments/e01_pair_rank_fm.yaml"],
        expected_metric_effects={"primary": "increase"},
        falsifier="Cheap primary does not improve enough.",
        leakage_analysis="Only prior shadow training targets are used.",
        estimated_seconds=30,
        cheap_rung={"fold": "A"},
        full_rung={"folds": ["A", "B", "C"]},
    )
    repository.create_experiment(
        context.run_id,
        proposal,
        context.root_commit,
        workspace_path=str(tmp_path / "worktree"),
        branch_name="codex/rex-scientific-e01",
        commit_sha=context.root_commit,
        config_sha256=sha256_file(binding.config_path),
        method_card_id="E01",
    )
    repository.register_artifact(
        artifact_ref(binding.config_path, "experiment_config"),
        experiment_id=proposal.experiment_id,
    )
    experiment = repository.get_experiment(proposal.experiment_id)
    card = next(item for item in SearchPolicy().cards if item.card_id == "E01")
    hooks = ProductionScientificHooks(
        config,
        ProductionFixedProvider(),
        execute=_FakeWorker(),
        settings=ScientificExecutionSettings(
            cheap_user_fraction=1.0,
            bootstrap_samples=20,
            max_memory_mb=128,
        ),
    )
    return hooks, context, experiment, card, binding, repository


def test_fixed_scientific_hook_runs_cheap_full_and_valid_with_durable_attempts(tmp_path: Path):
    hooks, context, experiment, card, binding, repository = _request(tmp_path)
    cheap = hooks.run_rung(RungRequest(context, experiment, card, binding, "cheap"))
    full = hooks.run_rung(RungRequest(context, experiment, card, binding, "full"))
    valid_features, valid_targets = hooks._split_paths("valid")
    assert valid_targets is not None
    valid_view = load_feature_view(valid_features)
    baseline_prediction = context.run_dir / "baseline/evidence/seed-0/predictions.npz"
    write_prediction_artifact(
        baseline_prediction,
        valid_view,
        np.zeros(valid_view.rows, dtype=np.float64),
    )
    baseline_metrics = hooks._metrics(
        valid_features,
        valid_targets,
        baseline_prediction,
        split="valid",
        fold=None,
        seed=0,
    )
    (baseline_prediction.parent / "metrics.json").write_text(
        json.dumps(baseline_metrics.model_dump(mode="json", by_alias=True)),
        encoding="utf-8",
    )
    gate_path = context.run_dir / "baseline/gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(
            {
                "metrics": baseline_metrics.model_dump(mode="json", by_alias=True),
                "selected_seed": {
                    "seed": 0,
                    "metrics": baseline_metrics.model_dump(mode="json", by_alias=True),
                    "prediction_path": str(baseline_prediction.resolve()),
                    "prediction_sha256": sha256_file(baseline_prediction),
                },
            }
        ),
        encoding="utf-8",
    )
    with repository.database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET search_champion_experiment_id='baseline' WHERE run_id=?",
            (context.run_id,),
        )
    valid = hooks.run_rung(RungRequest(context, experiment, card, binding, "official_valid"))

    assert len(cheap.observations) == 1
    assert len(full.observations) == 3
    assert len(valid.observations) == 1
    assert all(
        item.primary_delta > 0
        for item in (*cheap.observations, *full.observations, *valid.observations)
    )
    assert "prediction_correlation" in cheap.diagnostics
    assert any(item.kind == "model_bundle" for item in valid.artifacts)
    assert any(item.kind == "valid_predictions" for item in valid.artifacts)
    assert valid.observations[0].reference == baseline_metrics
    with repository.database.connect() as connection:
        attempts = connection.execute(
            "SELECT status,rung FROM attempts WHERE experiment_id=?",
            (experiment["experiment_id"],),
        ).fetchall()
        resources = connection.execute(
            "SELECT COUNT(*) FROM resource_usage WHERE experiment_id=?",
            (experiment["experiment_id"],),
        ).fetchone()[0]
    assert attempts and {row["status"] for row in attempts} == {"success"}
    assert not any(
        "official_valid" in row["rung"] and "reference" in row["rung"] for row in attempts
    )
    assert resources == len(attempts)


def test_scientific_rung_cache_replays_without_new_worker_calls(tmp_path: Path):
    hooks, context, experiment, card, binding, repository = _request(tmp_path)
    request = RungRequest(context, experiment, card, binding, "cheap")
    first = hooks.run_rung(request)
    second = hooks.run_rung(request)
    assert second.observations == first.observations
    with repository.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 4


def test_control_cache_skips_equivalent_reference_workers(tmp_path: Path):
    hooks, context, experiment, card, binding, _repository = _request(tmp_path)
    worker = _CountingFakeWorker()
    hooks.execute = worker
    hooks.config = replace(hooks.config, control_cache_dir=tmp_path / "shared-control-cache")
    request = RungRequest(context, experiment, card, binding, "cheap")

    first = hooks.run_rung(request)
    assert len(worker.calls) == 4
    rung_cache = context.run_dir / "scientific-cache" / experiment["experiment_id"] / "cheap.json"
    rung_cache.unlink()

    second = hooks.run_rung(request)

    assert second.observations == first.observations
    assert len(worker.calls) == 6
    assert [call.operation for call in worker.calls[-2:]] == ["fit", "predict"]
    assert all("candidate" in call.attempt_id for call in worker.calls[-2:])
    assert any(item.kind == "control_cache_hit" for item in second.artifacts)


def test_candidate_and_reference_execute_concurrently_within_bound(tmp_path: Path):
    hooks, context, experiment, card, binding, _repository = _request(tmp_path)
    worker = _ConcurrentFakeWorker()
    hooks.execute = worker
    hooks.settings = replace(
        hooks.settings,
        max_parallel_workers=2,
        max_parallel_folds=3,
        parallel_candidate_control=True,
    )

    result = hooks.run_rung(RungRequest(context, experiment, card, binding, "cheap"))

    assert len(result.observations) == 1
    assert worker.peak == 2


def test_full_rung_parallelism_is_bounded_and_results_are_canonical(tmp_path: Path):
    hooks, context, experiment, card, binding, repository = _request(tmp_path)
    worker = _ConcurrentFakeWorker()
    hooks.execute = worker
    hooks.settings = replace(
        hooks.settings,
        max_parallel_workers=6,
        max_parallel_folds=3,
        parallel_candidate_control=True,
    )

    result = hooks.run_rung(RungRequest(context, experiment, card, binding, "full"))

    assert [item.candidate.fold for item in result.observations] == ["A", "B", "C"]
    assert worker.peak == 6
    assert worker.completed[0].split(":full-", 1)[1][0] in {"B", "C"}
    with repository.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 12


def test_parallel_failure_drains_started_work_and_cancels_queued_folds(tmp_path: Path):
    hooks, context, experiment, card, binding, repository = _request(tmp_path)
    worker = _FailingConcurrentWorker()
    hooks.execute = worker
    hooks.settings = replace(
        hooks.settings,
        max_parallel_workers=4,
        max_parallel_folds=3,
        parallel_candidate_control=True,
    )

    with pytest.raises(ProductionRungFailure, match="candidate fit failed on A"):
        hooks.run_rung(RungRequest(context, experiment, card, binding, "full"))

    assert worker.active == 0
    assert not any(":full-C-" in attempt_id for attempt_id in worker.completed)
    assert not (
        context.run_dir / "scientific-cache" / experiment["experiment_id"] / "full.json"
    ).exists()
    with repository.database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM attempts WHERE status='reserved'").fetchone()[
                0
            ]
            == 0
        )


def test_e10_derives_weights_only_from_complete_full_shadow_artifacts(tmp_path: Path):
    hooks, context, _, _, _, repository = _request(tmp_path)
    project_configs = hooks.config.project_root / "configs/experiments"
    tree_config = project_configs / "e02_tree_ranker.yaml"
    tree_config.write_text("plugin: fake:Tree\nn_estimators: 5\n", encoding="utf-8")
    train_features, train_targets = hooks._split_paths("train")
    assert train_targets is not None
    folds = materialize_shadow_folds(
        train_features,
        train_targets,
        context.run_dir / "cache/shadow-folds",
    )

    def branch(
        experiment_id: str,
        card_id: str,
        config_path: Path,
        *,
        perfect: bool,
        primary: float,
        state: str,
        repaired_epochs: int | None = None,
    ) -> None:
        proposal = ExperimentProposal(
            experiment_id=experiment_id,
            parent_id=None,
            operator="MODEL_BLOCK",
            hypothesis="A complete shadow branch provides reusable diversity evidence.",
            mechanism="The branch is measured on all three rolling temporal folds.",
            primary_change="branch model",
            files_to_change=[f"configs/experiments/{config_path.name}"],
            expected_metric_effects={"primary": "increase"},
            falsifier="The full shadow mean does not improve.",
            leakage_analysis="Only each fold training prefix supplies labels.",
            estimated_seconds=30,
            cheap_rung={"fold": "A"},
            full_rung={"folds": ["A", "B", "C"]},
        )
        repository.create_experiment(
            context.run_id,
            proposal,
            context.root_commit,
            workspace_path=str(tmp_path / experiment_id),
            branch_name=f"codex/rex-{experiment_id}",
            commit_sha=context.root_commit,
            config_sha256=sha256_file(config_path),
            method_card_id=card_id,
        )
        repository.register_artifact(
            artifact_ref(config_path, "experiment_config"),
            experiment_id=experiment_id,
        )
        repair_number = 0
        if repaired_epochs is not None:
            repaired_config = (
                context.run_dir
                / "evidence"
                / experiment_id
                / "repairs"
                / "effective-config-repair-1.yaml"
            )
            repaired_config.parent.mkdir(parents=True, exist_ok=True)
            repaired_value = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
            repaired_value["epochs"] = repaired_epochs
            repaired_config.write_text(
                __import__("yaml").safe_dump(repaired_value, sort_keys=True),
                encoding="utf-8",
            )
            repository.register_artifact(
                artifact_ref(repaired_config, "repaired_experiment_config"),
                experiment_id=experiment_id,
            )
            with repository.database.transaction() as connection:
                connection.execute(
                    "UPDATE experiments SET config_sha256=? WHERE experiment_id=?",
                    (sha256_file(repaired_config), experiment_id),
                )
            repair_number = 1
        with repository.database.transaction() as connection:
            connection.execute(
                "UPDATE experiments SET state=? WHERE experiment_id=?",
                (state, experiment_id),
            )
        observations = []
        for fold in folds:
            view = load_feature_view(fold.valid_features)
            labels = np.load(fold.valid_targets, allow_pickle=False)["long_view"]
            scores = labels if perfect else 1.0 - labels
            path = (
                context.run_dir
                / "attempts"
                / experiment_id
                / f"full-{fold.name}-candidate-repair-{repair_number}"
                / "predict-output"
                / "predictions.npz"
            )
            write_prediction_artifact(path, view, scores)
            ref = artifact_ref(path, "shadow_predictions")
            repository.register_artifact(ref, experiment_id=experiment_id)
            metric = Metrics(
                GAUC=primary,
                **{"nDCG@5": primary},
                primary=primary,
                users=len(set(view.arrays["user_id"].tolist())),
                rows=view.rows,
                evaluator_sha256=HASH,
                split="shadow",
                fold=fold.name,
                seed=0,
            )
            repository.record_metrics(experiment_id, metric)
            observations.append(
                {
                    "candidate": metric.model_dump(mode="json", by_alias=True),
                    "reference": metric.model_copy(
                        update={
                            "GAUC": primary - 0.01,
                            "ndcg5": primary - 0.01,
                            "primary": primary - 0.01,
                        }
                    ).model_dump(mode="json", by_alias=True),
                }
            )
        full_result = context.run_dir / "evidence" / experiment_id / "full-result.json"
        full_result.parent.mkdir(parents=True, exist_ok=True)
        full_result.write_text(
            json.dumps({"rung": "full", "observations": observations}), encoding="utf-8"
        )
        repository.register_artifact(
            artifact_ref(full_result, "production_full_result"),
            experiment_id=experiment_id,
        )

    branch(
        "pair-branch",
        "E01",
        hooks.config.method_cards["E01"].config_path,
        perfect=True,
        primary=0.7,
        state="PROMOTED",
        repaired_epochs=2,
    )
    branch(
        "tree-branch",
        "E02",
        tree_config,
        perfect=False,
        primary=0.6,
        state="REJECTED",
    )
    derived, refs = hooks._derive_e10_config(context, "scientific-e10")
    value = __import__("yaml").safe_load(derived.read_text(encoding="utf-8"))
    selection = json.loads(
        next(Path(item.path) for item in refs if item.kind == "shadow_blend_selection").read_text(
            encoding="utf-8"
        )
    )
    assert value["inputs"] == ["pair-branch", "tree-branch"]
    assert value["weights"][0] > value["weights"][1]
    assert value["pair_config"]["epochs"] == 2
    assert value["tree_config"]["n_estimators"] == 5
    assert selection["selection_split"] == "shadow_only"
    assert selection["test_scored"] is False
    assert len(selection["grid"]) == 21


def test_shadow_blend_bundle_round_trip_is_complete_and_emits_components(tmp_path: Path):
    rows = 12
    arrays = {
        "row_id": np.arange(rows, dtype=np.int64),
        "date": np.asarray([20220408, 20220409, 20220410] * 4, dtype=np.int32),
        "user_id": np.repeat(np.asarray(["u1", "u2", "u3", "u4"]), 3),
        "video_id": np.asarray(["v1", "v2", "v3"] * 4),
        "author_id": np.asarray(["a1", "a2", "a3"] * 4),
        "tab": np.asarray(["1", "1", "2"] * 4),
        "duration_ms": np.asarray([8000, 15000, 24000] * 4, dtype=np.float32),
    }
    features = FeatureView(tmp_path / "features.npz", arrays, "1" * 64)
    targets = TargetView(
        tmp_path / "targets.npz",
        np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0], dtype=np.float32),
        "2" * 64,
    )
    config = {
        "weights": [0.7, 0.3],
        "normalization": "percentile",
        "pair_config": {
            "epochs": 1,
            "pair_batch_size": 16,
            "negatives_per_positive": 1,
            "bce_weight": 0.0,
        },
        "tree_config": {
            "n_estimators": 5,
            "num_leaves": 5,
            "min_child_samples": 1,
            "n_jobs": 1,
        },
    }
    plugin = ShadowBlendPlugin()
    output = tmp_path / "model"
    primary = plugin.fit(features, targets, config, 7, output)
    bundle_path = create_model_bundle(
        output,
        primary,
        plugin="rex.models.shadow_blend:ShadowBlendPlugin",
        seed=7,
        commit_sha="blend-commit",
        config_sha256="3" * 64,
        data_view_sha256=features.sha256,
        features=features,
    )
    bundle = validate_model_bundle(
        bundle_path,
        expected_plugin="rex.models.shadow_blend:ShadowBlendPlugin",
        expected_commit_sha="blend-commit",
        expected_config_sha256="3" * 64,
        expected_features=features,
    )
    predict_dir = tmp_path / "predict"
    scores = plugin.predict(bundle.primary_path, features, config, predict_dir)
    assert scores.shape == (rows,)
    assert np.isfinite(scores).all()
    with np.load(predict_dir / "component_predictions.npz", allow_pickle=False) as values:
        assert set(values.files) == {"pair", "tree"}


def test_baseline_gate_pins_exact_best_seed_not_the_seed_mean(tmp_path: Path):
    config, _, _, _ = _config(tmp_path)

    def verifier(data_dir, view_dir, evidence_dir, *, seeds):
        del data_dir, seeds
        view_root = Path(view_dir)
        view_root.mkdir(parents=True, exist_ok=True)
        (view_root / "data_manifest.json").write_text(
            json.dumps({"raw_dataset_identity_sha256": HASH}), encoding="utf-8"
        )
        root = Path(evidence_dir)
        results = []
        for seed, primary in ((0, 0.60), (1, 0.61)):
            directory = root / f"seed-{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "predictions.npz").write_bytes(f"prediction-{seed}".encode())
            (directory / "model.npz").write_bytes(f"model-{seed}".encode())
            (directory / "model_bundle.json").write_text("{}", encoding="utf-8")
            (directory / "config.json").write_text(
                json.dumps({"model": "fm", "seed": seed}), encoding="utf-8"
            )
            metrics = Metrics(
                GAUC=primary,
                **{"nDCG@5": primary},
                primary=primary,
                users=4,
                rows=8,
                evaluator_sha256=HASH,
                split="valid",
                seed=seed,
            )
            (directory / "metrics.json").write_text(
                json.dumps(metrics.model_dump(mode="json", by_alias=True)),
                encoding="utf-8",
            )
            results.append(BaselineSeedResult(seed, metrics, seed + 1, directory))
        summary = root / "summary.json"
        summary.write_text("{}", encoding="utf-8")
        bundle = BaselineBundle(tuple(results), 0.605, 0.005)
        acceptance = BaselineAcceptance(True, (), {"fm_mean_primary": 0.605})
        return BaselineEvidence(
            results[0].metrics,
            results[0].metrics,
            bundle,
            acceptance,
            summary,
        )

    context = ProductionContext(
        "baseline-unit",
        config.runs_dir / "baseline-unit",
        config.project_root,
        "root-commit",
        4_102_444_800_000,
    )
    hooks = ProductionScientificHooks(
        config,
        ProductionFixedProvider(),
        baseline_verifier=verifier,
    )
    result = hooks.verify_baseline(context)
    gate = json.loads((context.run_dir / "baseline/gate.json").read_text(encoding="utf-8"))
    assert result.accepted
    assert result.metrics is not None
    assert result.metrics.primary == 0.61
    assert result.metrics.seed == 1
    assert gate["five_seed_reproduction"]["mean_primary"] == 0.605
    assert gate["selected_seed"]["seed"] == 1
    assert gate["metrics"] == gate["selected_seed"]["metrics"]
    assert gate["selected_seed"]["config_path"].endswith("seed-1/config.json")
    assert gate["selected_seed"]["config_sha256"] == sha256_file(
        gate["selected_seed"]["config_path"]
    )
    assert any(item.kind == "baseline_incumbent_config" for item in result.artifacts)


def test_e01_control_changes_loss_only_in_the_shipped_configs():
    root = Path(__file__).resolve().parents[2]
    candidate = __import__("yaml").safe_load(
        (root / "configs/experiments/e01_pair_rank_fm.yaml").read_text(encoding="utf-8")
    )
    control = __import__("yaml").safe_load(
        (root / "configs/experiments/e01_pointwise_control.yaml").read_text(encoding="utf-8")
    )
    assert candidate["k"] == control["k"]
    assert candidate["lr"] == control["lr"]
    assert candidate["l2"] == control["l2"]
    assert candidate["epochs"] == control["epochs"] == 8
    assert candidate["pair_batch_size"] == control["batch_size"] == 4096


def test_timeout_repair_changes_effective_workload_config_before_retry(tmp_path: Path):
    hooks, context, experiment, card, binding, repository = _request(tmp_path)
    plan = TypedRepairPlan(
        repair=True,
        consumes_budget=True,
        repair_number=1,
        action=RepairAction.REDUCE_WORKLOAD,
        reason="bounded timeout workload reduction",
        overrides={"batch_size_scale": 0.5, "workers": 1},
    )
    reservation = repository.reserve_experiment_repair(
        experiment_id=experiment["experiment_id"],
        phase="cheap",
        failure_status=AttemptStatus.TIMEOUT,
        plan={
            "repair": plan.repair,
            "consumes_budget": plan.consumes_budget,
            "repair_number": plan.repair_number,
            "action": plan.action.value,
            "reason": plan.reason,
            "overrides": plan.overrides,
        },
        maximum=2,
    )
    refs = hooks.repair_candidate(RepairRequest(context, experiment, "cheap", plan))
    repaired_ref = next(item for item in refs if item.kind == "repaired_experiment_config")
    repaired = __import__("yaml").safe_load(Path(repaired_ref.path).read_text(encoding="utf-8"))
    original = __import__("yaml").safe_load(binding.config_path.read_text(encoding="utf-8"))
    assert repaired_ref.sha256 != sha256_file(binding.config_path)
    assert repaired["pair_batch_size"] == original["pair_batch_size"] // 2
    assert repaired["max_pairs"] == 250_000
    assert repaired["n_jobs"] == 1
    initial_hashes = [(item.kind, item.sha256) for item in refs]
    Path(next(item.path for item in refs if item.kind == "repair_override")).unlink()
    refs = hooks.repair_candidate(RepairRequest(context, experiment, "cheap", plan))
    assert [(item.kind, item.sha256) for item in refs] == initial_hashes
    for ref in refs:
        repository.register_artifact(ref, experiment_id=experiment["experiment_id"])
    with repository.database.transaction() as connection:
        connection.execute(
            "UPDATE experiments SET state='REPAIRING' WHERE experiment_id=?",
            (experiment["experiment_id"],),
        )
    repository.apply_experiment_repair_revision(
        reservation["repair_id"],
        repaired_commit_sha=experiment["commit_sha"],
        effective_config_artifact_id=repaired_ref.artifact_id,
    )
    experiment = repository.get_experiment(experiment["experiment_id"])
    selected = hooks._effective_config(RungRequest(context, experiment, card, binding, "cheap"))
    assert selected == Path(repaired_ref.path)
    assert sha256_file(selected) == repaired_ref.sha256
    repaired_control = hooks._reference_config(
        context, experiment["experiment_id"], "E01", selected
    )
    control = __import__("yaml").safe_load(repaired_control.read_text(encoding="utf-8"))
    assert repaired_control.is_relative_to(context.run_dir)
    assert control["batch_size"] == repaired["pair_batch_size"]
    assert control["n_jobs"] == repaired["n_jobs"] == 1
    assert control["plugin"] == "fake:Plugin"
    assert control["repair_provenance"]["scientific_control_preserved"] is True
    replayed = hooks.repair_candidate(RepairRequest(context, experiment, "cheap", plan))
    assert [(item.kind, item.sha256) for item in replayed] == [
        (item.kind, item.sha256) for item in refs
    ]
    assert __import__("yaml").safe_load(selected.read_text(encoding="utf-8")) == repaired


def test_corrupt_bundle_repair_quarantines_only_outputs_and_preserves_runner_evidence(
    tmp_path: Path,
):
    hooks, context, experiment, _, _, repository = _request(tmp_path)
    attempt = (
        context.run_dir / "attempts" / experiment["experiment_id"] / "cheap-A-candidate-repair-0"
    )
    (attempt / "fit-output").mkdir(parents=True)
    (attempt / "fit-output/model_bundle.json").write_text("corrupt", encoding="utf-8")
    (attempt / "predict-output").mkdir()
    (attempt / "predict-output/predictions.npz").write_text("corrupt", encoding="utf-8")
    (attempt / "fit").mkdir()
    (attempt / "fit/request.json").write_text("{}", encoding="utf-8")
    plan = TypedRepairPlan(
        repair=True,
        consumes_budget=True,
        repair_number=1,
        action=RepairAction.PATCH,
        reason="quarantine the corrupt output and rebuild from clean provenance",
        overrides={"rebuild_artifact": True},
    )
    repository.reserve_experiment_repair(
        experiment_id=experiment["experiment_id"],
        phase="cheap",
        failure_status=AttemptStatus.INVALID_ARTIFACT,
        plan={
            "repair": plan.repair,
            "consumes_budget": plan.consumes_budget,
            "repair_number": plan.repair_number,
            "action": plan.action.value,
            "reason": plan.reason,
            "overrides": plan.overrides,
        },
        maximum=2,
    )
    refs = hooks.repair_candidate(RepairRequest(context, experiment, "cheap", plan))
    override = next(item for item in refs if item.kind == "repair_override")
    payload = json.loads(Path(override.path).read_text(encoding="utf-8"))
    quarantine = Path(payload["applied_changes"]["quarantined_attempt"])
    assert (attempt / "fit/request.json").is_file()
    assert not (attempt / "fit-output").exists()
    assert not (attempt / "predict-output").exists()
    assert (quarantine / "fit-output/model_bundle.json").is_file()
    assert (quarantine / "predict-output/predictions.npz").is_file()
    marker = json.loads((quarantine / "QUARANTINED.json").read_text(encoding="utf-8"))
    assert marker["runner_evidence_preserved"] is True
    replayed = hooks.repair_candidate(RepairRequest(context, experiment, "cheap", plan))
    assert [(item.kind, item.sha256) for item in replayed] == [
        (item.kind, item.sha256) for item in refs
    ]
