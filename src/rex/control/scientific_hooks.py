"""Concrete valid-only scientific execution hooks for the production supervisor.

This module is the only bridge between the durable control plane and model/data
execution.  Candidate code never receives evaluator or current-row validation
labels: workers receive explicit feature/target capabilities, while evaluation
and diagnostics remain in this trusted coordinator process.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from rex.agents.coordinator import PatchTransactionCoordinator
from rex.agents.patch_guard import PatchPolicy, changed_paths
from rex.agents.provider import ProviderError, StructuredProvider, redact_secrets
from rex.agents.services import CodingService, ProposalService
from rex.agents.static_audit import audit_changed_files
from rex.agents.workspace import GitWorkspace
from rex.contracts import ArtifactRef, AttemptStatus, ExperimentProposal, Metrics, RunRequest, RunResult
from rex.control.budget import BudgetConfig
from rex.control.production_supervisor import (
    BaselineGateResult,
    CandidatePreparation,
    ComparisonObservation,
    MethodCardBinding,
    ProductionContext,
    ProductionFixedProvider,
    ProductionHooks,
    ProductionRungFailure,
    ProductionRungResult,
    ProductionRunConfig,
    RepairRequest,
    RungRequest,
)
from rex.data.bootstrap import default_data_dir
from rex.data.firewall import CapabilityRoots, validate_worker_request
from rex.data.manifest import canonical_json_bytes, sha256_bytes, sha256_file
from rex.data.shadow_views import MaterializedFold, materialize_cheap_view, materialize_shadow_folds
from rex.data.views import FeatureView, load_feature_view, load_target_view
from rex.evaluation.baseline import BaselineEvidence, run_baseline_verification
from rex.evaluation.diagnostics import compare_diagnostics
from rex.evaluation.official_adapter import EvaluationError, evaluate_predictions
from rex.execution.artifacts import (
    ArtifactError,
    artifact_ref,
    atomic_write_json,
    load_prediction_artifact,
)
from rex.execution.runner import execute_request
from rex.execution.sandbox import SandboxMode
from rex.execution.gate import execute_gate
from rex.features.recipes import (
    AUTHOR_DURATION_AFFINITY,
    HISTORY_LENGTH,
    RECENCY_HISTORY,
    REPEAT_EXPOSURE,
    VIDEO_STATISTICS,
    FeatureRecipe,
    RecipeArtifact,
    control_recipe,
    materialize_feature_recipe,
)
from rex.models.ensemble import blend_scores
from rex.store.db import Database
from rex.store.repository import ExperimentRepository


ExecuteRequest = Callable[..., RunResult]
BaselineVerifier = Callable[..., BaselineEvidence]


@dataclass(frozen=True)
class ScientificExecutionSettings:
    cheap_user_fraction: float = 0.25
    cheap_seed: int = 17
    model_seed: int = 0
    max_memory_mb: int = 8192
    bootstrap_samples: int = 500

    def __post_init__(self) -> None:
        if not 0 < self.cheap_user_fraction <= 1:
            raise ValueError("cheap_user_fraction must be in (0, 1]")
        if self.max_memory_mb <= 0 or self.bootstrap_samples <= 0:
            raise ValueError("resource and bootstrap bounds must be positive")


@dataclass(frozen=True)
class _Partition:
    name: str
    train_features: Path
    train_targets: Path
    valid_features: Path
    valid_targets: Path
    identity_sha256: str


@dataclass(frozen=True)
class _PreparedViews:
    train_features: Path
    apply_features: Path
    feature_root: Path
    manifests: tuple[Path, ...]


@dataclass(frozen=True)
class _ModelExecution:
    result: RunResult
    bundle_path: Path
    prediction_path: Path
    scores: np.ndarray
    artifacts: tuple[ArtifactRef, ...]
    component_scores: tuple[np.ndarray, np.ndarray] | None = None


RECIPE_BY_NAME: dict[str, FeatureRecipe] = {
    "video_statistics": VIDEO_STATISTICS,
    "history_length": HISTORY_LENGTH,
    "candidate_history": HISTORY_LENGTH,
    "author_duration_affinity": AUTHOR_DURATION_AFFINITY,
    "affinity": AUTHOR_DURATION_AFFINITY,
    "repeat_exposure": REPEAT_EXPOSURE,
    "recency": RECENCY_HISTORY,
    "recency_history": RECENCY_HISTORY,
    "shadow_blend": HISTORY_LENGTH,
}


REFERENCE_CONFIG_BY_CARD = {
    "E01": "configs/experiments/e01_pointwise_control.yaml",
    "E02": "configs/experiments/e02_tree_ranker_control.yaml",
    "E03": "configs/experiments/e03_candidate_history.yaml",
    "E04": "configs/experiments/e01_pair_rank_fm.yaml",
    "E05": "configs/experiments/e01_pair_rank_fm.yaml",
    "E06": "configs/experiments/e06_repeat_exposure.yaml",
    "E07": "configs/experiments/e07_affinity.yaml",
    "E08": "configs/experiments/e08_recency.yaml",
}

REPAIR_MIRROR_KEYS = frozenset(
    {
        "batch_size",
        "pair_batch_size",
        "bce_batch_size",
        "max_pairs",
        "n_estimators",
        "num_leaves",
        "n_jobs",
        "lr",
        "learning_rate",
        "l2",
        "reg_lambda",
        "bce_weight",
    }
)


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "-" for character in value)


def _all_files(root: Path) -> tuple[Path, ...]:
    return tuple(path for path in sorted(root.rglob("*")) if path.is_file())


def _artifact_kind(path: Path, default: str) -> str:
    if path.name == "model_bundle.json":
        return "model_bundle"
    if path.name == "predictions.npz":
        return "predictions"
    if path.name.endswith("sandbox_evidence.json"):
        return "sandbox_evidence"
    if path.suffix == ".sb":
        return "sandbox_profile"
    return default


def _copy_atomic(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return destination


def subprocess_run(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}: {completed.stderr[-500:]}")
    return completed.stdout.strip()


class ProductionScientificHooks(ProductionHooks):
    """Real-data, restart-safe implementation of every production scientific hook."""

    def __init__(
        self,
        config: ProductionRunConfig,
        provider: StructuredProvider,
        *,
        data_dir: str | Path | None = None,
        view_dir: str | Path | None = None,
        settings: ScientificExecutionSettings | None = None,
        execute: ExecuteRequest = execute_request,
        baseline_verifier: BaselineVerifier = run_baseline_verification,
        python_executable: str = sys.executable,
        preparation_checkpoint: Callable[[str, str], None] | None = None,
    ):
        self.config = config
        self.provider = provider
        self.data_dir = Path(data_dir or config.raw_data_dir or default_data_dir()).resolve()
        self.view_dir = Path(view_dir).resolve() if view_dir is not None else None
        self.settings = settings or ScientificExecutionSettings()
        self.execute = execute
        self.baseline_verifier = baseline_verifier
        self.python_executable = python_executable
        self.preparation_checkpoint = preparation_checkpoint
        self.budget = BudgetConfig.from_yaml(config.budget_config)

    # ---- baseline and immutable data contract ---------------------------------

    def _manifest(self) -> dict[str, Any]:
        payload = json.loads(self.config.data_manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("splits"), dict):
            raise RuntimeError("data manifest is missing split capabilities")
        for split in ("train", "valid", "test"):
            detail = payload["splits"].get(split)
            if not isinstance(detail, dict):
                raise RuntimeError(f"data manifest is missing {split} split")
            feature = Path(str(detail["feature_path"])).resolve()
            if not feature.is_file() or sha256_file(feature) != detail.get("feature_sha256"):
                raise RuntimeError(f"data manifest {split} feature capability drifted")
            target_raw = detail.get("target_path")
            if split == "test":
                if target_raw is not None or detail.get("target_sha256") is not None:
                    raise RuntimeError("data manifest must not contain a test target capability")
            else:
                target = Path(str(target_raw)).resolve()
                if not target.is_file() or sha256_file(target) != detail.get("target_sha256"):
                    raise RuntimeError(f"data manifest {split} target capability drifted")
        return payload

    def _split_paths(self, split: str) -> tuple[Path, Path | None]:
        detail = self._manifest()["splits"][split]
        return Path(detail["feature_path"]).resolve(), (
            Path(detail["target_path"]).resolve() if detail.get("target_path") else None
        )

    def verify_baseline(self, context: ProductionContext) -> BaselineGateResult:
        try:
            manifest = self._manifest()
            generated_view_root = context.run_dir / "baseline" / "views"
            evidence_root = context.run_dir / "baseline" / "evidence"
            evidence = self.baseline_verifier(
                self.data_dir,
                generated_view_root,
                evidence_root,
                seeds=(0, 1, 2, 3, 4),
            )
            generated_manifest = json.loads(
                (generated_view_root / "data_manifest.json").read_text(encoding="utf-8")
            )
            if generated_manifest.get("raw_dataset_identity_sha256") != manifest.get(
                "raw_dataset_identity_sha256"
            ):
                raise RuntimeError("baseline raw-data identity differs from configured manifest")
            if not evidence.acceptance.accepted:
                return BaselineGateResult(
                    False,
                    None,
                    tuple(artifact_ref(path, "baseline_evidence") for path in _all_files(evidence_root)),
                    "; ".join(evidence.acceptance.reasons),
                )
            results = evidence.fm.results
            selected = max(results, key=lambda item: (item.metrics.primary, -item.seed))
            metrics = selected.metrics.model_copy(update={"split": "valid", "fold": None})
            selected_dir = evidence_root / f"seed-{selected.seed}"
            selected_prediction = artifact_ref(
                selected_dir / "predictions.npz", "baseline_incumbent_predictions"
            )
            selected_checkpoint = artifact_ref(
                selected_dir / "model.npz", "baseline_incumbent_checkpoint"
            )
            selected_bundle = artifact_ref(
                selected_dir / "model_bundle.json", "baseline_incumbent_model_bundle"
            )
            selected_config = artifact_ref(
                selected_dir / "config.json", "baseline_incumbent_config"
            )
            baseline_manifest_path = context.run_dir / "baseline" / "gate.json"
            atomic_write_json(
                baseline_manifest_path,
                {
                    "accepted": True,
                    "test_scored": False,
                    "configured_data_manifest": str(self.config.data_manifest),
                    "configured_data_manifest_sha256": sha256_file(self.config.data_manifest),
                    "raw_dataset_identity_sha256": manifest.get("raw_dataset_identity_sha256"),
                    "generated_manifest_sha256": generated_manifest.get("manifest_sha256"),
                    "metrics": metrics.model_dump(mode="json", by_alias=True),
                    "five_seed_reproduction": {
                        "mean_primary": evidence.fm.mean_primary,
                        "std_primary": evidence.fm.std_primary,
                        "acceptance": evidence.acceptance.observed,
                    },
                    "selected_seed": {
                        "seed": selected.seed,
                        "best_epoch": selected.best_epoch,
                        "metrics": metrics.model_dump(mode="json", by_alias=True),
                        "prediction_artifact_id": selected_prediction.artifact_id,
                        "prediction_path": selected_prediction.path,
                        "prediction_sha256": selected_prediction.sha256,
                        "checkpoint_artifact_id": selected_checkpoint.artifact_id,
                        "checkpoint_path": selected_checkpoint.path,
                        "checkpoint_sha256": selected_checkpoint.sha256,
                        "model_bundle_artifact_id": selected_bundle.artifact_id,
                        "model_bundle_path": selected_bundle.path,
                        "model_bundle_sha256": selected_bundle.sha256,
                        "config_artifact_id": selected_config.artifact_id,
                        "config_path": selected_config.path,
                        "config_sha256": selected_config.sha256,
                    },
                },
            )
            refs = [artifact_ref(self.config.data_manifest, "data_manifest")]
            refs.extend(artifact_ref(path, "baseline_evidence") for path in _all_files(evidence_root))
            refs.extend(
                (selected_prediction, selected_checkpoint, selected_bundle, selected_config)
            )
            refs.append(artifact_ref(baseline_manifest_path, "baseline_gate"))
            return BaselineGateResult(True, metrics, tuple(refs))
        except Exception as error:
            failure_path = context.run_dir / "baseline" / "failure.json"
            atomic_write_json(
                failure_path,
                {
                    "accepted": False,
                    "error_type": type(error).__name__,
                    "error": redact_secrets(str(error))[-2000:],
                    "test_scored": False,
                },
            )
            return BaselineGateResult(
                False,
                None,
                (artifact_ref(failure_path, "baseline_failure"),),
                f"{type(error).__name__}: {str(error)[-1000:]}",
            )

    # ---- candidate preparation -------------------------------------------------

    def _main_repository(self, context: ProductionContext) -> ExperimentRepository:
        database = Database(context.run_dir / "state.sqlite3")
        database.initialize()
        return ExperimentRepository(database)

    def _candidate_result(
        self,
        proposal: ExperimentProposal,
        commit_sha: str,
        workspace: Path,
        branch: str,
        artifacts: tuple[ArtifactRef, ...],
        config_path: Path,
    ) -> CandidatePreparation:
        values: dict[str, Any] = {
            "proposal": proposal,
            "commit_sha": commit_sha,
            "workspace_path": workspace,
            "branch_name": branch,
            "artifacts": artifacts,
        }
        available = {item.name for item in fields(CandidatePreparation)}
        if "effective_config_path" in available:
            values["effective_config_path"] = config_path
        if "effective_config_sha256" in available:
            values["effective_config_sha256"] = sha256_file(config_path)
        return CandidatePreparation(**values)

    def _proposal_evidence(
        self,
        context: ProductionContext,
        proposal_context: dict[str, object],
        proposal: ExperimentProposal,
        *,
        provider: str,
        model: str,
    ) -> tuple[ArtifactRef, ...]:
        directory = context.run_dir / "evidence" / proposal.experiment_id / "preparation"
        request_path = atomic_write_json(directory / "proposal-request.json", proposal_context)
        response_path = atomic_write_json(
            directory / "proposal-response.json",
            {"provider": provider, "model": model, "proposal": proposal.model_dump(mode="json")},
        )
        return (
            artifact_ref(request_path, "llm_request"),
            artifact_ref(response_path, "llm_response"),
        )

    def _fixed_candidate(
        self,
        context: ProductionContext,
        card,
        binding: MethodCardBinding,
        proposal_context: dict[str, object],
        parent_commit: str,
    ) -> CandidatePreparation:
        experiment_id = str(
            proposal_context.get("resume_experiment_id")
            or f"{context.run_id}-{card.card_id.lower()}"
        )
        durable = proposal_context.get("durable_proposal")
        if durable is not None:
            proposal = ExperimentProposal.model_validate(durable)
            provider_name = "durable-replay"
            model_name = "fixed-method-card"
        else:
            decision = ProposalService(self.provider).propose(
                {**proposal_context, "experiment_id": experiment_id}
            )
            incumbent = proposal_context.get("incumbent")
            parent_id = incumbent.get("experiment_id") if isinstance(incumbent, dict) else None
            proposal = ExperimentProposal.model_validate(decision.parsed).model_copy(
                update={
                    "experiment_id": experiment_id,
                    "parent_id": parent_id,
                    "files_to_change": [
                        binding.config_path.relative_to(self.config.project_root).as_posix()
                    ],
                }
            )
            provider_name = decision.response.provider
            model_name = decision.response.model
        worktree_root = context.run_dir / "worktrees"
        workspace = self._workspace_for_candidate(
            worktree_root, proposal.experiment_id, parent_commit
        )
        refs = list(
            self._proposal_evidence(
                context,
                proposal_context,
                proposal,
                provider=provider_name,
                model=model_name,
            )
        )
        effective_config = binding.config_path
        if card.card_id == "E10":
            effective_config, selection_refs = self._derive_e10_config(context, proposal.experiment_id)
            refs.extend(selection_refs)
        transaction_path = context.run_dir / "evidence" / proposal.experiment_id / "fixed-config.json"
        atomic_write_json(
            transaction_path,
            {
                "mode": "fixed_config",
                "patch_authored": False,
                "config_path": str(effective_config),
                "config_sha256": sha256_file(effective_config),
                "workspace": str(workspace.root),
                "commit_sha": parent_commit,
            },
        )
        refs.append(artifact_ref(transaction_path, "config_transaction"))
        return self._candidate_result(
            proposal,
            parent_commit,
            workspace.root,
            workspace.branch,
            tuple(refs),
            effective_config,
        )

    def _derive_e10_config(
        self,
        context: ProductionContext,
        experiment_id: str,
    ) -> tuple[Path, tuple[ArtifactRef, ...]]:
        """Select E10 branches and weights using full shadow predictions only."""

        repository = self._main_repository(context)
        pair_id = self._best_branch(repository, context.run_id, {"E01", "E04", "E05"})
        tree_id = self._best_branch(
            repository,
            context.run_id,
            {"E02", "E03", "E06", "E07", "E08"},
        )
        folds = materialize_shadow_folds(
            self._split_paths("train")[0],
            self._split_paths("train")[1],  # type: ignore[arg-type]
            context.run_dir / "cache" / "shadow-folds",
        )
        fold_by_name = {item.name: item for item in folds}
        pair_predictions = self._full_shadow_predictions(repository, pair_id)
        tree_predictions = self._full_shadow_predictions(repository, tree_id)
        if set(pair_predictions) != {"A", "B", "C"} or set(tree_predictions) != {
            "A",
            "B",
            "C",
        }:
            raise RuntimeError("E10 requires complete A/B/C candidate prediction artifacts")
        grid: list[dict[str, Any]] = []
        for pair_weight in np.linspace(0.0, 1.0, 21):
            fold_metrics: list[Metrics] = []
            for fold_name in ("A", "B", "C"):
                fold = fold_by_name[fold_name]
                features = load_feature_view(fold.valid_features)
                labels = load_target_view(fold.valid_targets).labels
                pair_scores = load_prediction_artifact(
                    pair_predictions[fold_name], features
                )["score"]
                tree_scores = load_prediction_artifact(
                    tree_predictions[fold_name], features
                )["score"]
                scores = blend_scores(
                    features.arrays["user_id"],
                    [pair_scores, tree_scores],
                    np.asarray([pair_weight, 1.0 - pair_weight]),
                    normalization="percentile",
                )
                fold_metrics.append(
                    self._score_arrays(
                        features,
                        labels,
                        scores,
                        split="shadow",
                        fold=fold_name,
                    )
                )
            grid.append(
                {
                    "pair_weight": float(pair_weight),
                    "tree_weight": float(1.0 - pair_weight),
                    "mean_primary": float(np.mean([item.primary for item in fold_metrics])),
                    "folds": {
                        item.fold: item.model_dump(mode="json", by_alias=True)
                        for item in fold_metrics
                    },
                }
            )
        selected = max(
            grid,
            key=lambda item: (
                item["mean_primary"],
                -abs(float(item["pair_weight"]) - 0.5),
            ),
        )
        pair_config = self._experiment_config(repository, pair_id)
        tree_config = self._experiment_config(repository, tree_id)
        pair_plugin = str(
            pair_config.pop(
                "plugin",
                "rex.models.experimental.pair_rank_fm:ExperimentalPairRankFMPlugin",
            )
        )
        tree_plugin = str(
            tree_config.pop(
                "plugin",
                "rex.models.experimental.tree_history:ExperimentalTreeHistoryPlugin",
            )
        )
        output = context.run_dir / "evidence" / experiment_id
        selection_path = atomic_write_json(
            output / "shadow-blend-selection.json",
            {
                "schema_version": "1.0",
                "selection_split": "shadow_only",
                "test_scored": False,
                "pair_experiment_id": pair_id,
                "tree_experiment_id": tree_id,
                "grid": grid,
                "selected": selected,
                "prediction_sha256": {
                    "pair": {
                        name: sha256_file(path) for name, path in pair_predictions.items()
                    },
                    "tree": {
                        name: sha256_file(path) for name, path in tree_predictions.items()
                    },
                },
            },
        )
        derived_path = output / "derived-e10.yaml"
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        derived_path.write_text(
            yaml.safe_dump(
                {
                    "plugin": "rex.models.shadow_blend:ShadowBlendPlugin",
                    "normalization": "percentile",
                    "weights": [selected["pair_weight"], selected["tree_weight"]],
                    "selection_split": "shadow_only",
                    "inputs": [pair_id, tree_id],
                    "pair_config": pair_config,
                    "tree_config": tree_config,
                    "pair_plugin": pair_plugin,
                    "tree_plugin": tree_plugin,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return derived_path, (
            artifact_ref(selection_path, "shadow_blend_selection"),
            artifact_ref(derived_path, "derived_experiment_config"),
        )

    @staticmethod
    def _best_branch(
        repository: ExperimentRepository,
        run_id: str,
        card_ids: set[str],
    ) -> str:
        placeholders = ",".join("?" for _ in card_ids)
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT experiment.experiment_id,experiment.state FROM experiments experiment "
                "WHERE experiment.run_id=? "
                f"AND experiment.method_card_id IN ({placeholders}) "
                "AND experiment.state NOT IN ('ABANDONED','FAILED_FINAL') "
                "ORDER BY experiment.iteration_number",
                (run_id, *sorted(card_ids)),
            ).fetchall()
        ranked: list[tuple[float, str]] = []
        for row in rows:
            experiment_id = str(row["experiment_id"])
            if set(ProductionScientificHooks._full_shadow_predictions(repository, experiment_id)) != {
                "A",
                "B",
                "C",
            }:
                continue
            with repository.database.connect() as connection:
                result = connection.execute(
                    "SELECT link.artifact_path,artifact.sha256 FROM artifact_links link "
                    "JOIN artifacts artifact ON artifact.artifact_id=link.artifact_id "
                    "WHERE link.experiment_id=? AND artifact.kind='production_full_result' "
                    "ORDER BY artifact.created_at DESC LIMIT 1",
                    (experiment_id,),
                ).fetchone()
            if result is None:
                continue
            path = Path(str(result["artifact_path"]))
            if not path.is_file() or sha256_file(path) != result["sha256"]:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                observations = payload["observations"]
                if len(observations) != 3:
                    continue
                mean_primary = float(
                    np.mean([float(item["candidate"]["primary"]) for item in observations])
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            ranked.append((mean_primary, experiment_id))
        if not ranked:
            raise RuntimeError(
                "E10 requires supported pairwise and tree/history branch evidence"
            )
        return max(ranked, key=lambda item: (item[0], item[1]))[1]

    @staticmethod
    def _full_shadow_revision(
        repository: ExperimentRepository,
        experiment_id: str,
    ) -> tuple[int, dict[str, Path]]:
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT link.artifact_path,artifact.sha256,attempt.repair_number "
                "FROM artifact_links link JOIN artifacts artifact "
                "ON artifact.artifact_id=link.artifact_id "
                "LEFT JOIN attempts attempt ON attempt.attempt_id=link.attempt_id "
                "WHERE link.experiment_id=? AND artifact.kind='shadow_predictions' "
                "ORDER BY artifact.created_at",
                (experiment_id,),
            ).fetchall()
        revisions: dict[int, dict[str, Path]] = {}
        for row in rows:
            path = Path(str(row["artifact_path"]))
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                continue
            marker = path.as_posix()
            repair_number = row["repair_number"]
            if repair_number is None:
                match = re.search(r"/full-[ABC]-candidate-repair-(\d+)(?:/|$)", marker)
                repair_number = int(match.group(1)) if match else 0
            for fold in ("A", "B", "C"):
                if f"/full-{fold}-candidate-" in marker:
                    revisions.setdefault(int(repair_number), {})[fold] = path
        complete = [number for number, paths in revisions.items() if set(paths) == {"A", "B", "C"}]
        if not complete:
            return 0, {}
        selected = max(complete)
        return selected, revisions[selected]

    @staticmethod
    def _full_shadow_predictions(
        repository: ExperimentRepository,
        experiment_id: str,
    ) -> dict[str, Path]:
        return ProductionScientificHooks._full_shadow_revision(repository, experiment_id)[1]

    @staticmethod
    def _experiment_config(
        repository: ExperimentRepository,
        experiment_id: str,
    ) -> dict[str, Any]:
        experiment = repository.get_experiment(experiment_id)
        prediction_repair, predictions = ProductionScientificHooks._full_shadow_revision(
            repository, experiment_id
        )
        if set(predictions) != {"A", "B", "C"}:
            raise RuntimeError(f"branch {experiment_id} has incomplete prediction provenance")
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT link.artifact_path,artifact.sha256,artifact.kind FROM artifact_links link "
                "JOIN artifacts artifact ON artifact.artifact_id=link.artifact_id "
                "WHERE link.experiment_id=? AND artifact.kind IN "
                "('repaired_experiment_config','experiment_config') "
                "ORDER BY artifact.created_at DESC",
                (experiment_id,),
            ).fetchall()
        for row in rows:
            path = Path(str(row["artifact_path"]))
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                continue
            if prediction_repair:
                match = re.search(r"effective-config-repair-(\d+)", path.name)
                if row["kind"] != "repaired_experiment_config" or match is None:
                    continue
                if int(match.group(1)) != prediction_repair:
                    continue
                if experiment["config_sha256"] not in {None, row["sha256"]}:
                    continue
            elif row["kind"] != "experiment_config":
                continue
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return dict(value)
        raise RuntimeError(f"branch {experiment_id} has no durable effective config")

    def _workspace_for_candidate(
        self,
        worktree_root: Path,
        experiment_id: str,
        parent_commit: str,
    ) -> GitWorkspace:
        name = _safe_id(experiment_id)
        root = worktree_root / name
        branch = f"codex/rex-{name}"
        if root.is_dir():
            observed = subprocess_run(
                ["git", "rev-parse", "HEAD"], root
            )
            status = subprocess_run(
                ["git", "status", "--porcelain", "--untracked-files=normal"], root
            )
            if observed == parent_commit and not status:
                return GitWorkspace(root.resolve(), branch)
            raise RuntimeError("existing candidate worktree is dirty or at an unexpected commit")
        branch_query = subprocess_run(
            ["git", "branch", "--list", branch], self.config.project_root
        )
        if branch_query:
            branch_commit = subprocess_run(
                ["git", "rev-parse", branch], self.config.project_root
            )
            if branch_commit != parent_commit:
                raise RuntimeError("generated candidate branch points at an unexpected commit")
            completed = subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"cannot recycle generated candidate branch: {completed.stderr[-500:]}")
        return GitWorkspace.create(
            self.config.project_root, worktree_root, experiment_id, parent_commit
        )

    def _record_preparation_failure(
        self,
        context: ProductionContext,
        card_id: str,
        error: BaseException,
        proposal_context: dict[str, object],
        attempt: int,
    ) -> None:
        directory = context.run_dir / "evidence" / "preparation-failures" / card_id
        path = directory / f"attempt-{attempt}.json"
        error_code = error.code if isinstance(error, ProviderError) else type(error).__name__
        atomic_write_json(
            path,
            {
                "card_id": card_id,
                "attempt": attempt,
                "role": "proposal_or_patch",
                "provider_chain": type(self.provider).__name__,
                "error_code": error_code,
                "retryable": bool(getattr(error, "retryable", False)),
                "error": redact_secrets(str(error))[-2000:],
                "proposal_context_sha256": sha256_bytes(canonical_json_bytes(proposal_context)),
            },
        )
        repository = self._main_repository(context)
        ref = artifact_ref(path, "llm_failure")
        repository.register_artifact(ref)
        repository.record_llm_call(
            call_id=f"{context.run_id}:{card_id}:preparation-failure:{attempt}",
            run_id=context.run_id,
            experiment_id=None,
            role="proposal_or_patch",
            provider=type(self.provider).__name__,
            model="unknown",
            request_artifact_id=None,
            response_artifact_id=ref.artifact_id,
            schema_valid=False,
            input_tokens=0,
            output_tokens=0,
            wall_seconds=0,
            error=f"{error_code}: {redact_secrets(str(error))[-500:]}",
        )

    def _live_candidate(
        self,
        context: ProductionContext,
        card,
        binding: MethodCardBinding,
        proposal_context: dict[str, object],
        parent_commit: str,
    ) -> CandidatePreparation:
        if proposal_context.get("durable_proposal") is not None:
            return self._resume_live_candidate(context, proposal_context)
        relative_config = binding.config_path.relative_to(self.config.project_root).as_posix()
        failure_root = context.run_dir / "evidence" / "preparation-failures" / card.card_id
        attempt = len(list(failure_root.glob("attempt-*.json"))) + 1
        transaction_root = context.run_dir / "transactions" / card.card_id
        database = Database(transaction_root / "state.sqlite3")
        database.initialize()
        repository = ExperimentRepository(database)
        repository.create_run(
            run_id=context.run_id,
            deadline_epoch_ms=context.deadline_epoch_ms,
            root_commit=context.root_commit,
            environment_sha256="0" * 64,
            data_manifest_sha256=sha256_file(self.config.data_manifest),
            evaluator_sha256=sha256_file(self.config.evaluator_path),
        )
        enriched_context: dict[str, Any] = {
            **proposal_context,
            "experiment_id": f"{context.run_id}-{card.card_id.lower()}",
            "effective_patch_contract": {
                "bound_config": relative_config,
                "require_executed_change": True,
                "allowed_model_namespace": "src/rex/models/experimental/**",
            },
        }
        allowed_file_snapshots = self._allowed_file_snapshots(proposal_context)
        coordinator = PatchTransactionCoordinator(
            repository=repository,
            proposal_service=ProposalService(self.provider),
            coding_service=CodingService(self.provider),
            project_root=self.config.project_root,
            worktree_root=context.run_dir / "worktrees",
            patch_policy=PatchPolicy.from_yaml(self.config.protected_paths),
            sandbox_mode=SandboxMode.PRODUCTION,
            trusted_output_root=context.run_dir,
            command_timeout_seconds=min(180, self.budget.default_attempt_timeout_seconds),
            checkpoint=self.preparation_checkpoint,
        )
        try:
            prepared = coordinator.prepare(
                run_id=context.run_id,
                parent_commit=parent_commit,
                proposal_context=enriched_context,
                coding_context={
                    "method_card": proposal_context["method_card"],
                    "bound_config": relative_config,
                    "allowed_file_snapshots": allowed_file_snapshots,
                    "require_executed_change": True,
                    "test_scored": False,
                },
                external_parent=True,
            )
            if prepared.proposal.experiment_id != enriched_context["experiment_id"]:
                raise RuntimeError("live provider changed the coordinator-assigned experiment ID")
            patch_path = (
                context.run_dir
                / "worktrees"
                / "_artifacts"
                / prepared.proposal.experiment_id
                / "patch.diff"
            )
            paths = set(changed_paths(patch_path.read_text(encoding="utf-8")))
            worktree_config = prepared.workspace.root / relative_config
            config_value = yaml.safe_load(worktree_config.read_text(encoding="utf-8"))
            if not isinstance(config_value, dict) or not isinstance(config_value.get("plugin"), str):
                raise RuntimeError("live candidate config must name the exact model plugin")
            plugin_module = str(config_value["plugin"]).split(":", 1)[0]
            plugin_path = plugin_module.replace(".", "/") + ".py"
            if relative_config not in paths and plugin_path not in paths:
                raise RuntimeError(
                    "live patch does not change the bound config or the plugin it executes"
                )
            if plugin_path in paths and not plugin_path.startswith("src/rex/models/experimental/"):
                raise RuntimeError("live plugin patch is outside the experimental allowlist")
            durable_config = _copy_atomic(
                worktree_config,
                context.run_dir
                / "evidence"
                / prepared.proposal.experiment_id
                / f"effective-config{worktree_config.suffix}",
            )
            artifact_paths = _all_files(
                context.run_dir / "worktrees" / "_artifacts" / prepared.proposal.experiment_id
            )
            refs = tuple(artifact_ref(path, "preparation_evidence") for path in artifact_paths)
            self._mirror_preparation_llm_calls(
                context,
                repository,
                prepared.proposal.experiment_id,
            )
            return self._candidate_result(
                prepared.proposal,
                prepared.commit_sha,
                prepared.workspace.root,
                prepared.workspace.branch,
                (*refs, artifact_ref(durable_config, "effective_experiment_config")),
                durable_config,
            )
        except Exception as error:
            self._record_preparation_failure(
                context, card.card_id, error, enriched_context, attempt
            )
            raise

    def _mirror_preparation_llm_calls(
        self,
        context: ProductionContext,
        transaction_repository: ExperimentRepository,
        experiment_id: str,
    ) -> None:
        """Copy successful live proposal/coding accounting into the durable run DB."""

        with transaction_repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT call.*,request.path AS request_path,response.path AS response_path "
                "FROM llm_calls call LEFT JOIN artifacts request "
                "ON request.artifact_id=call.request_artifact_id LEFT JOIN artifacts response "
                "ON response.artifact_id=call.response_artifact_id ORDER BY call.role"
            ).fetchall()
        main = self._main_repository(context)
        for row in rows:
            request_ref = (
                artifact_ref(Path(str(row["request_path"])), "llm_request")
                if row["request_path"]
                else None
            )
            response_ref = (
                artifact_ref(Path(str(row["response_path"])), "llm_response")
                if row["response_path"]
                else None
            )
            for ref in (request_ref, response_ref):
                if ref is not None:
                    main.register_artifact(ref)
            main.record_llm_call(
                call_id=f"{experiment_id}:preparation:{row['role']}",
                run_id=context.run_id,
                experiment_id=None,
                role=str(row["role"]),
                provider=str(row["provider"]),
                model=str(row["model"]),
                request_artifact_id=request_ref.artifact_id if request_ref else None,
                response_artifact_id=response_ref.artifact_id if response_ref else None,
                schema_valid=bool(row["schema_valid"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                wall_seconds=float(row["wall_seconds"]),
                request_id=row["request_id"],
                error=row["error"],
            )

    def _allowed_file_snapshots(
        self,
        proposal_context: dict[str, object],
    ) -> dict[str, str]:
        """Give read-only source text to CLI/API coders without granting filesystem authority."""

        snapshots: dict[str, str] = {}
        for raw in proposal_context.get("allowed_files", []):
            relative = Path(str(raw))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"live coding allowlist path is not repository-relative: {raw}")
            path = (self.config.project_root / relative).resolve()
            try:
                path.relative_to(self.config.project_root)
            except ValueError as error:
                raise RuntimeError(f"live coding allowlist escaped the project: {raw}") from error
            if not path.is_file():
                raise RuntimeError(f"live coding allowlist file is missing: {raw}")
            if path.stat().st_size > 200_000:
                raise RuntimeError(f"live coding allowlist file is too large for bounded context: {raw}")
            snapshots[relative.as_posix()] = path.read_text(encoding="utf-8")
        return snapshots

    def _resume_live_candidate(
        self,
        context: ProductionContext,
        proposal_context: dict[str, object],
    ) -> CandidatePreparation:
        """Resume the exact committed transaction without another proposal call."""

        experiment_id = str(proposal_context.get("resume_experiment_id") or "")
        if not experiment_id:
            raise RuntimeError("durable live preparation resume lacks experiment identity")
        proposal = ExperimentProposal.model_validate(proposal_context["durable_proposal"])
        if proposal.experiment_id != experiment_id:
            raise RuntimeError("durable live proposal identity mismatch")
        repository = self._main_repository(context)
        experiment = repository.get_experiment(experiment_id)
        if proposal.model_dump(mode="json") != json.loads(experiment["proposal_json"]):
            raise RuntimeError("durable live proposal contents drifted")
        commit_sha = str(experiment.get("commit_sha") or "")
        branch = str(experiment.get("branch_name") or f"codex/rex-{_safe_id(experiment_id)}")
        workspace = Path(
            str(experiment.get("workspace_path") or context.run_dir / "worktrees" / _safe_id(experiment_id))
        ).resolve()
        if not commit_sha:
            raise RuntimeError("durable live preparation has no committed candidate snapshot")
        if workspace.exists():
            head = subprocess_run(["git", "rev-parse", "HEAD"], workspace)
            clean = not subprocess_run(
                ["git", "status", "--porcelain", "--untracked-files=normal"], workspace
            )
            if head != commit_sha or not clean:
                removed = subprocess.run(
                    ["git", "worktree", "remove", "--force", str(workspace)],
                    cwd=self.config.project_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if removed.returncode != 0:
                    raise RuntimeError(
                        f"cannot rebuild durable live worktree: {removed.stderr[-500:]}"
                    )
        if not workspace.exists():
            workspace.parent.mkdir(parents=True, exist_ok=True)
            branch_exists = bool(
                subprocess_run(["git", "branch", "--list", branch], self.config.project_root)
            )
            command = (
                ["git", "worktree", "add", str(workspace), branch]
                if branch_exists
                else ["git", "worktree", "add", "-b", branch, str(workspace), commit_sha]
            )
            completed = subprocess.run(
                command,
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"cannot restore durable live worktree: {completed.stderr[-500:]}"
                )
        if subprocess_run(["git", "rev-parse", "HEAD"], workspace) != commit_sha:
            raise RuntimeError("restored live worktree commit mismatch")
        if subprocess_run(
            ["git", "status", "--porcelain", "--untracked-files=normal"], workspace
        ):
            raise RuntimeError("restored live worktree is not clean")
        config_path = self._durable_config_for_experiment(repository, experiment)
        refs = self._experiment_artifacts(repository, experiment_id)
        return self._candidate_result(
            proposal,
            commit_sha,
            workspace,
            branch,
            refs,
            config_path,
        )

    @staticmethod
    def _durable_config_for_experiment(
        repository: ExperimentRepository,
        experiment: dict[str, Any],
    ) -> Path:
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT link.artifact_path,artifact.sha256 FROM artifact_links link "
                "JOIN artifacts artifact ON artifact.artifact_id=link.artifact_id "
                "WHERE link.experiment_id=? AND artifact.kind='experiment_config' "
                "ORDER BY artifact.created_at DESC",
                (experiment["experiment_id"],),
            ).fetchall()
        for row in rows:
            path = Path(str(row["artifact_path"])).resolve()
            if (
                row["sha256"] == experiment["config_sha256"]
                and path.is_file()
                and sha256_file(path) == row["sha256"]
            ):
                return path
        evidence_dir = (
            repository.database.path.parent
            / "evidence"
            / str(experiment["experiment_id"])
        )
        for path in sorted(evidence_dir.glob("effective-config.*"), reverse=True):
            if path.is_file() and sha256_file(path) == experiment["config_sha256"]:
                return path.resolve()
        raise RuntimeError("durable live effective config is missing or corrupt")

    @staticmethod
    def _experiment_artifacts(
        repository: ExperimentRepository,
        experiment_id: str,
    ) -> tuple[ArtifactRef, ...]:
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT artifact.artifact_id,artifact.kind,link.artifact_path AS path,"
                "artifact.sha256,artifact.size_bytes,artifact.schema_version "
                "FROM artifact_links link JOIN artifacts artifact "
                "ON artifact.artifact_id=link.artifact_id WHERE link.experiment_id=?",
                (experiment_id,),
            ).fetchall()
        return tuple(ArtifactRef.model_validate(dict(row)) for row in rows)

    def prepare_candidate(
        self,
        context: ProductionContext,
        card,
        binding: MethodCardBinding,
        proposal_context: dict[str, object],
        parent_commit: str,
    ) -> CandidatePreparation:
        if isinstance(self.provider, ProductionFixedProvider) or self.config.llm.get("mode") == "fixed":
            return self._fixed_candidate(context, card, binding, proposal_context, parent_commit)
        return self._live_candidate(context, card, binding, proposal_context, parent_commit)

    # ---- model rungs -----------------------------------------------------------

    def _partition_for_rung(self, context: ProductionContext, rung: str) -> tuple[_Partition, ...]:
        train_features, train_targets = self._split_paths("train")
        assert train_targets is not None
        if rung == "official_valid":
            valid_features, valid_targets = self._split_paths("valid")
            assert valid_targets is not None
            identity = sha256_bytes(
                canonical_json_bytes(
                    {
                        "train": sha256_file(train_features),
                        "target": sha256_file(train_targets),
                        "valid": sha256_file(valid_features),
                        "valid_target": sha256_file(valid_targets),
                    }
                )
            )
            return (
                _Partition(
                    "official-valid",
                    train_features,
                    train_targets,
                    valid_features,
                    valid_targets,
                    identity,
                ),
            )
        folds = materialize_shadow_folds(
            train_features,
            train_targets,
            context.run_dir / "cache" / "shadow-folds",
        )
        if rung == "cheap":
            fold_a = next(item for item in folds if item.name == "A")
            cheap = materialize_cheap_view(
                fold_a,
                context.run_dir / "cache" / "cheap",
                fraction=self.settings.cheap_user_fraction,
                seed=self.settings.cheap_seed,
            )
            return (self._from_fold(cheap, "A-cheap"),)
        if rung == "full":
            return tuple(self._from_fold(item, item.name) for item in folds)
        raise ValueError(f"unsupported production rung: {rung}")

    @staticmethod
    def _from_fold(fold: MaterializedFold, name: str) -> _Partition:
        return _Partition(
            name,
            fold.train_features,
            fold.train_targets,
            fold.valid_features,
            fold.valid_targets,
            fold.identity_sha256,
        )

    def _recipe(self, card_id: str, binding_name: str) -> FeatureRecipe | None:
        if binding_name == "control":
            return None
        if card_id == "E03":
            return HISTORY_LENGTH
        if card_id == "E07":
            return AUTHOR_DURATION_AFFINITY
        try:
            return RECIPE_BY_NAME[binding_name]
        except KeyError as error:
            raise RuntimeError(f"unknown leakage-safe recipe: {binding_name}") from error

    def _views(
        self,
        context: ProductionContext,
        partition: _Partition,
        card_id: str,
        binding_name: str,
        *,
        reference: bool,
    ) -> _PreparedViews:
        recipe = self._recipe(card_id, binding_name)
        if recipe is None:
            common_root = Path(
                os.path.commonpath(
                    [partition.train_features.parent, partition.valid_features.parent]
                )
            )
            return _PreparedViews(
                partition.train_features,
                partition.valid_features,
                common_root,
                (),
            )
        selected = control_recipe(recipe) if reference else recipe
        artifact: RecipeArtifact = materialize_feature_recipe(
            selected,
            partition.train_features,
            partition.train_targets,
            partition.valid_features,
            context.run_dir / "cache" / "recipes" / partition.name,
        )
        return _PreparedViews(
            artifact.train_features,
            artifact.apply_features,
            artifact.root,
            (artifact.manifest,),
        )

    def _effective_config(self, request: RungRequest) -> Path:
        expected = str(request.experiment.get("config_sha256") or "")
        with self._main_repository(request.context).database.connect() as connection:
            repaired_rows = connection.execute(
                "SELECT link.artifact_path,artifact.sha256 FROM artifact_links link "
                "JOIN artifacts artifact ON artifact.artifact_id=link.artifact_id "
                "WHERE link.experiment_id=? AND artifact.kind='repaired_experiment_config' "
                "ORDER BY artifact.created_at DESC",
                (request.experiment["experiment_id"],),
            ).fetchall()
            rows = connection.execute(
                "SELECT link.artifact_path,artifact.sha256 FROM artifact_links link "
                "JOIN artifacts artifact ON artifact.artifact_id=link.artifact_id "
                "WHERE link.experiment_id=? AND artifact.kind='experiment_config' "
                "ORDER BY artifact.created_at DESC",
                (request.experiment["experiment_id"],),
            ).fetchall()
        for row in repaired_rows:
            path = Path(str(row["artifact_path"])).resolve()
            if (
                row["sha256"] == expected
                and path.is_file()
                and sha256_file(path) == expected
            ):
                return path
        for row in rows:
            path = Path(str(row["artifact_path"])).resolve()
            if (
                row["sha256"] == expected
                and path.is_file()
                and sha256_file(path) == expected
            ):
                return path
        evidence_dir = request.context.run_dir / "evidence" / str(request.experiment["experiment_id"])
        snapshots = sorted(evidence_dir.glob("effective-config.*"))
        for path in reversed(snapshots):
            if expected and sha256_file(path) == expected:
                return path
        if expected and sha256_file(request.binding.config_path) == expected:
            return request.binding.config_path
        raise RuntimeError("durable effective experiment config is missing or corrupt")

    def _reference_config(
        self,
        context: ProductionContext,
        experiment_id: str,
        card_id: str,
        candidate_config: Path,
    ) -> Path:
        relative = REFERENCE_CONFIG_BY_CARD.get(card_id)
        reference = (self.config.project_root / relative).resolve() if relative else candidate_config
        candidate_hash = sha256_file(candidate_config)
        with self._main_repository(context).database.connect() as connection:
            repaired = connection.execute(
                "SELECT 1 FROM artifact_links link JOIN artifacts artifact "
                "ON artifact.artifact_id=link.artifact_id "
                "WHERE link.experiment_id=? AND artifact.kind='repaired_experiment_config' "
                "AND artifact.sha256=? LIMIT 1",
                (experiment_id, candidate_hash),
            ).fetchone()
        if repaired is None:
            return reference
        candidate_value = yaml.safe_load(candidate_config.read_text(encoding="utf-8"))
        reference_value = yaml.safe_load(reference.read_text(encoding="utf-8"))
        if not isinstance(candidate_value, dict) or not isinstance(reference_value, dict):
            raise RuntimeError("repaired treatment/control configs must be mappings")
        mirrored: dict[str, Any] = {}
        for key in REPAIR_MIRROR_KEYS:
            if key in candidate_value and (key in reference_value or key == "n_jobs"):
                reference_value[key] = candidate_value[key]
                mirrored[key] = candidate_value[key]
        # Pointwise and pairwise controls use different names for the same
        # minibatch resource.  Mirror the repaired size semantically without
        # copying pairwise-only scientific parameters into the pointwise model.
        batch_value = next(
            (
                candidate_value[key]
                for key in ("pair_batch_size", "batch_size", "bce_batch_size")
                if key in candidate_value
            ),
            None,
        )
        if batch_value is not None:
            for key in ("pair_batch_size", "batch_size", "bce_batch_size"):
                if key in reference_value:
                    reference_value[key] = batch_value
                    mirrored[key] = batch_value
        reference_value["repair_provenance"] = {
            "candidate_config_sha256": candidate_hash,
            "reference_source_sha256": sha256_file(reference),
            "mirrored_keys": mirrored,
            "scientific_control_preserved": True,
        }
        destination = (
            context.run_dir
            / "evidence"
            / experiment_id
            / "repairs"
            / f"reference-effective-{candidate_hash[:16]}.yaml"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(reference_value, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
        return destination

    @staticmethod
    def _plugin(config_path: Path, *, card_id: str) -> str:
        if card_id == "E10":
            return "rex.models.shadow_blend:ShadowBlendPlugin"
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("plugin"), str):
            raise RuntimeError(f"experiment config has no plugin: {config_path}")
        return str(value["plugin"])

    def _environment_sha(self, context: ProductionContext) -> str:
        return str(self._main_repository(context).get_run(context.run_id)["environment_sha256"])

    def _repair_number(self, context: ProductionContext, experiment_id: str) -> int:
        return self._main_repository(context).experiment_repairs_used(experiment_id)

    def _execute_one(
        self,
        *,
        context: ProductionContext,
        experiment: dict[str, Any],
        rung: str,
        fold: str,
        side: str,
        config_path: Path,
        plugin: str,
        views: _PreparedViews,
        targets: Path,
        split: str,
    ) -> _ModelExecution:
        experiment_id = str(experiment["experiment_id"])
        workspace = Path(str(experiment["workspace_path"])).resolve()
        commit_sha = str(experiment["commit_sha"])
        repair_number = self._repair_number(context, experiment_id)
        prefix = f"{rung}-{fold}-{side}-repair-{repair_number}"
        attempt_root = context.run_dir / "attempts" / experiment_id / prefix
        config_hash = sha256_file(config_path)
        environment_sha = self._environment_sha(context)
        fit_id = f"{experiment_id}:{prefix}:fit"
        fit_output = attempt_root / "fit-output"
        fit_request = RunRequest(
            run_id=context.run_id,
            experiment_id=experiment_id,
            attempt_id=fit_id,
            parent_id=experiment.get("parent_id"),
            commit_sha=commit_sha,
            plugin=plugin,
            operation="fit",
            config_path=str(config_path),
            config_sha256=config_hash,
            seed=self.settings.model_seed,
            rung="cheap" if rung == "cheap" else "full",
            split="shadow" if split == "shadow" else "train",
            fold=fold,
            feature_view_path=str(views.train_features),
            target_view_path=str(targets),
            workspace_path=str(workspace),
            output_dir=str(fit_output),
            deadline_epoch_ms=context.deadline_epoch_ms,
            timeout_seconds=self.budget.default_attempt_timeout_seconds,
            max_memory_mb=self.settings.max_memory_mb,
            data_view_sha256=sha256_file(views.train_features),
            environment_sha256=environment_sha,
        )
        validate_worker_request(
            fit_request,
            roots=CapabilityRoots(features=views.feature_root, targets=targets.parent),
        )
        fit_result = self._run_and_record(
            context, fit_request, attempt_root / "fit", f"{rung}:{fold}:{side}:fit"
        )
        if fit_result.status != AttemptStatus.SUCCESS:
            raise ProductionRungFailure(
                fit_result.status,
                f"{side} fit failed on {fold}: {fit_result.error_summary or fit_result.error_type}",
            )
        bundle_ref = next(
            (item for item in fit_result.artifacts if item.kind.endswith("model_bundle")), None
        )
        if bundle_ref is None:
            raise ProductionRungFailure(AttemptStatus.INVALID_ARTIFACT, "fit returned no model bundle")
        predict_id = f"{experiment_id}:{prefix}:predict"
        predict_output = attempt_root / "predict-output"
        predict_request = RunRequest(
            run_id=context.run_id,
            experiment_id=experiment_id,
            attempt_id=predict_id,
            parent_id=experiment.get("parent_id"),
            commit_sha=commit_sha,
            plugin=plugin,
            operation="predict",
            config_path=str(config_path),
            config_sha256=config_hash,
            seed=self.settings.model_seed,
            rung="predict",
            split="shadow" if split == "shadow" else "valid",
            fold=fold,
            feature_view_path=str(views.apply_features),
            target_view_path=None,
            workspace_path=str(workspace),
            model_bundle_path=bundle_ref.path,
            output_dir=str(predict_output),
            deadline_epoch_ms=context.deadline_epoch_ms,
            timeout_seconds=self.budget.default_attempt_timeout_seconds,
            max_memory_mb=self.settings.max_memory_mb,
            data_view_sha256=sha256_file(views.apply_features),
            environment_sha256=environment_sha,
        )
        validate_worker_request(
            predict_request,
            roots=CapabilityRoots(features=views.feature_root),
        )
        predict_result = self._run_and_record(
            context,
            predict_request,
            attempt_root / "predict",
            f"{rung}:{fold}:{side}:predict",
        )
        if predict_result.status != AttemptStatus.SUCCESS:
            raise ProductionRungFailure(
                predict_result.status,
                f"{side} prediction failed on {fold}: "
                f"{predict_result.error_summary or predict_result.error_type}",
            )
        prediction_ref = next(
            (item for item in predict_result.artifacts if item.kind.endswith("predictions")), None
        )
        if prediction_ref is None:
            raise ProductionRungFailure(
                AttemptStatus.INVALID_ARTIFACT, "prediction returned no prediction artifact"
            )
        arrays = load_prediction_artifact(prediction_ref.path, views.apply_features)
        component_scores: tuple[np.ndarray, np.ndarray] | None = None
        component_path = predict_output / "component_predictions.npz"
        if component_path.is_file():
            with np.load(component_path, allow_pickle=False) as values:
                if set(values.files) != {"pair", "tree"}:
                    raise ProductionRungFailure(
                        AttemptStatus.INVALID_ARTIFACT, "blend component sidecar has invalid fields"
                    )
                pair = np.asarray(values["pair"], dtype=np.float64)
                tree = np.asarray(values["tree"], dtype=np.float64)
            if pair.shape != arrays["score"].shape or tree.shape != arrays["score"].shape:
                raise ProductionRungFailure(
                    AttemptStatus.INVALID_ARTIFACT, "blend component predictions are misaligned"
                )
            component_scores = (pair, tree)
        artifacts = tuple((*fit_result.artifacts, *predict_result.artifacts))
        return _ModelExecution(
            predict_result,
            Path(bundle_ref.path),
            Path(prediction_ref.path),
            np.asarray(arrays["score"], dtype=np.float64),
            artifacts,
            component_scores,
        )

    def _run_and_record(
        self,
        context: ProductionContext,
        request: RunRequest,
        attempt_dir: Path,
        rung: str,
    ) -> RunResult:
        repository = self._main_repository(context)
        repository.reserve_attempt(
            attempt_id=request.attempt_id,
            experiment_id=request.experiment_id,
            rung=rung,
            repair_number=self._repair_number(context, request.experiment_id),
            commit_sha=request.commit_sha,
        )
        result = self.execute(
            request,
            attempt_dir,
            python_executable=self.python_executable,
            trusted_worktree_root=context.run_dir / "worktrees",
            sandbox_mode=SandboxMode.PRODUCTION,
            trusted_output_root=context.run_dir,
        )
        scoped: list[ArtifactRef] = []
        candidate_official = "official_valid" in rung and "candidate" in rung
        reference_official = "official_valid" in rung and "reference" in rung
        for ref in result.artifacts:
            kind = ref.kind
            if kind == "model_bundle":
                if candidate_official:
                    kind = "model_bundle"
                elif reference_official:
                    kind = "reference_model_bundle"
                elif "reference" in rung:
                    kind = "reference_shadow_model_bundle"
                else:
                    kind = "shadow_model_bundle"
            elif kind == "predictions":
                if candidate_official:
                    kind = "valid_predictions"
                elif reference_official:
                    kind = "reference_valid_predictions"
                elif "reference" in rung:
                    kind = "reference_shadow_predictions"
                else:
                    kind = "shadow_predictions"
            scoped.append(
                ref.model_copy(
                    update={"artifact_id": f"{kind}-{ref.sha256[:16]}", "kind": kind}
                )
            )
        result = result.model_copy(update={"artifacts": scoped})
        for ref in result.artifacts:
            repository.register_artifact(ref, attempt_id=request.attempt_id)
        repository.record_attempt(
            result,
            rung=rung,
            repair_number=self._repair_number(context, request.experiment_id),
        )
        return result

    @staticmethod
    def _metrics(
        feature: Path,
        target: Path,
        prediction: Path,
        *,
        split: str,
        fold: str | None,
        seed: int,
    ) -> Metrics:
        return evaluate_predictions(
            feature,
            target,
            prediction,
            split=split,
            fold=fold,
            seed=seed,
        )

    def _evaluate_pair(
        self,
        context: ProductionContext,
        request: RungRequest,
        partition: _Partition,
    ) -> tuple[ComparisonObservation, tuple[ArtifactRef, ...], dict[str, Any]]:
        card_id = request.method_card.card_id
        candidate_config = self._effective_config(request)
        candidate_views = self._views(
            context, partition, card_id, request.binding.feature_recipe, reference=False
        )
        split = "valid" if request.rung == "official_valid" else "shadow"
        evaluation_fold = None if split == "valid" else partition.name
        candidate = self._execute_one(
            context=context,
            experiment=request.experiment,
            rung=request.rung,
            fold=partition.name,
            side="candidate",
            config_path=candidate_config,
            plugin=self._plugin(candidate_config, card_id=card_id),
            views=candidate_views,
            targets=partition.train_targets,
            split=split,
        )
        if request.rung == "official_valid" and self._search_champion(context) is not None:
            candidate_metrics = self._metrics(
                candidate_views.apply_features,
                partition.valid_targets,
                candidate.prediction_path,
                split=split,
                fold=evaluation_fold,
                seed=self.settings.model_seed,
            )
            reference_metrics, reference_scores, incumbent_ref = self._incumbent_reference(
                context,
                candidate_views.apply_features,
            )
            reference_artifacts = (incumbent_ref,)
        elif card_id == "E10":
            if candidate.component_scores is None:
                raise ProductionRungFailure(
                    AttemptStatus.INVALID_ARTIFACT, "E10 did not emit component predictions"
                )
            evaluation = load_feature_view(candidate_views.apply_features)
            labels = load_target_view(partition.valid_targets).labels
            component_metrics = [
                self._score_arrays(
                    evaluation,
                    labels,
                    values,
                    split=split,
                    fold=evaluation_fold,
                )
                for values in candidate.component_scores
            ]
            strongest = int(np.argmax([item.primary for item in component_metrics]))
            reference_scores = candidate.component_scores[strongest]
            reference_metrics = component_metrics[strongest]
            candidate_metrics = self._metrics(
                candidate_views.apply_features,
                partition.valid_targets,
                candidate.prediction_path,
                split=split,
                fold=evaluation_fold,
                seed=self.settings.model_seed,
            )
            reference_path = context.run_dir / "evidence" / str(
                request.experiment["experiment_id"]
            ) / request.rung / partition.name / "strongest-component.npz"
            from rex.execution.artifacts import write_prediction_artifact

            write_prediction_artifact(reference_path, evaluation, reference_scores)
            reference_artifacts = (artifact_ref(reference_path, "shadow_component_predictions"),)
        else:
            reference_config = self._reference_config(
                context,
                str(request.experiment["experiment_id"]),
                card_id,
                candidate_config,
            )
            reference_views = self._views(
                context, partition, card_id, request.binding.feature_recipe, reference=True
            )
            reference = self._execute_one(
                context=context,
                experiment=request.experiment,
                rung=request.rung,
                fold=partition.name,
                side="reference",
                config_path=reference_config,
                plugin=self._plugin(reference_config, card_id=card_id),
                views=reference_views,
                targets=partition.train_targets,
                split=split,
            )
            candidate_metrics = self._metrics(
                candidate_views.apply_features,
                partition.valid_targets,
                candidate.prediction_path,
                split=split,
                fold=evaluation_fold,
                seed=self.settings.model_seed,
            )
            reference_metrics = self._metrics(
                reference_views.apply_features,
                partition.valid_targets,
                reference.prediction_path,
                split=split,
                fold=evaluation_fold,
                seed=self.settings.model_seed,
            )
            reference_scores = reference.scores
            reference_artifacts = reference.artifacts
            if reference_config.is_relative_to(request.context.run_dir):
                reference_artifacts = (
                    *reference_artifacts,
                    artifact_ref(reference_config, "repaired_reference_config"),
                )
        evaluation = load_feature_view(candidate_views.apply_features)
        labels = load_target_view(partition.valid_targets).labels
        history = load_feature_view(candidate_views.train_features)
        diagnostics = compare_diagnostics(
            evaluation,
            labels,
            candidate.scores,
            reference_scores,
            history=history,
            bootstrap_samples=self.settings.bootstrap_samples,
            seed=self.settings.model_seed,
        )
        evidence_dir = (
            context.run_dir
            / "evidence"
            / str(request.experiment["experiment_id"])
            / request.rung
            / partition.name
        )
        diagnostics_path = atomic_write_json(evidence_dir / "diagnostics.json", diagnostics)
        metrics_path = atomic_write_json(
            evidence_dir / "metrics.json",
            {
                "candidate": candidate_metrics.model_dump(mode="json", by_alias=True),
                "reference": reference_metrics.model_dump(mode="json", by_alias=True),
                "test_scored": False,
            },
        )
        refs = [*candidate.artifacts, *reference_artifacts]
        refs.extend(artifact_ref(path, "feature_recipe_manifest") for path in candidate_views.manifests)
        refs.extend(
            (
                artifact_ref(diagnostics_path, "diagnostics"),
                artifact_ref(metrics_path, "metrics_evidence"),
            )
        )
        return (
            ComparisonObservation(candidate_metrics, reference_metrics),
            tuple(refs),
            diagnostics,
        )

    def _search_champion(self, context: ProductionContext) -> str | None:
        value = self._main_repository(context).get_run(context.run_id).get(
            "search_champion_experiment_id"
        )
        return str(value) if value else None

    def _incumbent_reference(
        self,
        context: ProductionContext,
        valid_features: Path,
    ) -> tuple[Metrics, np.ndarray, ArtifactRef]:
        repository = self._main_repository(context)
        champion = self._search_champion(context)
        if champion == "baseline":
            gate_path = context.run_dir / "baseline" / "gate.json"
            if not gate_path.is_file():
                raise ProductionRungFailure(
                    AttemptStatus.INVALID_ARTIFACT,
                    "baseline gate selection evidence is missing",
                )
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            selected = gate.get("selected_seed", {})
            prediction = Path(str(selected.get("prediction_path", ""))).resolve()
            if (
                not prediction.is_file()
                or sha256_file(prediction) != selected.get("prediction_sha256")
            ):
                raise ProductionRungFailure(
                    AttemptStatus.INVALID_ARTIFACT,
                    "selected baseline validation prediction is missing or corrupt",
                )
            metrics = Metrics.model_validate(selected.get("metrics"))
            if metrics.seed != selected.get("seed") or metrics != Metrics.model_validate(
                gate.get("metrics")
            ):
                raise ProductionRungFailure(
                    AttemptStatus.INVALID_ARTIFACT,
                    "selected baseline metric does not match the established baseline",
                )
            ref = artifact_ref(prediction, "incumbent_valid_predictions")
        elif champion:
            with repository.database.connect() as connection:
                prediction_row = connection.execute(
                    "SELECT artifact.artifact_id,artifact.kind,link.artifact_path AS path,"
                    "artifact.sha256,artifact.size_bytes,artifact.schema_version "
                    "FROM artifact_links link JOIN artifacts artifact "
                    "ON artifact.artifact_id=link.artifact_id WHERE link.experiment_id=? "
                    "AND artifact.kind='valid_predictions' ORDER BY artifact.created_at DESC LIMIT 1",
                    (champion,),
                ).fetchone()
                metric_row = connection.execute(
                    "SELECT * FROM metrics WHERE experiment_id=? AND split='valid' "
                    "ORDER BY metric_id DESC LIMIT 1",
                    (champion,),
                ).fetchone()
            if prediction_row is None or metric_row is None:
                raise ProductionRungFailure(
                    AttemptStatus.INVALID_ARTIFACT,
                    f"promoted incumbent {champion} lacks durable validation evidence",
                )
            ref = ArtifactRef.model_validate(dict(prediction_row))
            if not Path(ref.path).is_file() or sha256_file(ref.path) != ref.sha256:
                raise ProductionRungFailure(
                    AttemptStatus.INVALID_ARTIFACT,
                    f"promoted incumbent {champion} prediction drifted",
                )
            metrics = Metrics(
                GAUC=float(metric_row["gauc"]),
                **{"nDCG@5": float(metric_row["ndcg5"])},
                primary=float(metric_row["primary_score"]),
                users=int(metric_row["users"]),
                rows=int(metric_row["rows"]),
                evaluator_sha256=str(metric_row["evaluator_sha256"]),
                split="valid",
                fold=None,
                seed=metric_row["seed"],
            )
        else:
            raise ProductionRungFailure(
                AttemptStatus.CONTRACT, "official validation has no established search champion"
            )
        arrays = load_prediction_artifact(ref.path, valid_features)
        return metrics, np.asarray(arrays["score"], dtype=np.float64), ref

    @staticmethod
    def _score_arrays(
        features: FeatureView,
        labels: np.ndarray,
        scores: np.ndarray,
        *,
        split: str,
        fold: str | None,
    ) -> Metrics:
        from rex.evaluation.official_adapter import evaluate_arrays

        return evaluate_arrays(
            features.arrays["user_id"],
            labels,
            scores,
            split=split,
            fold=fold,
            seed=0,
        )

    def _cached_rung(self, request: RungRequest) -> ProductionRungResult | None:
        path = (
            request.context.run_dir
            / "scientific-cache"
            / str(request.experiment["experiment_id"])
            / f"{request.rung}.json"
        )
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        refs = tuple(ArtifactRef.model_validate(item) for item in payload["artifacts"])
        if any(not Path(ref.path).is_file() or sha256_file(ref.path) != ref.sha256 for ref in refs):
            return None
        result_values: dict[str, Any] = {
            "observations": tuple(
                ComparisonObservation(
                    Metrics.model_validate(item["candidate"]),
                    Metrics.model_validate(item["reference"]),
                )
                for item in payload["observations"]
            ),
            "artifacts": refs,
        }
        if "diagnostics" in {item.name for item in fields(ProductionRungResult)}:
            result_values["diagnostics"] = payload.get("diagnostics", {})
        return ProductionRungResult(**result_values)

    def run_rung(self, request: RungRequest) -> ProductionRungResult:
        cached = self._cached_rung(request)
        if cached is not None:
            return cached
        observations: list[ComparisonObservation] = []
        refs: list[ArtifactRef] = []
        diagnostics_by_fold: dict[str, dict[str, Any]] = {}
        try:
            for partition in self._partition_for_rung(request.context, request.rung):
                observation, evidence, report = self._evaluate_pair(
                    request.context, request, partition
                )
                observations.append(observation)
                refs.extend(evidence)
                diagnostics_by_fold[partition.name] = report
        except ProductionRungFailure:
            raise
        except (ArtifactError, EvaluationError, ValueError) as error:
            status = AttemptStatus.NAN if "nan" in str(error).lower() else AttemptStatus.INVALID_ARTIFACT
            raise ProductionRungFailure(status, str(error)[-1000:]) from error
        except TimeoutError as error:
            raise ProductionRungFailure(AttemptStatus.TIMEOUT, str(error)) from error
        except MemoryError as error:
            raise ProductionRungFailure(AttemptStatus.OOM, str(error)) from error
        diagnostics = self._summarize_diagnostics(diagnostics_by_fold)
        cache_path = (
            request.context.run_dir
            / "scientific-cache"
            / str(request.experiment["experiment_id"])
            / f"{request.rung}.json"
        )
        atomic_write_json(
            cache_path,
            {
                "observations": [
                    {
                        "candidate": item.candidate.model_dump(mode="json", by_alias=True),
                        "reference": item.reference.model_dump(mode="json", by_alias=True),
                    }
                    for item in observations
                ],
                "artifacts": [item.model_dump(mode="json") for item in refs],
                "diagnostics": diagnostics,
            },
        )
        refs.append(artifact_ref(cache_path, "scientific_rung_cache"))
        values: dict[str, Any] = {"observations": tuple(observations), "artifacts": tuple(refs)}
        if "diagnostics" in {item.name for item in fields(ProductionRungResult)}:
            values["diagnostics"] = diagnostics
        return ProductionRungResult(**values)

    @staticmethod
    def _summarize_diagnostics(
        diagnostics_by_fold: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if len(diagnostics_by_fold) == 1:
            return next(iter(diagnostics_by_fold.values()))
        correlations = [
            float(value["prediction_correlation"])
            for value in diagnostics_by_fold.values()
            if "prediction_correlation" in value
        ]
        segment_names = sorted(
            {
                name
                for value in diagnostics_by_fold.values()
                for name in value.get("segment_primary_deltas", {})
            }
        )
        segment_deltas = {
            name: float(
                np.mean(
                    [
                        float(value["segment_primary_deltas"][name])
                        for value in diagnostics_by_fold.values()
                        if name in value.get("segment_primary_deltas", {})
                    ]
                )
            )
            for name in segment_names
        }
        deltas = [value.get("delta", {}) for value in diagnostics_by_fold.values()]
        mean_delta = {
            name: float(np.mean([float(value[name]) for value in deltas if name in value]))
            for name in ("GAUC", "nDCG@5", "primary")
            if any(name in value for value in deltas)
        }
        return {
            "delta": mean_delta,
            "prediction_correlation": float(np.mean(correlations)) if correlations else 0.0,
            "segment_primary_deltas": segment_deltas,
            "segment_wins": sorted(name for name, value in segment_deltas.items() if value >= 0.002),
            "segment_regressions": sorted(
                name for name, value in segment_deltas.items() if value <= -0.002
            ),
            "folds": diagnostics_by_fold,
        }

    # ---- bounded repair handoff ------------------------------------------------

    def repair_candidate(self, request: RepairRequest) -> tuple[ArtifactRef, ...]:
        """Apply one real typed repair; the supervisor owns the global two-repair cap."""

        directory = (
            request.context.run_dir
            / "evidence"
            / str(request.experiment["experiment_id"])
            / "repairs"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"repair-{request.plan.repair_number}-{request.phase}.json"
        replay = self._replay_repair(request, path)
        if replay is not None:
            return replay
        artifacts: list[ArtifactRef] = []
        effective_config = self._config_for_repair(request)
        config_value = yaml.safe_load(effective_config.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise RuntimeError("repair source config is not a mapping")
        repaired_config: Path | None = None
        update: dict[str, Any] = {}
        if request.plan.action.value == "reduce_workload":
            update = self._workload_repair(config_value)
        elif request.plan.action.value == "request_constrained_patch":
            failure_status = self._latest_repair_failure_status(request)
            if failure_status in {
                AttemptStatus.INVALID_ARTIFACT,
                AttemptStatus.CONTRACT,
            }:
                quarantined = self._quarantine_failed_attempt(request)
                update = {"quarantined_attempt": str(quarantined) if quarantined else None}
                if quarantined is not None:
                    artifacts.append(
                        artifact_ref(quarantined / "QUARANTINED.json", "quarantine_evidence")
                    )
            elif self.config.llm.get("mode") == "fixed":
                if failure_status == AttemptStatus.NAN:
                    update = self._numeric_stabilization(config_value)
                else:
                    raise RuntimeError(
                        f"fixed config mode cannot safely patch {failure_status}; terminalizing"
                    )
            else:
                return self._live_patch_repair(request, failure_status, path)
        if any(key in config_value for key in update) or any(
            key in update
            for key in (
                "batch_size",
                "pair_batch_size",
                "bce_batch_size",
                "max_pairs",
                "n_estimators",
                "num_leaves",
                "n_jobs",
                "lr",
                "l2",
                "bce_weight",
            )
        ):
            config_value.update(
                {
                    key: value
                    for key, value in update.items()
                    if key != "quarantined_attempt"
                }
            )
            config_value["repair_provenance"] = {
                "repair_number": request.plan.repair_number,
                "phase": request.phase,
                "action": request.plan.action.value,
            }
            repaired_config = directory / f"effective-config-repair-{request.plan.repair_number}.yaml"
            repaired_config.write_text(
                yaml.safe_dump(config_value, sort_keys=True), encoding="utf-8"
            )
            artifacts.append(artifact_ref(repaired_config, "repaired_experiment_config"))
        self._write_fixed_repair_manifest(
            request,
            path,
            effective_config,
            repaired_config,
            update,
            tuple(artifacts),
        )
        artifacts.append(artifact_ref(path, "repair_override"))
        return tuple(artifacts)

    def _replay_repair(
        self,
        request: RepairRequest,
        evidence_path: Path,
    ) -> tuple[ArtifactRef, ...] | None:
        """Replay one completed repair transaction without mutating it again."""

        update_path = evidence_path.parent / (
            f"repair-{request.plan.repair_number}-candidate-update.json"
        )
        if not evidence_path.is_file() and not update_path.is_file():
            self._recover_partial_repair_evidence(request, evidence_path, update_path)
        if not evidence_path.is_file() and update_path.is_file():
            refs = self._existing_live_repair_refs(request, update_path)
            atomic_write_json(
                evidence_path,
                {
                    "repair_number": request.plan.repair_number,
                    "phase": request.phase,
                    "action": request.plan.action.value,
                    "reason": request.plan.reason,
                    "overrides": request.plan.overrides,
                    "live_patch": True,
                    "candidate_update_path": str(update_path),
                    "candidate_update_sha256": sha256_file(update_path),
                    "test_scored": False,
                    "artifacts": [item.model_dump(mode="json") for item in refs],
                },
            )
        if not evidence_path.is_file():
            return None
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        expected = {
            "repair_number": request.plan.repair_number,
            "phase": request.phase,
            "action": request.plan.action.value,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"repair replay evidence conflicts on {key}")
        raw_refs = payload.get("artifacts")
        if not isinstance(raw_refs, list):
            raise RuntimeError("repair replay evidence has no immutable artifact manifest")
        refs = tuple(ArtifactRef.model_validate(item) for item in raw_refs)
        for ref in refs:
            artifact_path = Path(ref.path)
            if not artifact_path.is_file() or sha256_file(artifact_path) != ref.sha256:
                raise RuntimeError(f"repair replay artifact is missing or corrupt: {ref.kind}")
        if payload.get("live_patch") or any(item.kind == "repair_candidate_update" for item in refs):
            update_ref = next(
                (item for item in refs if item.kind == "repair_candidate_update"), None
            )
            if update_ref is None:
                raise RuntimeError("live repair replay is missing candidate update evidence")
            self._validate_live_repair_update(request, Path(update_ref.path))
            self._record_replayed_live_repair(request, refs)
        quarantine = payload.get("applied_changes", {}).get("quarantined_attempt")
        if quarantine is not None:
            marker = Path(str(quarantine)) / "QUARANTINED.json"
            if not marker.is_file():
                raise RuntimeError("quarantined repair evidence is missing")
        return (*refs, artifact_ref(evidence_path, "repair_override"))

    def _recover_partial_repair_evidence(
        self,
        request: RepairRequest,
        evidence_path: Path,
        update_path: Path,
    ) -> None:
        """Finish evidence after a kill without repeating an already-materialized repair."""

        number = request.plan.repair_number
        configs = sorted(evidence_path.parent.glob(f"effective-config-repair-{number}.*"))
        if len(configs) > 1:
            raise RuntimeError("partial repair has conflicting effective configs")
        if not configs:
            quarantines = sorted(
                (
                    request.context.run_dir
                    / "quarantine"
                    / str(request.experiment["experiment_id"])
                ).glob(f"repair-{number}-*")
            )
            if not quarantines:
                return
            if len(quarantines) != 1 or not (quarantines[0] / "QUARANTINED.json").is_file():
                raise RuntimeError("partial quarantine repair evidence is ambiguous")
            source = self._config_for_repair(request)
            marker_ref = artifact_ref(
                quarantines[0] / "QUARANTINED.json", "quarantine_evidence"
            )
            self._write_fixed_repair_manifest(
                request,
                evidence_path,
                source,
                None,
                {"quarantined_attempt": str(quarantines[0])},
                (marker_ref,),
            )
            return
        repaired = configs[0]
        value = yaml.safe_load(repaired.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("partial repaired config is not a mapping")
        provenance = value.get("repair_provenance")
        fixed_materialization = (
            isinstance(provenance, dict)
            and int(provenance.get("repair_number", -1)) == number
            and provenance.get("phase") == request.phase
            and provenance.get("action") == request.plan.action.value
        )
        if fixed_materialization:
            source = self._config_for_repair(request)
            source_value = yaml.safe_load(source.read_text(encoding="utf-8"))
            if not isinstance(source_value, dict):
                raise RuntimeError("partial repair source config is not a mapping")
            update = {
                key: item
                for key, item in value.items()
                if key != "repair_provenance" and source_value.get(key) != item
            }
            config_ref = artifact_ref(repaired, "repaired_experiment_config")
            self._write_fixed_repair_manifest(
                request,
                evidence_path,
                source,
                repaired,
                update,
                (config_ref,),
            )
            return
        self._recover_live_repair_update(request, repaired, update_path)

    @staticmethod
    def _write_fixed_repair_manifest(
        request: RepairRequest,
        evidence_path: Path,
        source_config: Path,
        repaired_config: Path | None,
        update: dict[str, Any],
        artifacts: tuple[ArtifactRef, ...],
    ) -> None:
        atomic_write_json(
            evidence_path,
            {
                "repair_number": request.plan.repair_number,
                "phase": request.phase,
                "action": request.plan.action.value,
                "reason": request.plan.reason,
                "overrides": request.plan.overrides,
                "applied_changes": update,
                "source_config_path": str(source_config),
                "source_config_sha256": sha256_file(source_config),
                "repaired_config_path": str(repaired_config) if repaired_config else None,
                "repaired_config_sha256": (
                    sha256_file(repaired_config) if repaired_config else None
                ),
                "sandbox_threads_already_fixed_to_one": True,
                "science_gate_unchanged": True,
                "test_scored": False,
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
            },
        )

    def _recover_live_repair_update(
        self,
        request: RepairRequest,
        repaired_config: Path,
        update_path: Path,
    ) -> None:
        """Reconstruct post-commit metadata; never call the coding provider again."""

        proposal = ExperimentProposal.model_validate_json(request.experiment["proposal_json"])
        workspace = Path(str(request.experiment["workspace_path"])).resolve()
        commit_sha = subprocess_run(["git", "rev-parse", "HEAD"], workspace)
        if commit_sha == request.experiment["commit_sha"]:
            raise RuntimeError("partial live repair config has no repaired worktree commit")
        if subprocess_run(
            ["git", "status", "--porcelain", "--untracked-files=normal"], workspace
        ):
            raise RuntimeError("partial live repair worktree is dirty")
        number = request.plan.repair_number
        directory = update_path.parent
        patch_path = directory / f"repair-{number}.diff"
        if not patch_path.is_file():
            raise RuntimeError("partial live repair is missing its constrained patch")
        patch_text = patch_path.read_text(encoding="utf-8")
        observed_paths = changed_paths(patch_text)
        if not set(observed_paths).issubset(set(proposal.files_to_change)):
            raise RuntimeError("partial live repair changed a non-allowlisted path")
        request_path = directory / f"repair-{number}-llm-request.json"
        if not request_path.is_file():
            atomic_write_json(
                request_path,
                {
                    "experiment_id": proposal.experiment_id,
                    "failure_status": self._latest_repair_failure_status(request).value,
                    "phase": request.phase,
                    "allowed_files": proposal.files_to_change,
                    "recovered_after_commit": True,
                },
            )
        response_path = directory / f"repair-{number}-llm-response.json"
        if not response_path.is_file():
            atomic_write_json(
                response_path,
                {
                    "provider": "durable-patch-recovery",
                    "model": "unknown-after-coordinator-interruption",
                    "request_id": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "wall_seconds": 0.0,
                    "attempts": 1,
                    "schema_valid": True,
                    "value": {"patch": patch_text},
                    "recovered_after_commit": True,
                },
            )
        atomic_write_json(
            update_path,
            {
                "experiment_id": proposal.experiment_id,
                "repair_number": number,
                "previous_commit_sha": request.experiment["commit_sha"],
                "commit_sha": commit_sha,
                "workspace_path": str(workspace),
                "branch_name": request.experiment["branch_name"],
                "config_path": str(repaired_config),
                "config_sha256": sha256_file(repaired_config),
                "changed_paths": list(observed_paths),
                "failure_status": self._latest_repair_failure_status(request).value,
                "recovered_after_commit": True,
            },
        )

    def _existing_live_repair_refs(
        self,
        request: RepairRequest,
        update_path: Path,
    ) -> tuple[ArtifactRef, ...]:
        number = request.plan.repair_number
        directory = update_path.parent
        configs = sorted(directory.glob(f"effective-config-repair-{number}.*"))
        if len(configs) != 1:
            raise RuntimeError("live repair replay has no unique effective config")
        specs: list[tuple[Path, str]] = [
            (directory / f"repair-{number}.diff", "repair_patch"),
            (configs[0], "repaired_experiment_config"),
            (directory / f"repair-{number}-llm-request.json", "llm_request"),
            (directory / f"repair-{number}-llm-response.json", "llm_response"),
            (update_path, "repair_candidate_update"),
        ]
        for name in ("repair-static", "repair-fixture"):
            specs.extend(
                (
                    (directory / f"{name}-sandbox.json", f"{name}_sandbox_evidence"),
                    (directory / f"{name}.stdout.log", f"{name}_stdout"),
                    (directory / f"{name}.stderr.log", f"{name}_stderr"),
                )
            )
            profile = directory / f"{name}-sandbox.sb"
            if profile.is_file():
                specs.append((profile, f"{name}_sandbox_profile"))
        if any(not file_path.is_file() for file_path, _ in specs):
            raise RuntimeError("live repair replay evidence is incomplete")
        refs = tuple(artifact_ref(file_path, kind) for file_path, kind in specs)
        self._validate_live_repair_update(request, update_path)
        return refs

    @staticmethod
    def _validate_live_repair_update(request: RepairRequest, update_path: Path) -> None:
        update = json.loads(update_path.read_text(encoding="utf-8"))
        if update.get("experiment_id") != request.experiment["experiment_id"]:
            raise RuntimeError("live repair update targets a different experiment")
        if int(update.get("repair_number", -1)) != request.plan.repair_number:
            raise RuntimeError("live repair update has a different repair number")
        config = Path(str(update.get("config_path", ""))).resolve()
        if not config.is_file() or sha256_file(config) != update.get("config_sha256"):
            raise RuntimeError("live repair update config is missing or corrupt")
        workspace = Path(str(update.get("workspace_path", ""))).resolve()
        if subprocess_run(["git", "rev-parse", "HEAD"], workspace) != update.get("commit_sha"):
            raise RuntimeError("live repair replay worktree HEAD drifted")
        if subprocess_run(
            ["git", "status", "--porcelain", "--untracked-files=normal"], workspace
        ):
            raise RuntimeError("live repair replay worktree is dirty")

    def _record_replayed_live_repair(
        self,
        request: RepairRequest,
        refs: tuple[ArtifactRef, ...],
    ) -> None:
        request_ref = next(item for item in refs if item.kind == "llm_request")
        response_ref = next(item for item in refs if item.kind == "llm_response")
        response = json.loads(Path(response_ref.path).read_text(encoding="utf-8"))
        repository = self._main_repository(request.context)
        for ref in refs:
            repository.register_artifact(
                ref, experiment_id=str(request.experiment["experiment_id"])
            )
        repository.record_llm_call(
            call_id=(
                f"{request.experiment['experiment_id']}:repair:"
                f"{request.plan.repair_number}:patch"
            ),
            run_id=request.context.run_id,
            experiment_id=str(request.experiment["experiment_id"]),
            role="repair_patch",
            provider=str(response["provider"]),
            model=str(response["model"]),
            request_artifact_id=request_ref.artifact_id,
            response_artifact_id=response_ref.artifact_id,
            schema_valid=bool(response["schema_valid"]),
            input_tokens=int(response["input_tokens"]),
            output_tokens=int(response["output_tokens"]),
            wall_seconds=float(response["wall_seconds"]),
            request_id=response.get("request_id"),
        )

    def _config_for_repair(self, request: RepairRequest) -> Path:
        repository = self._main_repository(request.context)
        experiment = repository.get_experiment(str(request.experiment["experiment_id"]))
        with repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT link.artifact_path,artifact.sha256,artifact.kind FROM artifact_links link "
                "JOIN artifacts artifact ON artifact.artifact_id=link.artifact_id "
                "WHERE link.experiment_id=? AND artifact.kind IN "
                "('repaired_experiment_config','experiment_config') "
                "ORDER BY CASE artifact.kind WHEN 'repaired_experiment_config' THEN 0 ELSE 1 END,"
                "artifact.created_at DESC",
                (experiment["experiment_id"],),
            ).fetchall()
        for row in rows:
            candidate = Path(str(row["artifact_path"])).resolve()
            if (
                row["sha256"] == experiment["config_sha256"]
                and candidate.is_file()
                and sha256_file(candidate) == row["sha256"]
            ):
                return candidate
        raise RuntimeError("repair cannot resolve a durable effective config")

    @staticmethod
    def _workload_repair(config: dict[str, Any]) -> dict[str, Any]:
        update: dict[str, Any] = {"n_jobs": 1}
        for key in ("batch_size", "pair_batch_size", "bce_batch_size"):
            if key in config:
                update[key] = max(128, int(config[key]) // 2)
        if "max_pairs" in config:
            update["max_pairs"] = max(10_000, int(config["max_pairs"]) // 2)
        elif "pair_batch_size" in config:
            update["max_pairs"] = 250_000
        if "n_estimators" in config:
            update["n_estimators"] = max(50, int(config["n_estimators"]) // 2)
        if "num_leaves" in config:
            update["num_leaves"] = max(7, int(config["num_leaves"]) // 2)
        return update

    @staticmethod
    def _numeric_stabilization(config: dict[str, Any]) -> dict[str, Any]:
        update: dict[str, Any] = {"n_jobs": 1}
        if "lr" in config:
            update["lr"] = max(1e-5, float(config["lr"]) * 0.5)
        if "learning_rate" in config:
            update["learning_rate"] = max(
                1e-4, float(config["learning_rate"]) * 0.5
            )
        if "l2" in config:
            update["l2"] = max(1e-6, float(config["l2"]) * 10.0)
        if "reg_lambda" in config:
            update["reg_lambda"] = max(1.0, float(config["reg_lambda"]) * 2.0)
        if "bce_weight" in config:
            update["bce_weight"] = max(0.01, float(config["bce_weight"]))
        return update

    def _latest_repair_failure_status(self, request: RepairRequest) -> AttemptStatus:
        repository = self._main_repository(request.context)
        with repository.database.connect() as connection:
            row = connection.execute(
                "SELECT failure_status FROM experiment_repairs WHERE experiment_id=? "
                "ORDER BY repair_number DESC LIMIT 1",
                (request.experiment["experiment_id"],),
            ).fetchone()
        return AttemptStatus(row["failure_status"]) if row else AttemptStatus.CONTRACT

    def _quarantine_failed_attempt(self, request: RepairRequest) -> Path | None:
        root = (
            request.context.run_dir
            / "attempts"
            / str(request.experiment["experiment_id"])
        )
        candidates = sorted(
            (path for path in root.glob(f"{request.phase}-*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            return None
        source = candidates[-1]
        quarantine = (
            request.context.run_dir
            / "quarantine"
            / str(request.experiment["experiment_id"])
            / f"repair-{request.plan.repair_number}-{source.name}"
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            return quarantine
        quarantine.mkdir(parents=True)
        moved: list[str] = []
        # Preserve the runner's immutable request/result/log evidence and remove
        # only the model outputs that could be corrupt.  The next repair number
        # produces a fresh attempt id and cannot reuse either output directory.
        for name in ("fit-output", "predict-output"):
            output = source / name
            if output.exists():
                output.replace(quarantine / name)
                moved.append(name)
        atomic_write_json(
            quarantine / "QUARANTINED.json",
            {
                "source_attempt": str(source),
                "repair_number": request.plan.repair_number,
                "moved_output_directories": moved,
                "runner_evidence_preserved": True,
            },
        )
        return quarantine

    def _live_patch_repair(
        self,
        request: RepairRequest,
        failure_status: AttemptStatus,
        evidence_path: Path,
    ) -> tuple[ArtifactRef, ...]:
        """Request and gate one constrained patch without proposing a new experiment."""

        proposal = ExperimentProposal.model_validate_json(request.experiment["proposal_json"])
        workspace = GitWorkspace(
            Path(str(request.experiment["workspace_path"])).resolve(),
            str(request.experiment["branch_name"]),
        )
        if subprocess_run(["git", "rev-parse", "HEAD"], workspace.root) != str(
            request.experiment["commit_sha"]
        ):
            raise RuntimeError("live repair worktree is not at the durable candidate commit")
        if subprocess_run(
            ["git", "status", "--porcelain", "--untracked-files=normal"], workspace.root
        ):
            raise RuntimeError("live repair requires a clean candidate worktree")
        decision = CodingService(self.provider).create_patch(
            proposal,
            {
                "repair": True,
                "repair_number": request.plan.repair_number,
                "failure_status": failure_status,
                "phase": request.phase,
                "allowed_files": proposal.files_to_change,
                "instruction": "Fix only the typed failure; preserve the scientific primary change.",
                "test_scored": False,
            },
        )
        patch_text = decision.parsed.patch  # type: ignore[attr-defined]
        changed = workspace.apply(
            patch_text,
            PatchPolicy.from_yaml(self.config.protected_paths),
            proposal.files_to_change,
        )
        audit_changed_files(workspace.root, changed)
        directory = evidence_path.parent
        patch_path = directory / f"repair-{request.plan.repair_number}.diff"
        patch_path.write_text(patch_text, encoding="utf-8")
        gate_refs: list[ArtifactRef] = []
        for name, command in (
            ("repair-static", (self.python_executable, "-m", "compileall", "-q", "src")),
            ("repair-fixture", (self.python_executable, "-m", "pytest", "-q", "tests/fixture")),
        ):
            result = execute_gate(
                name=name,
                command=command,
                workspace=workspace.root,
                artifact_dir=directory,
                timeout_seconds=min(180, self.budget.default_attempt_timeout_seconds),
                sandbox_mode=SandboxMode.PRODUCTION,
                trusted_worktree_root=request.context.run_dir / "worktrees",
                trusted_output_root=request.context.run_dir,
            )
            gate_refs.append(artifact_ref(result.evidence_path, f"{name}_sandbox_evidence"))
            if result.profile_path is not None:
                gate_refs.append(artifact_ref(result.profile_path, f"{name}_sandbox_profile"))
            stdout_path = directory / f"{name}.stdout.log"
            stderr_path = directory / f"{name}.stderr.log"
            stdout_path.write_text(result.stdout, encoding="utf-8")
            stderr_path.write_text(result.stderr, encoding="utf-8")
            gate_refs.extend(
                (
                    artifact_ref(stdout_path, f"{name}_stdout"),
                    artifact_ref(stderr_path, f"{name}_stderr"),
                )
            )
            if not result.successful:
                raise RuntimeError(
                    f"live repair {name} gate failed: {(result.stderr or result.stdout)[-1000:]}"
                )
        new_commit = workspace.commit(
            f"rex: repair {proposal.experiment_id} #{request.plan.repair_number}"
        )
        binding = self.config.method_cards[str(request.experiment["method_card_id"])]
        try:
            relative_config = binding.config_path.relative_to(self.config.project_root)
            worktree_config = workspace.root / relative_config
        except ValueError:
            worktree_config = binding.config_path
        source_config = worktree_config if worktree_config.is_file() else self._config_for_repair(request)
        repaired_config = _copy_atomic(
            source_config,
            directory / f"effective-config-repair-{request.plan.repair_number}{source_config.suffix}",
        )
        request_path = atomic_write_json(
            directory / f"repair-{request.plan.repair_number}-llm-request.json",
            {
                "experiment_id": proposal.experiment_id,
                "failure_status": failure_status,
                "phase": request.phase,
                "allowed_files": proposal.files_to_change,
            },
        )
        response_path = atomic_write_json(
            directory / f"repair-{request.plan.repair_number}-llm-response.json",
            {
                "provider": decision.response.provider,
                "model": decision.response.model,
                "request_id": decision.response.request_id,
                "input_tokens": decision.response.input_tokens,
                "output_tokens": decision.response.output_tokens,
                "wall_seconds": decision.response.wall_seconds,
                "attempts": decision.response.attempts,
                "schema_valid": decision.response.schema_valid,
                "value": decision.response.value,
            },
        )
        update_path = atomic_write_json(
            directory / f"repair-{request.plan.repair_number}-candidate-update.json",
            {
                "experiment_id": proposal.experiment_id,
                "repair_number": request.plan.repair_number,
                "previous_commit_sha": request.experiment["commit_sha"],
                "commit_sha": new_commit,
                "workspace_path": str(workspace.root),
                "branch_name": workspace.branch,
                "config_path": str(repaired_config),
                "config_sha256": sha256_file(repaired_config),
                "changed_paths": list(changed),
                "failure_status": failure_status,
            },
        )
        refs = [
            artifact_ref(patch_path, "repair_patch"),
            artifact_ref(repaired_config, "repaired_experiment_config"),
            artifact_ref(request_path, "llm_request"),
            artifact_ref(response_path, "llm_response"),
            artifact_ref(update_path, "repair_candidate_update"),
            *gate_refs,
        ]
        atomic_write_json(
            evidence_path,
            {
                "repair_number": request.plan.repair_number,
                "phase": request.phase,
                "action": request.plan.action.value,
                "reason": request.plan.reason,
                "overrides": request.plan.overrides,
                "live_patch": True,
                "candidate_update_path": str(update_path),
                "candidate_update_sha256": sha256_file(update_path),
                "test_scored": False,
                "artifacts": [item.model_dump(mode="json") for item in refs],
            },
        )
        refs.append(artifact_ref(evidence_path, "repair_override"))
        repository = self._main_repository(request.context)
        for ref in refs:
            repository.register_artifact(ref, experiment_id=proposal.experiment_id)
        repository.record_llm_call(
            call_id=f"{proposal.experiment_id}:repair:{request.plan.repair_number}:patch",
            run_id=request.context.run_id,
            experiment_id=proposal.experiment_id,
            role="repair_patch",
            provider=decision.response.provider,
            model=decision.response.model,
            request_artifact_id=refs[2].artifact_id,
            response_artifact_id=refs[3].artifact_id,
            schema_valid=decision.response.schema_valid,
            input_tokens=decision.response.input_tokens,
            output_tokens=decision.response.output_tokens,
            wall_seconds=decision.response.wall_seconds,
            request_id=decision.response.request_id,
        )
        return tuple(refs)


def build_scientific_hooks(
    config: ProductionRunConfig,
    provider: StructuredProvider,
    **kwargs: Any,
) -> ProductionScientificHooks:
    """Small CLI-friendly factory kept free of provider configuration logic."""

    return ProductionScientificHooks(config, provider, **kwargs)
