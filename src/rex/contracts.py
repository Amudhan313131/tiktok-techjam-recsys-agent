"""Versioned contracts shared across coordinator, workers, agents, and evaluators."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class RunState(StrEnum):
    INITIALIZING = "INITIALIZING"
    BASELINE_VERIFYING = "BASELINE_VERIFYING"
    SEARCHING = "SEARCHING"
    FINALIZING = "FINALIZING"
    COMPLETE = "COMPLETE"
    BASELINE_BLOCKED = "BASELINE_BLOCKED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    FATAL = "FATAL"


class ExperimentState(StrEnum):
    PROPOSED = "PROPOSED"
    WORKTREE_READY = "WORKTREE_READY"
    PATCHED = "PATCHED"
    STATIC_VALID = "STATIC_VALID"
    FIXTURE_VALID = "FIXTURE_VALID"
    CHEAP_RUNNING = "CHEAP_RUNNING"
    CHEAP_COMPLETE = "CHEAP_COMPLETE"
    FULL_RESERVED = "FULL_RESERVED"
    FULL_RUNNING = "FULL_RUNNING"
    FULL_COMPLETE = "FULL_COMPLETE"
    DIAGNOSED = "DIAGNOSED"
    CONFIRMING = "CONFIRMING"
    CONFIRMED = "CONFIRMED"
    SUBMISSION_BUILDING = "SUBMISSION_BUILDING"
    SUBMISSION_VALID = "SUBMISSION_VALID"
    PROMOTED = "PROMOTED"
    FAILED_REPAIRABLE = "FAILED_REPAIRABLE"
    REPAIRING = "REPAIRING"
    REJECTED = "REJECTED"
    ABANDONED = "ABANDONED"
    FAILED_FINAL = "FAILED_FINAL"


class Operator(StrEnum):
    REPAIR = "REPAIR"
    LOSS = "LOSS"
    FEATURE = "FEATURE"
    SEQUENCE = "SEQUENCE"
    AUX_TASK = "AUX_TASK"
    MODEL_BLOCK = "MODEL_BLOCK"
    HYPERPARAMETER = "HYPERPARAMETER"
    ENSEMBLE = "ENSEMBLE"
    ABANDON = "ABANDON"


class AttemptStatus(StrEnum):
    SUCCESS = "success"
    SYNTAX = "syntax"
    IMPORT = "import"
    CONTRACT = "contract"
    TIMEOUT = "timeout"
    OOM = "oom"
    NAN = "nan"
    CRASH = "crash"
    INVALID_ARTIFACT = "invalid_artifact"
    INTERRUPTED = "interrupted"


class ArtifactRef(StrictModel):
    artifact_id: str
    kind: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    schema_version: str = SCHEMA_VERSION


class Metrics(StrictModel):
    GAUC: float = Field(ge=0.0, le=1.0)
    ndcg5: float = Field(alias="nDCG@5", ge=0.0, le=1.0)
    primary: float = Field(ge=0.0, le=1.0)
    users: int = Field(ge=0)
    rows: int = Field(ge=0)
    evaluator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str
    fold: str | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def validate_primary(self) -> "Metrics":
        expected = (self.GAUC + self.ndcg5) / 2.0
        if abs(self.primary - expected) > 1e-9:
            raise ValueError(f"primary must equal metric mean: expected {expected}")
        return self


class PredictionManifest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    split: Literal["train", "valid", "test", "shadow"]
    experiment_id: str
    commit_sha: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=0)
    seed: int
    feature_cutoff: str | None = None


class ModelBundleMember(StrictModel):
    """One immutable, portable member of a trained-model bundle."""

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("name")
    @classmethod
    def require_relative_member_name(cls, name: str) -> str:
        path = name.replace("\\", "/")
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"unsafe model bundle member name: {name}")
        return path


class ModelBundleManifest(StrictModel):
    """Content-addressed description of everything required to reload a model."""

    schema_version: str = SCHEMA_VERSION
    plugin: str
    seed: int
    commit_sha: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_member: str
    feature_schema: dict[str, str]
    members: list[ModelBundleMember] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_primary_member(self) -> "ModelBundleManifest":
        names = [member.name for member in self.members]
        if len(names) != len(set(names)):
            raise ValueError("model bundle member names must be unique")
        if self.primary_member not in names:
            raise ValueError("model bundle primary_member is not present in members")
        return self


class RunRequest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    experiment_id: str
    attempt_id: str
    parent_id: str | None = None
    commit_sha: str
    plugin: str
    operation: Literal["fit", "predict"] | None = None
    config_path: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    rung: Literal["fixture", "cheap", "full", "confirm", "predict"]
    split: Literal["train", "valid", "test", "shadow"]
    fold: str | None = None
    feature_view_path: str
    target_view_path: str | None = None
    workspace_path: str | None = None
    model_bundle_path: str | None = None
    output_dir: str
    deadline_epoch_ms: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    max_memory_mb: int | None = Field(default=None, gt=0)
    data_view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def enforce_target_capability(self) -> "RunRequest":
        operation = self.operation or ("predict" if self.rung == "predict" else "fit")
        if operation == "predict" and self.rung != "predict":
            raise ValueError("predict operation requires the predict rung")
        if operation == "fit" and self.rung == "predict":
            raise ValueError("predict rung requires the predict operation")
        if self.split in {"valid", "test"} and operation == "predict" and self.target_view_path:
            raise ValueError("inference requests may not receive validation/test targets")
        if operation == "predict" and self.model_bundle_path is None:
            # Legacy requests locate the primary model through their immutable config.
            # New callers must use model_bundle_path and never rewrite the config.
            return self
        return self

    @property
    def effective_operation(self) -> Literal["fit", "predict"]:
        return self.operation or ("predict" if self.rung == "predict" else "fit")


class RunResult(StrictModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    experiment_id: str
    attempt_id: str
    status: AttemptStatus
    exit_code: int | None = None
    signal: int | None = None
    error_type: str | None = None
    error_summary: str | None = None
    command_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    commit_sha: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    wall_seconds: float = Field(ge=0.0)
    cpu_user_seconds: float = Field(default=0.0, ge=0.0)
    cpu_system_seconds: float = Field(default=0.0, ge=0.0)
    peak_rss_bytes: int = Field(default=0, ge=0)
    gpu_seconds: float = Field(default=0.0, ge=0.0)


class PromotionRule(StrictModel):
    min_primary_delta: float = 0.001
    max_gauc_regression: float = 0.002
    max_ndcg5_regression: float = 0.002
    min_positive_shadow_folds: int = Field(default=2, ge=1)


class ExperimentProposal(StrictModel):
    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    parent_id: str | None
    operator: Operator
    hypothesis: str = Field(min_length=12)
    mechanism: str = Field(min_length=12)
    primary_change: str = Field(min_length=5)
    files_to_change: list[str] = Field(min_length=1)
    expected_metric_effects: dict[str, str]
    expected_segment_effects: dict[str, str] = Field(default_factory=dict)
    falsifier: str = Field(min_length=8)
    leakage_analysis: str = Field(min_length=8)
    estimated_seconds: int = Field(gt=0)
    cheap_rung: dict[str, Any]
    full_rung: dict[str, Any]
    promotion_rule: PromotionRule = Field(default_factory=PromotionRule)
    rollback: str = "discard worktree"

    @field_validator("files_to_change")
    @classmethod
    def reject_absolute_paths(cls, paths: list[str]) -> list[str]:
        for path in paths:
            if path.startswith("/") or ".." in path.split("/"):
                raise ValueError(f"unsafe path in proposal: {path}")
        return paths


class Reflection(StrictModel):
    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    outcome: Literal["supported", "contradicted", "inconclusive"]
    evidence_artifact_ids: list[str] = Field(min_length=1)
    metric_deltas: dict[str, float]
    uncertainty: str
    segment_wins: list[str] = Field(default_factory=list)
    segment_regressions: list[str] = Field(default_factory=list)
    leakage_or_proxy_concerns: list[str] = Field(default_factory=list)
    next_operator: Operator
    next_parent_id: str | None = None
    reusable_lesson: str = Field(min_length=8)


class FinalBundle(StrictModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    incumbent_experiment_id: str
    submission: ArtifactRef
    checkpoint: ArtifactRef
    predictions: ArtifactRef
    evidence_index: ArtifactRef
    bundle_manifest: ArtifactRef
    metrics: Metrics
    intervention_count: int = Field(ge=0)
    total_llm_tokens: int = Field(ge=0)
    wall_seconds: float = Field(ge=0.0)
