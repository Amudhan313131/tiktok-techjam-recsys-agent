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


class ValidationPhase(StrEnum):
    """One-way validation lifecycle for production model selection.

    Shadow data is the only source of discovery feedback.  Exactly one finalist
    can then consume the official-validation capability, after which the phase
    is terminal for the run.
    """

    DISCOVERY = "DISCOVERY"
    FINALIST_LOCKED = "FINALIST_LOCKED"
    OFFICIAL_EVALUATED = "OFFICIAL_EVALUATED"


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
    OFFICIAL_VALID_RUNNING = "OFFICIAL_VALID_RUNNING"
    OFFICIAL_VALID_COMPLETE = "OFFICIAL_VALID_COMPLETE"
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


class RehearsalR3Spec(StrictModel):
    """Immutable envelope for a clean, validation-only dress rehearsal."""

    schema_version: str = SCHEMA_VERSION
    rehearsal_id: str = Field(min_length=1)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    source_commit: str = Field(min_length=7)
    repository: str = Field(min_length=1)
    data_dir: str = Field(min_length=1)
    artifact_root: str = Field(min_length=1)
    provider_mode: Literal["codex_cli", "claude_cli", "openai_api", "auto"]
    allow_paid_api: bool = False
    wall_seconds: int = Field(default=21_600, gt=0, le=21_600)
    finalization_reserve_seconds: int = Field(default=1_200, ge=0)
    inject_controlled_failure: bool = True
    expected_test_rows: int = Field(default=170_588, ge=1)

    @model_validator(mode="after")
    def validate_r3_policy(self) -> "RehearsalR3Spec":
        if self.finalization_reserve_seconds >= self.wall_seconds:
            raise ValueError("R3 finalization reserve must be smaller than the wall ceiling")
        if self.provider_mode == "openai_api" and not self.allow_paid_api:
            raise ValueError("OpenAI API mode requires explicit paid-API authorization")
        return self


class RehearsalR3Manifest(StrictModel):
    """Sealed evidence produced by the outer R3 launcher."""

    schema_version: str = SCHEMA_VERSION
    level: Literal["R3"] = "R3"
    rehearsal_id: str
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    state: Literal["COMPLETE"]
    stop_reason: str | None = None
    started_epoch_ms: int = Field(gt=0)
    deadline_epoch_ms: int = Field(gt=0)
    elapsed_seconds: float = Field(ge=0.0)
    source_commit: str
    source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    starter_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_requested: Literal["codex_cli", "claude_cli", "openai_api", "auto"]
    provider_actual: str = Field(min_length=1)
    fault_injected: Literal[True]
    fault_recovered: Literal[True]
    source_unchanged: Literal[True]
    best_valid_manifest: ArtifactRef
    report_artifacts: list[ArtifactRef] = Field(min_length=1)
    test_prediction_created: Literal[False] = False
    test_scored: Literal[False] = False
    submission_created: Literal[False] = False
    started_at: str
    completed_at: str
    wall_clock_ceiling_seconds: int = Field(gt=0, le=21_600)
    within_six_hour_ceiling: Literal[True]
    llm: str
    provider_calls: list[dict[str, Any]] = Field(min_length=1)
    paid_api_authorized: bool
    dependency: dict[str, Any]
    preflight: dict[str, Any]
    controlled_failure: dict[str, Any]
    source_audit: dict[str, Any]
    clone_audit: dict[str, Any]
    validation: dict[str, Any]
    winner: dict[str, Any]
    status: dict[str, Any]
    hourly_snapshot_count: int = Field(ge=1)
    evidence: dict[str, dict[str, Any]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_completed_envelope(self) -> "RehearsalR3Manifest":
        if self.deadline_epoch_ms <= self.started_epoch_ms:
            raise ValueError("R3 deadline must be later than its start")
        if self.elapsed_seconds > self.wall_clock_ceiling_seconds:
            raise ValueError("R3 elapsed time exceeds its declared ceiling")
        if self.provider_actual == "fixed" or "fixed" in self.provider_actual.split(","):
            raise ValueError("R3 must use an authorized live researcher provider")
        return self


class SubmissionJobState(StrEnum):
    CREATED = "CREATED"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    WORKTREE_READY = "WORKTREE_READY"
    PREDICTING = "PREDICTING"
    PREDICTED = "PREDICTED"
    CSV_BUILT = "CSV_BUILT"
    FIRST_CHECK_VALID = "FIRST_CHECK_VALID"
    STAGING = "STAGING"
    SECOND_CHECK_VALID = "SECOND_CHECK_VALID"
    SEALED = "SEALED"
    READY_FOR_HANDOFF = "READY_FOR_HANDOFF"
    HANDOFF_IN_PROGRESS = "HANDOFF_IN_PROGRESS"
    HANDED_OFF = "HANDED_OFF"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class SubmissionCheckEvidence(StrictModel):
    ordinal: Literal[1, 2]
    command: list[str]
    checker_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    csv_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stderr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    returncode: int
    valid: bool

    @model_validator(mode="after")
    def prohibit_test_scoring(self) -> "SubmissionCheckEvidence":
        forbidden = {"--score", "--make"}
        if forbidden.intersection(self.command):
            raise ValueError("submission checker evidence may not contain score/make operations")
        if self.command.count("--check") != 1:
            raise ValueError("submission checker command must contain --check exactly once")
        return self


class FinalSubmissionSpec(StrictModel):
    schema_version: str = SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    source_run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    best_valid_manifest_path: str = Field(min_length=1)
    best_valid_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_feature_path: str = Field(min_length=1)
    test_feature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_dir: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    expected_rows: int = Field(default=170_588, ge=1)
    authorize_test_prediction: bool = False

    @model_validator(mode="after")
    def require_test_authorization(self) -> "FinalSubmissionSpec":
        if not self.authorize_test_prediction:
            raise ValueError("final submission requires explicit test-prediction authorization")
        return self


class FinalSubmissionManifest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    job_id: str
    source_run_id: str
    source_experiment_id: str
    commit_sha: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_feature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_rows: int = Field(default=170_588, ge=1)
    artifacts: dict[str, ArtifactRef]
    checks: tuple[SubmissionCheckEvidence, SubmissionCheckEvidence]
    handoff_id: str
    test_scored: Literal[False] = False


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
