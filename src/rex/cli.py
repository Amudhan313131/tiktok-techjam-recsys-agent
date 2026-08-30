"""Command-line entry point for contracts, data, rehearsals, and reports."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import yaml

from rex.agents.provider import FixedQueueProvider, ProviderRouter
from rex.agents.provider_config import load_provider_config
from rex.control.budget import BudgetConfig, deadline_epoch_ms
from rex.control.fixture_supervisor import (
    FixtureRunConfig,
    FixtureScriptProvider,
    run_fixture_autopilot,
)
from rex.control.production_supervisor import (
    ProductionAutopilot,
    ProductionFixedProvider,
    ProductionRunConfig,
    environment_provenance_sha256,
)
from rex.control.scientific_hooks import build_scientific_hooks
from rex.contracts import RunState
from rex.data.bootstrap import bootstrap_views, default_data_dir
from rex.data.manifest import (
    load_benchmark_manifest,
    repo_root,
    sha256_file,
    verify_starter_manifest,
)
from rex.evaluation.baseline import reproduce_fm_bundle
from rex.evaluation.submission import TEST_ROW_COUNT, build_submission, validate_submission
from rex.execution.runner import execute_request
from rex.execution.runtime import ExecutionRuntimeError, production_runtime
from rex.execution.sandbox import SandboxMode
from rex.models.tree_ranker import tree_ranker_doctor
from rex.rehearsal import (
    rehearsal_requirements,
    run_fixture_rehearsal,
    run_production_rehearsal,
    run_r0,
)
from rex.reporting.report import build_report
from rex.store.db import Database
from rex.store.repository import ExperimentRepository
from rex.submission import (
    FinalSubmissionCoordinator,
    SubmissionDependencies,
    SubmissionJobConfig,
    SubmissionRepository,
)
from rex.submission.repository import discover_completed_source


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _environment_hash(root: Path) -> str:
    provenance = {
        "python": sys.version,
        "platform": sys.platform,
        "files": {
            name: sha256_file(root / name)
            for name in ("requirements-lock.txt", "pyproject.toml")
            if (root / name).is_file()
        },
    }
    return hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resume_openai_usage(
    config: FixtureRunConfig | ProductionRunConfig, run_id: str | None
) -> tuple[int, int]:
    """Rehydrate durable successful API usage before resuming a fixture run.

    Router attempt counts are read from response artifacts. In automatic mode
    they can conservatively include failed providers that preceded OpenAI.
    """

    if run_id is None:
        return 0, 0
    database = Database(config.runs_dir / run_id / "state.sqlite3")
    if not database.path.is_file():
        return 0, 0
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT call.input_tokens,call.output_tokens,artifact.path "
            "FROM llm_calls call LEFT JOIN artifacts artifact "
            "ON artifact.artifact_id=call.response_artifact_id "
            "WHERE call.run_id=? AND call.provider='openai_api'",
            (run_id,),
        ).fetchall()
    calls = 0
    tokens = 0
    for row in rows:
        attempts = 1
        if row["path"]:
            try:
                payload = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
                attempts = max(1, int(payload.get("attempts", 1)))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                attempts = 1
        calls += attempts
        tokens += int(row["input_tokens"]) + int(row["output_tokens"])
    return calls, tokens


def _llm_doctor(args: argparse.Namespace) -> dict[str, object] | None:
    if args.llm is None:
        return None
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    llm = dict(raw.get("llm", {}))
    llm["mode"] = args.llm
    config = load_provider_config(llm)
    checks: dict[str, object] = {"mode": args.llm, "live": bool(args.live)}
    if args.llm in {"codex_cli", "auto"}:
        executable = shutil.which(config.codex_cli.executable)
        compatible = False
        authenticated = False
        if executable is not None:
            try:
                probe = subprocess.run(
                    [executable, "--ask-for-approval", "never", "exec", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                compatible = probe.returncode == 0
                auth_probe = subprocess.run(
                    [executable, "login", "status"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                authenticated = auth_probe.returncode == 0
            except (OSError, subprocess.SubprocessError):
                compatible = False
        checks["codex_cli"] = {
            "executable": executable,
            "available": executable is not None,
            "noninteractive_flags_compatible": compatible,
            "authenticated": authenticated,
        }
        if args.llm == "codex_cli" and not (compatible and authenticated):
            checks["ok"] = False
    if args.llm in {"claude_cli", "auto"}:
        executable = shutil.which(config.claude_cli.executable)
        compatible = False
        authenticated = False
        if executable is not None:
            try:
                probe = subprocess.run(
                    [executable, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                required_flags = (
                    "--print",
                    "--output-format",
                    "--json-schema",
                    "--no-session-persistence",
                    "--tools",
                )
                compatible = probe.returncode == 0 and all(
                    flag in probe.stdout for flag in required_flags
                )
                auth_probe = subprocess.run(
                    [executable, "auth", "status"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                auth_status = json.loads(auth_probe.stdout) if auth_probe.returncode == 0 else {}
                authenticated = bool(
                    isinstance(auth_status, dict) and auth_status.get("loggedIn") is True
                )
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
                compatible = False
        checks["claude_cli"] = {
            "executable": executable,
            "available": executable is not None,
            "noninteractive_flags_compatible": compatible,
            "authenticated": authenticated,
        }
        if args.llm == "claude_cli" and not (compatible and authenticated):
            checks["ok"] = False
    if args.llm in {"openai_api", "auto"}:
        try:
            sdk_version = importlib.metadata.version("openai")
        except importlib.metadata.PackageNotFoundError:
            sdk_version = None
        api_key_present = bool(os.environ.get(config.openai_api.api_key_env))
        model_present = bool(config.openai_api.model or os.environ.get(config.openai_api.model_env))
        checks["openai_api"] = {
            "sdk_version": sdk_version,
            "api_key_present": api_key_present,
            "model_present": model_present,
        }
        if args.llm == "openai_api" and not (sdk_version and api_key_present and model_present):
            checks["ok"] = False
    if args.live and checks.get("ok") is not False:
        fixed_provider = (
            FixedQueueProvider([{"ok": True}]) if args.llm in {"fixed", "auto"} else None
        )
        router = ProviderRouter.from_config(config, fixed_provider=fixed_provider)
        response = router.generate(
            role="doctor",
            system="Return the requested structured health response.",
            prompt='{"request":"provider health check"}',
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
        checks["live_response"] = {
            "provider": response.provider,
            "model": response.model,
            "request_id": response.request_id,
            "schema_valid": response.schema_valid,
            "ok": response.value.get("ok") is True,
        }
        if response.value.get("ok") is not True:
            checks["ok"] = False
    elif args.live:
        checks["live_response"] = {"skipped": "provider prerequisites are not satisfied"}
    checks.setdefault("ok", True)
    return checks


def command_doctor(args: argparse.Namespace) -> int:
    starter = verify_starter_manifest()
    benchmark = load_benchmark_manifest()
    tree = tree_ranker_doctor() if args.tree else None
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw = raw if isinstance(raw, dict) else {}
    production = raw.get("execution_mode") == "production"
    production_enabled = bool(raw.get("scientific_execution_enabled", False))
    manifest_ready: bool | None = None
    sandbox: dict[str, object] | None = None
    production_ok = True
    if production:
        production_config = ProductionRunConfig.load(args.config)
        manifest_ready = production_config.data_manifest.is_file()
        try:
            runtime_result = production_runtime().doctor()
            source_commit = _commit(production_config.project_root)
            image_commit = str(runtime_result.environment_identity.get("source_git_commit", ""))
            source_revision_matches = (
                source_commit != "uncommitted" and source_commit == image_commit
            )
            sandbox = {
                "backend": runtime_result.runtime_kind.value,
                "available": runtime_result.available,
                "safe_for_production": runtime_result.safe_for_production,
                "detail": runtime_result.detail,
                "checks": [
                    {"name": item.name, "passed": item.passed, "detail": item.detail}
                    for item in runtime_result.checks
                ]
                + [
                    {
                        "name": "source_revision_matches_image",
                        "passed": source_revision_matches,
                        "detail": f"source={source_commit}, image={image_commit or 'missing'}",
                    }
                ],
                "environment_identity": dict(runtime_result.environment_identity),
            }
            sandbox["safe_for_production"] = bool(
                runtime_result.safe_for_production and source_revision_matches
            )
        except ExecutionRuntimeError as error:
            sandbox = {
                "backend": "unavailable",
                "available": False,
                "safe_for_production": False,
                "detail": str(error),
            }
        production_ok = bool(
            production_enabled and manifest_ready and sandbox["safe_for_production"]
        )
    llm = (
        _llm_doctor(args)
        if production_ok
        else {
            "ok": False,
            "live_response": {
                "skipped": "production Docker doctor must pass before an LLM request"
            },
        }
    )
    result = {
        "ok": bool((llm or {}).get("ok", True) and (tree or {}).get("ok", True) and production_ok),
        "python": sys.version,
        "label": benchmark["label"],
        "metrics": benchmark["metrics"],
        "starter_manifest_sha256": starter.manifest_sha256,
        "evaluator_sha256": starter.hashes["evaluate.py"],
        "data_present": default_data_dir().is_dir(),
        "production_science_enabled": production_enabled,
        "production_data_manifest_ready": manifest_ready,
        "production_sandbox": sandbox,
        "llm": llm,
        "tree": tree,
    }
    _json(result)
    return 0 if result["ok"] else 2


def command_run(args: argparse.Namespace) -> int:
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("run configuration must be a YAML mapping")
    llm = dict(raw.get("llm", {}))
    if args.llm is not None:
        llm["mode"] = args.llm
    if llm.get("mode") == "openai_api" and not getattr(args, "authorize_paid_api", False):
        raise RuntimeError("direct OpenAI API use requires the explicit --authorize-paid-api flag")
    if getattr(args, "authorize_paid_api", False) and llm.get("mode") != "openai_api":
        raise RuntimeError("--authorize-paid-api requires --llm openai_api")
    if args.allow_paid_api_fallback:
        if llm.get("mode") != "auto":
            raise RuntimeError("--allow-paid-api-fallback requires --llm auto")
        auto = dict(llm.get("auto", {}))
        auto["allow_paid_api_fallback"] = True
        llm["auto"] = auto
    provider_config = load_provider_config(llm)
    execution_mode = raw.get("execution_mode")
    if execution_mode == "production":
        production_config = ProductionRunConfig.load(args.config)
        try:
            runtime_result = production_runtime().doctor()
        except ExecutionRuntimeError as error:
            raise RuntimeError(f"production Docker runtime is unavailable: {error}") from error
        if not runtime_result.safe_for_production:
            failed = "; ".join(
                f"{item.name}: {item.detail}" for item in runtime_result.checks if not item.passed
            )
            raise RuntimeError(f"production Docker doctor failed closed: {failed}")
        source_commit = _commit(production_config.project_root)
        image_commit = str(runtime_result.environment_identity.get("source_git_commit", ""))
        if source_commit == "uncommitted" or image_commit != source_commit:
            raise RuntimeError(
                "production source revision does not match the verified worker image: "
                f"source={source_commit}, image={image_commit or 'missing'}"
            )
        production_config = replace(
            production_config,
            llm=llm,
            runtime_environment_identity=dict(runtime_result.environment_identity),
        )
        if provider_config.mode == "auto":
            provider_config = replace(
                provider_config,
                auto=replace(
                    provider_config.auto,
                    provider_order=tuple(
                        item for item in provider_config.auto.provider_order if item != "fixed"
                    ),
                ),
            )
        prior_openai_calls, prior_openai_tokens = _resume_openai_usage(
            production_config, args.resume
        )
        provider = ProviderRouter.from_config(
            provider_config,
            fixed_provider=(ProductionFixedProvider() if provider_config.mode == "fixed" else None),
            initial_openai_calls=prior_openai_calls,
            initial_openai_tokens=prior_openai_tokens,
        )
        hooks = build_scientific_hooks(production_config, provider)
        selected_run_id = args.resume or args.run_id
        result = ProductionAutopilot(production_config, provider, hooks).run(
            run_id=selected_run_id,
            create_only=args.run_id is not None,
            resume_only=args.resume is not None,
            external_deadline_epoch_ms=args.external_deadline_epoch_ms,
        )
        _json(result)
        return 0
    if execution_mode != "fixture":
        raise RuntimeError(f"unsupported execution_mode: {execution_mode!r}")
    fixture_config = FixtureRunConfig.load(args.config)
    prior_openai_calls, prior_openai_tokens = _resume_openai_usage(fixture_config, args.resume)
    provider = ProviderRouter.from_config(
        provider_config,
        fixed_provider=FixtureScriptProvider(),
        initial_openai_calls=prior_openai_calls,
        initial_openai_tokens=prior_openai_tokens,
    )
    if args.external_deadline_epoch_ms is not None:
        raise RuntimeError("external deadline is supported only for production R3 runs")
    result = run_fixture_autopilot(
        fixture_config.source_path,
        provider=provider,
        run_id=args.resume or args.run_id,
    )
    _json(result)
    return 0


def command_status(args: argparse.Namespace) -> int:
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("execution_mode") == "production":
        config = ProductionRunConfig.load(args.config)
        autopilot = ProductionAutopilot(config, ProductionFixedProvider())
        _json(
            autopilot.compact_status(args.run_id) if args.compact else autopilot.status(args.run_id)
        )
        return 0
    config = FixtureRunConfig.load(args.config)
    database = Database(config.runs_dir / args.run_id / "state.sqlite3")
    if not database.path.is_file():
        raise RuntimeError(f"unknown fixture run: {args.run_id}")
    with database.connect() as connection:
        run = connection.execute("SELECT * FROM runs WHERE run_id=?", (args.run_id,)).fetchone()
        if run is None:
            raise RuntimeError(f"unknown fixture run: {args.run_id}")
        experiments = connection.execute(
            "SELECT experiment_id,iteration_number,state,operator FROM experiments "
            "WHERE run_id=? ORDER BY iteration_number",
            (args.run_id,),
        ).fetchall()
        sessions = connection.execute(
            "SELECT session_id,pid,started_at,ended_at,exit_reason FROM process_sessions "
            "WHERE run_id=? ORDER BY started_at",
            (args.run_id,),
        ).fetchall()
    _json(
        {
            "run": dict(run),
            "experiments": [dict(row) for row in experiments],
            "sessions": [dict(row) for row in sessions],
            "production_science_enabled": False,
        }
    )
    return 0


def command_bootstrap(args: argparse.Namespace) -> int:
    manifest = bootstrap_views(args.data_dir, args.output_dir)
    _json(manifest)
    return 0


def command_baseline(args: argparse.Namespace) -> int:
    bundle = reproduce_fm_bundle(
        args.data_dir,
        args.view_dir,
        seeds=tuple(int(seed) for seed in args.seeds.split(",")),
    )
    _json(
        {
            "mean_primary": bundle.mean_primary,
            "std_primary": bundle.std_primary,
            "results": [
                {
                    "seed": item.seed,
                    "best_epoch": item.best_epoch,
                    "metrics": item.metrics.model_dump(mode="json", by_alias=True),
                }
                for item in bundle.results
            ],
        }
    )
    return 0


def command_init_run(args: argparse.Namespace) -> int:
    root = repo_root()
    config = BudgetConfig.from_yaml(args.config)
    starter = verify_starter_manifest()
    database = Database(args.database)
    database.initialize()
    repository = ExperimentRepository(database)
    run_id = args.run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    repository.create_run(
        run_id=run_id,
        deadline_epoch_ms=deadline_epoch_ms(config.wall_clock_seconds),
        root_commit=_commit(root),
        environment_sha256=_environment_hash(root),
        data_manifest_sha256=sha256_file(args.data_manifest),
        evaluator_sha256=starter.hashes["evaluate.py"],
    )
    repository.transition_run(run_id, RunState.INITIALIZING, RunState.BASELINE_VERIFYING)
    _json(repository.get_run(run_id))
    return 0


def command_report(args: argparse.Namespace) -> int:
    _json(build_report(Database(args.database), args.run_id, args.output_dir))
    return 0


def _submission_coordinator(
    args: argparse.Namespace,
) -> tuple[FinalSubmissionCoordinator, Path]:
    """Wire production-safe adapters without granting the model test labels."""

    config = ProductionRunConfig.load(args.config)
    run_dir = (config.runs_dir / args.run_id).resolve()
    source_database = run_dir / "state.sqlite3"
    if not source_database.is_file():
        raise RuntimeError(f"unknown production run: {args.run_id}")
    try:
        data_manifest = json.loads(config.data_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid production data manifest: {error}") from error
    if not isinstance(data_manifest, dict):
        raise RuntimeError("production data manifest must be a JSON object")
    test = data_manifest.get("splits", {}).get("test", {})
    if not isinstance(test, dict):
        raise RuntimeError("production data manifest has no test split")
    if test.get("target_path") is not None or test.get("target_sha256") is not None:
        raise RuntimeError("test submission refuses a manifest containing test targets")
    if int(test.get("row_count", -1)) != TEST_ROW_COUNT:
        raise RuntimeError(f"test split must contain exactly {TEST_ROW_COUNT:,} rows")
    feature_path = Path(str(test.get("feature_path", ""))).resolve(strict=True)
    feature_sha256 = str(test.get("feature_sha256", ""))
    if sha256_file(feature_path) != feature_sha256:
        raise RuntimeError("canonical test feature view has drifted")
    source = discover_completed_source(source_database, args.run_id)
    if source.source_run["data_manifest_sha256"] != sha256_file(config.data_manifest):
        raise RuntimeError("production run was completed against a different data manifest")
    try:
        runtime = production_runtime()
        runtime_result = runtime.doctor()
    except ExecutionRuntimeError as error:
        raise RuntimeError(f"final-prediction Docker runtime is unavailable: {error}") from error
    if not runtime_result.safe_for_production:
        failed = "; ".join(
            f"{item.name}: {item.detail}" for item in runtime_result.checks if not item.passed
        )
        raise RuntimeError(f"final-prediction Docker doctor failed closed: {failed}")
    current_identity = dict(runtime_result.environment_identity)
    identity_path = source.source_report_path / "environment_identity.json"
    try:
        winning_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"winning run has no valid Docker environment identity: {error}"
        ) from error
    if not isinstance(winning_identity, dict):
        raise RuntimeError("winning run Docker environment identity must be an object")
    identity_drift = {
        key: {"winning": winning_identity.get(key), "current": value}
        for key, value in sorted(current_identity.items())
        if winning_identity.get(key) != value
    }
    if identity_drift:
        raise RuntimeError(
            "final prediction must use the winning Docker image and platform: "
            + json.dumps(identity_drift, sort_keys=True)
        )
    config = replace(config, runtime_environment_identity=current_identity)
    if environment_provenance_sha256(config) != source.source_run["environment_sha256"]:
        raise RuntimeError("final-prediction environment differs from the completed winning run")

    output_root = _submission_output_root(args, run_dir, source_database)
    jobs_root = output_root / "jobs"
    repository = SubmissionRepository(output_root / "state.sqlite3")
    repository.initialize()

    def predict(request, attempt_dir):
        return execute_request(
            request,
            attempt_dir,
            trusted_worktree_root=jobs_root,
            sandbox_mode=SandboxMode.PRODUCTION,
            trusted_output_root=jobs_root,
            execution_runtime=runtime,
        )

    def build_csv(prediction_path, csv_path, expected_features, expected_rows):
        return build_submission(
            prediction_path,
            csv_path,
            expected_features=expected_features,
            expected_rows=expected_rows,
        )

    checker = functools.partial(
        validate_submission,
        data_dir=Path(args.data_dir).resolve(strict=True),
        split="test",
        sandbox_mode=SandboxMode.PRODUCTION,
        execution_runtime=runtime,
    )
    coordinator = FinalSubmissionCoordinator(
        repository,
        SubmissionJobConfig(
            repository_root=config.project_root,
            jobs_root=jobs_root,
            test_feature_path=feature_path,
            test_data_view_sha256=feature_sha256,
            environment_sha256=environment_provenance_sha256(config),
            expected_test_rows=TEST_ROW_COUNT,
        ),
        SubmissionDependencies(
            predictor=predict,
            csv_builder=build_csv,
            checker=checker,
        ),
    )
    return coordinator, source_database


def _submission_output_root(args: argparse.Namespace, run_dir: Path, source_database: Path) -> Path:
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (run_dir / "submission").resolve()
    )
    if output_root == run_dir or output_root in run_dir.parents:
        raise RuntimeError("submission output may not replace the production run or its parent")
    for protected in (source_database, run_dir / "best-valid", run_dir / "report"):
        protected = protected.resolve()
        if output_root == protected or protected in output_root.parents:
            raise RuntimeError(f"submission output overlaps immutable run evidence: {protected}")
    return output_root


def _handoff_coordinator(args: argparse.Namespace) -> FinalSubmissionCoordinator:
    """Open only sealed submission state; handoff needs no test-data capability."""

    config = ProductionRunConfig.load(args.config)
    run_dir = (config.runs_dir / args.run_id).resolve()
    source_database = run_dir / "state.sqlite3"
    output_root = _submission_output_root(args, run_dir, source_database)
    state_path = output_root / "state.sqlite3"
    if not state_path.is_file():
        raise RuntimeError(f"submission state does not exist: {state_path}")
    repository = SubmissionRepository(state_path)
    repository.initialize()

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("handoff command has no prediction, data, or checker capability")

    coordinator = FinalSubmissionCoordinator(
        repository,
        SubmissionJobConfig(
            repository_root=config.project_root,
            jobs_root=output_root / "jobs",
            test_feature_path=output_root / "no-test-feature-capability",
            test_data_view_sha256="0" * 64,
            environment_sha256="0" * 64,
            expected_test_rows=TEST_ROW_COUNT,
        ),
        SubmissionDependencies(
            predictor=unavailable,
            csv_builder=unavailable,
            checker=unavailable,
        ),
    )
    return coordinator


def command_finalize_submission(args: argparse.Namespace) -> int:
    if not args.authorize_test_prediction:
        raise RuntimeError(
            "final test prediction requires the explicit --authorize-test-prediction flag"
        )
    coordinator, source_database = _submission_coordinator(args)
    job = coordinator.create(source_database, args.run_id)
    result = coordinator.run_until_ready(str(job["job_id"]))
    _json(result)
    return 0


def command_handoff_submission(args: argparse.Namespace) -> int:
    if not args.authorize_once:
        raise RuntimeError("submission handoff requires the explicit --authorize-once flag")
    if len(args.seal_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.seal_sha256
    ):
        raise ValueError("--seal-sha256 must be a lowercase SHA-256 digest")
    coordinator = _handoff_coordinator(args)
    job = coordinator.repository.get_job(args.job_id)
    if job["source_run_id"] != args.run_id:
        raise RuntimeError("submission job belongs to a different production run")
    result = coordinator.handoff(
        args.job_id,
        args.target_dir,
        authorized_seal_sha256=args.seal_sha256,
    )
    _json(result)
    return 0


def command_rehearse(args: argparse.Namespace) -> int:
    requirements = rehearsal_requirements(args.level)
    if args.level.upper() == "R0":
        result = run_r0(args.output_dir)
        _json({"requirements": requirements, **result})
        return 0 if result["event_chain_valid"] else 1
    if args.level.upper() in {"R1", "R2"}:
        result = run_production_rehearsal(args.level, args.output_dir)
        _json({"requirements": requirements, **result})
        return 0 if result["all_cases_passed"] else 1
    result = run_fixture_rehearsal(args.config, args.output_dir)
    _json({"requirements": requirements, **result})
    checks = (
        result["event_chain_valid"],
        result["provider_interruption_recovered"],
        result["worker_nan_recovered"],
        result["worker_repair_limit_enforced"],
        result["protected_patch_rejected"],
        result["production_promotion_blocked"],
    )
    return 0 if all(checks) else 1


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(prog="rex")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="verify frozen contracts and local prerequisites")
    doctor.add_argument("--llm", choices=["codex_cli", "claude_cli", "openai_api", "auto", "fixed"])
    doctor.add_argument("--live", action="store_true", help="perform an explicit live LLM call")
    doctor.add_argument("--tree", action="store_true", help="run the synthetic LightGBM doctor")
    doctor.add_argument("--config", type=Path, default=root / "configs/run/fixture.yaml")
    doctor.set_defaults(handler=command_doctor)
    run = sub.add_parser("run", help="run or resume the configured autonomous control plane")
    run.add_argument("--config", type=Path, default=root / "configs/run/fixture.yaml")
    run_identity = run.add_mutually_exclusive_group()
    run_identity.add_argument("--run-id", metavar="RUN_ID", help="create one named run")
    run_identity.add_argument("--resume", metavar="RUN_ID", help="resume one existing run")
    run.add_argument(
        "--external-deadline-epoch-ms",
        type=int,
        help="R3 envelope deadline; may only shorten the configured wall ceiling",
    )
    run.add_argument("--llm", choices=["codex_cli", "claude_cli", "openai_api", "auto", "fixed"])
    run.add_argument(
        "--allow-paid-api-fallback",
        action="store_true",
        help="allow auto mode to fall back from local CLIs to the paid OpenAI API",
    )
    run.add_argument(
        "--authorize-paid-api",
        action="store_true",
        help="explicitly authorize direct paid OpenAI API use for this run invocation",
    )
    run.set_defaults(handler=command_run)
    status = sub.add_parser("status", help="show durable run state")
    status.add_argument("--run-id", required=True)
    status.add_argument("--config", type=Path, default=root / "configs/run/fixture.yaml")
    status.add_argument(
        "--compact", action="store_true", help="emit a small hourly-monitoring snapshot"
    )
    status.set_defaults(handler=command_status)
    bootstrap = sub.add_parser("bootstrap", help="build sanitized data views and target vault")
    bootstrap.add_argument("--data-dir", type=Path, default=default_data_dir())
    bootstrap.add_argument("--output-dir", type=Path, default=root / "runs/data")
    bootstrap.set_defaults(handler=command_bootstrap)
    baseline = sub.add_parser("baseline", help="reproduce valid-only official FM seeds")
    baseline.add_argument("--data-dir", type=Path, default=default_data_dir())
    baseline.add_argument("--view-dir", type=Path, default=root / "runs/data")
    baseline.add_argument("--seeds", default="0,1,2,3,4")
    baseline.set_defaults(handler=command_baseline)
    init_run = sub.add_parser("init-run", help="create a transactional run record")
    init_run.add_argument("--database", type=Path, required=True)
    init_run.add_argument("--data-manifest", type=Path, required=True)
    init_run.add_argument("--config", type=Path, default=root / "configs/budget.yaml")
    init_run.add_argument("--run-id")
    init_run.set_defaults(handler=command_init_run)
    report = sub.add_parser("report", help="export the evidence bundle")
    report.add_argument("--database", type=Path, required=True)
    report.add_argument("--run-id", required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.set_defaults(handler=command_report)
    finalize = sub.add_parser(
        "finalize-submission",
        help="create and check one test submission from a completed production winner",
    )
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--config", type=Path, default=root / "configs/run/production.yaml")
    finalize.add_argument("--data-dir", type=Path, default=default_data_dir())
    finalize.add_argument(
        "--output-dir",
        type=Path,
        help="submission state/output root; defaults to runs/<run-id>/submission",
    )
    finalize.add_argument(
        "--authorize-test-prediction",
        action="store_true",
        help="explicitly authorize the single resumable prediction over the test feature view",
    )
    finalize.set_defaults(handler=command_finalize_submission)
    handoff = sub.add_parser(
        "handoff-submission",
        help="copy one sealed submission bundle to its explicitly authorized handoff path",
    )
    handoff.add_argument("--run-id", required=True)
    handoff.add_argument("--job-id", required=True)
    handoff.add_argument("--seal-sha256", required=True)
    handoff.add_argument("--target-dir", type=Path, required=True)
    handoff.add_argument("--config", type=Path, default=root / "configs/run/production.yaml")
    handoff.add_argument(
        "--output-dir",
        type=Path,
        help="submission state/output root; defaults to runs/<run-id>/submission",
    )
    handoff.add_argument(
        "--authorize-once",
        action="store_true",
        help="authorize one filesystem handoff of this exact seal",
    )
    handoff.set_defaults(handler=command_handoff_submission)
    rehearse = sub.add_parser("rehearse", help="run a bounded offline control-plane rehearsal")
    rehearse.add_argument(
        "--level",
        choices=["R0", "R1", "R2", "FIXTURE", "r0", "r1", "r2", "fixture"],
        default="FIXTURE",
    )
    rehearse.add_argument("--output-dir", type=Path, default=root / "runs/rehearsal-fixture")
    rehearse.add_argument("--config", type=Path, default=root / "configs/run/fixture.yaml")
    rehearse.set_defaults(handler=command_rehearse)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (RuntimeError, ValueError, OSError) as error:
        print(f"rex: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
