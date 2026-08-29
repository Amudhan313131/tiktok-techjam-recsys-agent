"""Command-line entry point for contracts, data, rehearsals, and reports."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
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
from rex.contracts import RunState
from rex.data.bootstrap import bootstrap_views, default_data_dir
from rex.data.manifest import load_benchmark_manifest, repo_root, sha256_file, verify_starter_manifest
from rex.evaluation.baseline import reproduce_fm_bundle
from rex.models.tree_ranker import tree_ranker_doctor
from rex.rehearsal import rehearsal_requirements, run_fixture_rehearsal, run_r0
from rex.reporting.report import build_report
from rex.store.db import Database
from rex.store.repository import ExperimentRepository


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _environment_hash(root: Path) -> str:
    return sha256_file(root / "requirements.txt")


def _resume_openai_usage(config: FixtureRunConfig, run_id: str | None) -> tuple[int, int]:
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
    llm = _llm_doctor(args)
    tree = tree_ranker_doctor() if args.tree else None
    result = {
        "ok": bool((llm or {}).get("ok", True) and (tree or {}).get("ok", True)),
        "python": sys.version,
        "label": benchmark["label"],
        "metrics": benchmark["metrics"],
        "starter_manifest_sha256": starter.manifest_sha256,
        "evaluator_sha256": starter.hashes["evaluate.py"],
        "data_present": default_data_dir().is_dir(),
        "production_science_enabled": False,
        "llm": llm,
        "tree": tree,
    }
    _json(result)
    return 0 if result["ok"] else 2


def command_run(args: argparse.Namespace) -> int:
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fixture_config = FixtureRunConfig.load(args.config)
    llm = dict(raw.get("llm", {}))
    if args.llm is not None:
        llm["mode"] = args.llm
    if args.allow_paid_api_fallback:
        if llm.get("mode") != "auto":
            raise RuntimeError("--allow-paid-api-fallback requires --llm auto")
        auto = dict(llm.get("auto", {}))
        auto["allow_paid_api_fallback"] = True
        llm["auto"] = auto
    provider_config = load_provider_config(llm)
    prior_openai_calls, prior_openai_tokens = _resume_openai_usage(
        fixture_config, args.resume
    )
    provider = ProviderRouter.from_config(
        provider_config,
        fixed_provider=FixtureScriptProvider(),
        initial_openai_calls=prior_openai_calls,
        initial_openai_tokens=prior_openai_tokens,
    )
    result = run_fixture_autopilot(
        fixture_config.source_path,
        provider=provider,
        run_id=args.resume,
    )
    _json(result)
    return 0


def command_status(args: argparse.Namespace) -> int:
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


def command_rehearse(args: argparse.Namespace) -> int:
    requirements = rehearsal_requirements(args.level)
    if args.level.upper() == "R0":
        result = run_r0(args.output_dir)
        _json({"requirements": requirements, **result})
        return 0 if result["event_chain_valid"] else 1
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
    doctor.add_argument(
        "--llm", choices=["codex_cli", "claude_cli", "openai_api", "auto", "fixed"]
    )
    doctor.add_argument("--live", action="store_true", help="perform an explicit live LLM call")
    doctor.add_argument("--tree", action="store_true", help="run the synthetic LightGBM doctor")
    doctor.add_argument("--config", type=Path, default=root / "configs/run/fixture.yaml")
    doctor.set_defaults(handler=command_doctor)
    run = sub.add_parser("run", help="run or resume the fixture-only autonomous loop")
    run.add_argument("--config", type=Path, default=root / "configs/run/fixture.yaml")
    run.add_argument("--resume", metavar="RUN_ID")
    run.add_argument(
        "--llm", choices=["codex_cli", "claude_cli", "openai_api", "auto", "fixed"]
    )
    run.add_argument(
        "--allow-paid-api-fallback",
        action="store_true",
        help="allow auto mode to fall back from local CLIs to the paid OpenAI API",
    )
    run.set_defaults(handler=command_run)
    status = sub.add_parser("status", help="show durable fixture-run state")
    status.add_argument("--run-id", required=True)
    status.add_argument("--config", type=Path, default=root / "configs/run/fixture.yaml")
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
    rehearse = sub.add_parser("rehearse", help="run a short generated-fixture rehearsal")
    rehearse.add_argument(
        "--level", choices=["R0", "FIXTURE", "r0", "fixture"], default="FIXTURE"
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
