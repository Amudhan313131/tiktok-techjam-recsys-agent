from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import rex.cli as cli


class FakeSubmissionCoordinator:
    def __init__(self) -> None:
        self.created: tuple[Path, str] | None = None
        self.resumed: str | None = None
        self.handoff_request: tuple[str, Path, str] | None = None
        self.repository = SimpleNamespace(
            get_job=lambda _job_id: {"source_run_id": "production-run"}
        )

    def create(self, database: Path, run_id: str):
        self.created = (database, run_id)
        return {"job_id": "submission-job"}

    def run_until_ready(self, job_id: str):
        self.resumed = job_id
        return {
            "job_id": job_id,
            "state": "READY_FOR_HANDOFF",
            "seal_sha256": "a" * 64,
        }

    def handoff(self, job_id: str, target: Path, *, authorized_seal_sha256: str):
        self.handoff_request = (job_id, target, authorized_seal_sha256)
        return {"job_id": job_id, "state": "HANDED_OFF"}


def _run_args(*, llm: str, authorize_paid_api: bool) -> Namespace:
    return Namespace(
        config=cli.repo_root() / "configs/run/fixture.yaml",
        llm=llm,
        authorize_paid_api=authorize_paid_api,
        allow_paid_api_fallback=False,
        resume=None,
        run_id=None,
        external_deadline_epoch_ms=None,
    )


def test_direct_openai_run_requires_explicit_paid_authorization() -> None:
    with pytest.raises(RuntimeError, match="authorize-paid-api"):
        cli.command_run(_run_args(llm="openai_api", authorize_paid_api=False))


def test_paid_authorization_cannot_be_applied_to_local_provider() -> None:
    with pytest.raises(RuntimeError, match="requires --llm openai_api"):
        cli.command_run(_run_args(llm="codex_cli", authorize_paid_api=True))


def test_authorized_direct_openai_run_passes_provider_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.ProviderRouter, "from_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "run_fixture_autopilot", lambda *args, **kwargs: {"ok": True})

    result = cli.command_run(_run_args(llm="openai_api", authorize_paid_api=True))

    assert result == 0


def test_finalize_requires_explicit_test_prediction_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected(_args):
        nonlocal called
        called = True
        raise AssertionError("submission wiring must not run")

    monkeypatch.setattr(cli, "_submission_coordinator", unexpected)
    args = Namespace(authorize_test_prediction=False)

    with pytest.raises(RuntimeError, match="authorize-test-prediction"):
        cli.command_finalize_submission(args)
    assert called is False


def test_finalize_creates_or_resumes_one_durable_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = FakeSubmissionCoordinator()
    source_database = tmp_path / "state.sqlite3"
    monkeypatch.setattr(
        cli,
        "_submission_coordinator",
        lambda _args: (coordinator, source_database),
    )
    args = Namespace(authorize_test_prediction=True, run_id="production-run")

    result = cli.command_finalize_submission(args)

    assert result == 0
    assert coordinator.created == (source_database, "production-run")
    assert coordinator.resumed == "submission-job"


def test_handoff_requires_one_time_authorization_before_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected(_args):
        nonlocal called
        called = True
        raise AssertionError("handoff wiring must not run")

    monkeypatch.setattr(cli, "_handoff_coordinator", unexpected)
    args = Namespace(authorize_once=False)

    with pytest.raises(RuntimeError, match="authorize-once"):
        cli.command_handoff_submission(args)
    assert called is False


def test_handoff_rejects_malformed_seal_before_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected(_args):
        nonlocal called
        called = True
        raise AssertionError("handoff wiring must not run")

    monkeypatch.setattr(cli, "_handoff_coordinator", unexpected)
    args = Namespace(authorize_once=True, seal_sha256="not-a-digest")

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        cli.command_handoff_submission(args)
    assert called is False


def test_handoff_binds_job_target_and_exact_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = FakeSubmissionCoordinator()
    monkeypatch.setattr(
        cli,
        "_handoff_coordinator",
        lambda _args: coordinator,
    )
    target = tmp_path / "export"
    args = Namespace(
        authorize_once=True,
        job_id="submission-job",
        target_dir=target,
        seal_sha256="b" * 64,
        run_id="production-run",
    )

    result = cli.command_handoff_submission(args)

    assert result == 0
    assert coordinator.handoff_request == ("submission-job", target, "b" * 64)


def test_submission_parser_exposes_all_destructive_authorization_flags() -> None:
    parser = cli.build_parser()
    run = parser.parse_args(["run", "--llm", "openai_api", "--authorize-paid-api"])
    finalize = parser.parse_args(
        ["finalize-submission", "--run-id", "run-1", "--authorize-test-prediction"]
    )
    handoff = parser.parse_args(
        [
            "handoff-submission",
            "--run-id",
            "run-1",
            "--job-id",
            "job-1",
            "--seal-sha256",
            "c" * 64,
            "--target-dir",
            "/tmp/rex-handoff",
            "--authorize-once",
        ]
    )

    assert run.authorize_paid_api is True
    assert finalize.authorize_test_prediction is True
    assert handoff.authorize_once is True


def test_submission_wiring_rejects_any_test_target_before_job_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.sqlite3").write_bytes(b"not opened before manifest gate")
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text(
        '{"splits":{"test":{"row_count":170588,"target_path":"labels.npz",'
        '"target_sha256":"' + "d" * 64 + '"}}}\n',
        encoding="utf-8",
    )
    config = SimpleNamespace(
        runs_dir=tmp_path / "runs",
        data_manifest=manifest,
        project_root=tmp_path,
        environment_lock=tmp_path / "requirements-lock.txt",
    )
    monkeypatch.setattr(cli.ProductionRunConfig, "load", lambda _path: config)
    args = Namespace(
        config=tmp_path / "production.yaml",
        run_id="run-1",
        output_dir=None,
        data_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="test targets"):
        cli._submission_coordinator(args)
