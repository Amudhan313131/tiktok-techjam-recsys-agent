from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rex.agents.coordinator import PatchRepairsExhausted, PatchTransactionCoordinator
from rex.agents.patch_guard import PatchPolicy
from rex.agents.provider import FakeProvider
from rex.agents.services import CodingService, ProposalService
from rex.contracts import ExperimentState
from rex.control.budget import deadline_epoch_ms
from rex.store.db import Database
from rex.store.repository import ExperimentRepository


HASH = "0" * 64


class InjectedPreparationKill(BaseException):
    pass


def git(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=cwd, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_agent_patch_transaction_commits_in_isolated_worktree(tmp_path: Path) -> None:
    project = tmp_path / "project"
    model = project / "src/rex/models/experimental/model.py"
    fixture = project / "tests/fixture/test_smoke.py"
    model.parent.mkdir(parents=True)
    fixture.parent.mkdir(parents=True)
    model.write_text("VALUE = 1\n", encoding="utf-8")
    fixture.write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    git(project, "init")
    git(project, "config", "user.email", "rex@example.invalid")
    git(project, "config", "user.name", "REX Fixture")
    git(project, "add", "--all")
    git(project, "commit", "-m", "fixture root")
    parent = git(project, "rev-parse", "HEAD")

    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    repository.create_run(
        run_id="run",
        deadline_epoch_ms=deadline_epoch_ms(60),
        root_commit=parent,
        environment_sha256=HASH,
        data_manifest_sha256=HASH,
        evaluator_sha256=HASH,
    )
    proposal = {
        "experiment_id": "accepted-patch",
        "parent_id": None,
        "operator": "HYPERPARAMETER",
        "hypothesis": "Increasing the fixture value verifies isolated autonomous patching.",
        "mechanism": "A declared one-line model edit traverses all preparation gates.",
        "primary_change": "fixture value",
        "files_to_change": ["src/rex/models/experimental/model.py"],
        "expected_metric_effects": {"fixture": "pass"},
        "falsifier": "Static or fixture command fails.",
        "leakage_analysis": "The patch contains no data or target access.",
        "estimated_seconds": 5,
        "cheap_rung": {"fixture": True},
        "full_rung": {"fixture": True},
    }
    patch = {
        "patch": (
            "--- a/src/rex/models/experimental/model.py\n"
            "+++ b/src/rex/models/experimental/model.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        "rationale": "Exercises the accepted patch transaction.",
        "tests": ["fixture value equals two"],
    }
    provider = FakeProvider([proposal, patch])
    coordinator = PatchTransactionCoordinator(
        repository=repository,
        proposal_service=ProposalService(provider),
        coding_service=CodingService(provider),
        project_root=project,
        worktree_root=tmp_path / "worktrees",
        patch_policy=PatchPolicy(
            allowed=("src/rex/models/experimental/**",),
            denied=("src/rex/evaluation/**",),
        ),
        static_command=("python3", "-m", "compileall", "-q", "src"),
        fixture_command=(
            "python3",
            "-c",
            "from pathlib import Path; assert 'VALUE = 2' in Path('src/rex/models/experimental/model.py').read_text()",
        ),
    )
    prepared = coordinator.prepare(
        run_id="run",
        parent_commit=parent,
        proposal_context={"artifact_ids": []},
        coding_context={"artifact_ids": []},
    )
    assert repository.get_experiment("accepted-patch")["state"] == ExperimentState.FIXTURE_VALID
    assert git(prepared.workspace.root, "show", "HEAD:src/rex/models/experimental/model.py") == "VALUE = 2"
    assert prepared.commit_sha != parent


@pytest.mark.parametrize(
    "kill_checkpoint", ["proposal_boundary", "patch_applied", "patched"]
)
def test_patch_transaction_resumes_without_duplicate_provider_calls_or_iteration(
    tmp_path: Path,
    kill_checkpoint: str,
) -> None:
    project = tmp_path / "project"
    model = project / "src/rex/models/experimental/model.py"
    fixture = project / "tests/fixture/test_smoke.py"
    model.parent.mkdir(parents=True)
    fixture.parent.mkdir(parents=True)
    model.write_text("VALUE = 1\n", encoding="utf-8")
    fixture.write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    git(project, "init")
    git(project, "config", "user.email", "rex@example.invalid")
    git(project, "config", "user.name", "REX Resume")
    git(project, "add", "--all")
    git(project, "commit", "-m", "resume root")
    parent = git(project, "rev-parse", "HEAD")
    database = Database(tmp_path / "transaction.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    repository.create_run(
        run_id="run",
        deadline_epoch_ms=deadline_epoch_ms(60),
        root_commit=parent,
        environment_sha256=HASH,
        data_manifest_sha256=HASH,
        evaluator_sha256=HASH,
    )
    proposal = {
        "experiment_id": "resume-patch",
        "parent_id": None,
        "operator": "HYPERPARAMETER",
        "hypothesis": "A durable one-line patch proves preparation crash recovery.",
        "mechanism": "The same proposal and patch continue from durable coordinator state.",
        "primary_change": "fixture value",
        "files_to_change": ["src/rex/models/experimental/model.py"],
        "expected_metric_effects": {"fixture": "pass"},
        "falsifier": "The resumed static or fixture gate fails.",
        "leakage_analysis": "No data or target capability is involved.",
        "estimated_seconds": 5,
        "cheap_rung": {"fixture": True},
        "full_rung": {"fixture": True},
    }
    patch = {
        "patch": (
            "--- a/src/rex/models/experimental/model.py\n"
            "+++ b/src/rex/models/experimental/model.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        "rationale": "The durable patch should be applied exactly once.",
        "tests": ["fixture value equals two"],
    }
    provider = FakeProvider([proposal, patch])
    killed = False

    def checkpoint(name: str, _experiment_id: str) -> None:
        nonlocal killed
        if name == kill_checkpoint and not killed:
            killed = True
            raise InjectedPreparationKill(name)

    def coordinator(checkpoint_hook=None) -> PatchTransactionCoordinator:
        return PatchTransactionCoordinator(
            repository=repository,
            proposal_service=ProposalService(provider),
            coding_service=CodingService(provider),
            project_root=project,
            worktree_root=tmp_path / "worktrees",
            patch_policy=PatchPolicy(allowed=("src/rex/models/experimental/**",), denied=()),
            static_command=("python3", "-m", "compileall", "-q", "src"),
            fixture_command=(
                "python3",
                "-c",
                "from pathlib import Path; assert 'VALUE = 2' in "
                "Path('src/rex/models/experimental/model.py').read_text()",
            ),
            checkpoint=checkpoint_hook,
        )

    with pytest.raises(InjectedPreparationKill):
        coordinator(checkpoint).prepare(
            run_id="run",
            parent_commit=parent,
            proposal_context={"experiment_id": "resume-patch", "artifact_ids": []},
            coding_context={"artifact_ids": []},
        )

    prepared = coordinator().prepare(
        run_id="run",
        parent_commit=parent,
        proposal_context={"experiment_id": "resume-patch", "artifact_ids": []},
        coding_context={"artifact_ids": []},
    )

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
        assert connection.execute("SELECT hypothesis_count FROM runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 2
    assert [item["role"] for item in provider.calls] == ["proposal", "patch"]
    assert git(prepared.workspace.root, "rev-parse", "HEAD") == prepared.commit_sha
    assert git(prepared.workspace.root, "status", "--porcelain") == ""
    assert git(prepared.workspace.root, "show", "HEAD:src/rex/models/experimental/model.py") == "VALUE = 2"


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_unapplicable_live_patch_repairs_are_durable_and_bounded(
    tmp_path: Path, repair_succeeds: bool
) -> None:
    project = tmp_path / "project"
    model = project / "src/rex/models/experimental/model.py"
    fixture = project / "tests/fixture/test_smoke.py"
    model.parent.mkdir(parents=True)
    fixture.parent.mkdir(parents=True)
    model.write_text("VALUE = 1\n", encoding="utf-8")
    fixture.write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    git(project, "init")
    git(project, "config", "user.email", "rex@example.invalid")
    git(project, "config", "user.name", "REX Repair")
    git(project, "add", "--all")
    git(project, "commit", "-m", "repair root")
    parent = git(project, "rev-parse", "HEAD")

    database = Database(tmp_path / "transaction.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    repository.create_run(
        run_id="run",
        deadline_epoch_ms=deadline_epoch_ms(60),
        root_commit=parent,
        environment_sha256=HASH,
        data_manifest_sha256=HASH,
        evaluator_sha256=HASH,
    )
    proposal = {
        "experiment_id": "repaired-patch",
        "parent_id": None,
        "operator": "HYPERPARAMETER",
        "hypothesis": "A corrected diff proves bounded coding repair.",
        "mechanism": "The second patch uses the authoritative source snapshot.",
        "primary_change": "fixture value",
        "files_to_change": ["src/rex/models/experimental/model.py"],
        "expected_metric_effects": {"fixture": "pass"},
        "falsifier": "The corrected patch remains inapplicable.",
        "leakage_analysis": "No data or target capability is involved.",
        "estimated_seconds": 5,
        "cheap_rung": {"fixture": True},
        "full_rung": {"fixture": True},
    }
    bad_patch = {
        "patch": (
            "--- a/src/rex/models/experimental/model.py\n"
            "+++ b/src/rex/models/experimental/model.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 9\n"
            "+VALUE = 2\n"
        ),
        "rationale": "This intentionally mismatches the parent source.",
        "tests": ["fixture value equals two"],
    }
    good_patch = {
        "patch": (
            "--- a/src/rex/models/experimental/model.py\n"
            "+++ b/src/rex/models/experimental/model.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        ),
        "rationale": "Uses the exact authoritative parent line.",
        "tests": ["fixture value equals two"],
    }
    provider = FakeProvider(
        [proposal, bad_patch, good_patch]
        if repair_succeeds
        else [proposal, bad_patch, bad_patch, bad_patch]
    )
    coordinator = PatchTransactionCoordinator(
        repository=repository,
        proposal_service=ProposalService(provider),
        coding_service=CodingService(provider),
        project_root=project,
        worktree_root=tmp_path / "worktrees",
        patch_policy=PatchPolicy(allowed=("src/rex/models/experimental/**",), denied=()),
        fixture_command=(
            "python3",
            "-c",
            "from pathlib import Path; assert 'VALUE = 2' in "
            "Path('src/rex/models/experimental/model.py').read_text()",
        ),
        max_patch_repairs=2,
    )

    def prepare():
        return coordinator.prepare(
            run_id="run",
            parent_commit=parent,
            proposal_context={"experiment_id": "repaired-patch", "artifact_ids": []},
            coding_context={
                "artifact_ids": [],
                "allowed_file_snapshots": {
                    "src/rex/models/experimental/model.py": "VALUE = 1\n"
                },
            },
        )

    if repair_succeeds:
        prepared = prepare()
    else:
        with pytest.raises(PatchRepairsExhausted):
            prepare()

    with database.connect() as connection:
        roles = [
            row[0]
            for row in connection.execute("SELECT role FROM llm_calls ORDER BY rowid")
        ]
        repair = connection.execute(
            "SELECT repair_number,phase,completed_at FROM experiment_repairs"
        ).fetchone()
    if repair_succeeds:
        assert [item["role"] for item in provider.calls] == ["proposal", "patch", "patch"]
        assert roles == ["proposal", "patch-attempt-1", "patch-attempt-2"]
        assert tuple(repair[:2]) == (1, "preparation")
        assert repair[2] is not None
        assert (
            repository.get_experiment("repaired-patch")["state"]
            == ExperimentState.FIXTURE_VALID
        )
        assert git(prepared.workspace.root, "status", "--porcelain") == ""
        assert (
            git(prepared.workspace.root, "show", "HEAD:src/rex/models/experimental/model.py")
            == "VALUE = 2"
        )
    else:
        assert [item["role"] for item in provider.calls] == [
            "proposal",
            "patch",
            "patch",
            "patch",
        ]
        assert roles == [
            "proposal",
            "patch-attempt-1",
            "patch-attempt-2",
            "patch-attempt-3",
        ]
        with database.connect() as connection:
            repairs = connection.execute(
                "SELECT repair_number,completed_at FROM experiment_repairs ORDER BY repair_number"
            ).fetchall()
        assert [row[0] for row in repairs] == [1, 2]
        assert all(row[1] is not None for row in repairs)
        assert (
            repository.get_experiment("repaired-patch")["state"]
            == ExperimentState.FAILED_REPAIRABLE
        )
        worktree = tmp_path / "worktrees" / "repaired-patch"
        assert git(worktree, "status", "--porcelain") == ""
        assert git(worktree, "rev-parse", "HEAD") == parent
