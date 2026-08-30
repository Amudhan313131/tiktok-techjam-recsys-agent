from __future__ import annotations

import os
import shutil

import pytest

from rex.execution.runtime_docker import DockerCLIClient, DockerExecutionRuntime


@pytest.mark.skipif(
    os.environ.get("REX_DOCKER_INTEGRATION") != "1"
    or not os.environ.get("REX_WORKER_IMAGE")
    or not shutil.which("docker"),
    reason="requires an explicitly authorized live Docker controller environment",
)
def test_live_docker_security_doctor() -> None:
    """Run the full disposable isolation/recovery probe when Docker is opt-in."""

    controller_id = (
        os.environ.get("REX_CONTROLLER_ID")
        or os.environ.get("REX_CONTROLLER_CONTAINER_ID")
        or os.environ.get("HOSTNAME")
    )
    assert controller_id
    runtime = DockerExecutionRuntime(
        client=DockerCLIClient(),
        image_reference=os.environ["REX_WORKER_IMAGE"],
        controller_id=controller_id,
    )

    result = runtime.doctor()

    assert result.available
    assert result.safe_for_production, result.detail
    assert all(check.passed for check in result.checks), result.checks
