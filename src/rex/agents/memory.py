"""Evidence-bound scientific memory stored through the transactional repository."""

from __future__ import annotations

import uuid

from rex.contracts import Reflection
from rex.store.repository import ExperimentRepository


def remember_reflection(repository: ExperimentRepository, run_id: str, reflection: Reflection) -> str:
    lesson_id = f"lesson-{uuid.uuid4().hex}"
    repository.record_lesson(
        lesson_id=lesson_id,
        run_id=run_id,
        experiment_id=reflection.experiment_id,
        scope=reflection.next_operator.value,
        lesson=reflection.reusable_lesson,
        evidence_artifact_ids=reflection.evidence_artifact_ids,
    )
    return lesson_id
