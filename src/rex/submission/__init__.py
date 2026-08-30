"""Crash-safe, explicitly authorized final-submission jobs.

The production search database is an immutable input to this package.  Submission
state is kept in a separate SQLite database so generating a test prediction can
never rewrite the research history that selected the winning model.
"""

from rex.submission.coordinator import (
    CheckResult,
    DefaultBundleStager,
    FinalSubmissionCoordinator,
    SubmissionCoordinatorError,
    SubmissionDependencies,
    SubmissionJobConfig,
    build_aligned_csv,
)
from rex.submission.repository import (
    SubmissionRepository,
    SubmissionRepositoryError,
    SubmissionState,
)
from rex.submission.transport import FilesystemHandoff

__all__ = [
    "CheckResult",
    "DefaultBundleStager",
    "FilesystemHandoff",
    "FinalSubmissionCoordinator",
    "SubmissionCoordinatorError",
    "SubmissionDependencies",
    "SubmissionJobConfig",
    "SubmissionRepository",
    "SubmissionRepositoryError",
    "SubmissionState",
    "build_aligned_csv",
]
