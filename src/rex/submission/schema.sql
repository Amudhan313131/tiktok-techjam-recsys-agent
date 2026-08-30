PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS submission_jobs (
    job_id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL,
    source_database_path TEXT NOT NULL,
    source_run_fingerprint TEXT NOT NULL,
    source_report_path TEXT NOT NULL,
    source_report_sha256 TEXT NOT NULL,
    best_valid_path TEXT NOT NULL,
    best_valid_sha256 TEXT NOT NULL,
    source_commit TEXT,
    config_sha256 TEXT,
    incumbent_experiment_id TEXT,
    state TEXT NOT NULL,
    worktree_path TEXT,
    prediction_request_json TEXT,
    prediction_path TEXT,
    prediction_sha256 TEXT,
    csv_path TEXT,
    csv_sha256 TEXT,
    staging_path TEXT,
    sealed_path TEXT,
    seal_sha256 TEXT,
    error_code TEXT,
    error_summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(source_run_id, best_valid_sha256)
);

CREATE TABLE IF NOT EXISTS submission_transitions (
    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES submission_jobs(job_id),
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(job_id, to_state)
);

CREATE TABLE IF NOT EXISTS submission_checks (
    check_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES submission_jobs(job_id),
    ordinal INTEGER NOT NULL,
    csv_path TEXT NOT NULL,
    csv_sha256 TEXT NOT NULL,
    command_json TEXT NOT NULL,
    stdout TEXT NOT NULL,
    stderr TEXT NOT NULL,
    returncode INTEGER NOT NULL,
    transcript_path TEXT NOT NULL,
    transcript_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(job_id, ordinal)
);

CREATE TABLE IF NOT EXISTS submission_handoffs (
    job_id TEXT PRIMARY KEY REFERENCES submission_jobs(job_id),
    authorized_seal_sha256 TEXT NOT NULL,
    target_path TEXT NOT NULL,
    target_manifest_sha256 TEXT,
    status TEXT NOT NULL,
    authorized_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(target_path)
);

PRAGMA user_version = 1;
