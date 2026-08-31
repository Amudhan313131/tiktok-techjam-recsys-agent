PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deadline_epoch_ms INTEGER NOT NULL,
    root_commit TEXT NOT NULL,
    environment_sha256 TEXT NOT NULL,
    data_manifest_sha256 TEXT NOT NULL,
    evaluator_sha256 TEXT NOT NULL,
    hypothesis_count INTEGER NOT NULL DEFAULT 0,
    official_evaluation_count INTEGER NOT NULL DEFAULT 0,
    non_improvement_streak INTEGER NOT NULL DEFAULT 0,
    best_primary_units INTEGER,
    best_ever_experiment_id TEXT,
    search_champion_experiment_id TEXT,
    shadow_best_primary_units INTEGER,
    shadow_champion_experiment_id TEXT,
    validation_phase TEXT NOT NULL DEFAULT 'DISCOVERY',
    finalist_experiment_id TEXT,
    official_evaluated_at TEXT,
    stop_reason TEXT
);

CREATE TABLE IF NOT EXISTS process_sessions (
    session_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    pid INTEGER,
    host TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    monotonic_seconds REAL NOT NULL DEFAULT 0,
    last_heartbeat TEXT,
    exit_reason TEXT,
    console_log_sha256 TEXT
);

CREATE TABLE IF NOT EXISTS baseline_gates (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    primary_units INTEGER NOT NULL,
    gauc REAL NOT NULL,
    ndcg5 REAL NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS validation_phase_transitions (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    from_phase TEXT NOT NULL,
    to_phase TEXT NOT NULL,
    finalist_experiment_id TEXT REFERENCES experiments(experiment_id),
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    PRIMARY KEY(run_id, to_phase)
);

CREATE TABLE IF NOT EXISTS shadow_evaluations (
    experiment_id TEXT PRIMARY KEY REFERENCES experiments(experiment_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    family TEXT NOT NULL,
    primary_units INTEGER NOT NULL,
    supported INTEGER NOT NULL,
    delta_units INTEGER,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    iteration_number INTEGER NOT NULL,
    parent_id TEXT REFERENCES experiments(experiment_id),
    operator TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    proposal_json TEXT NOT NULL,
    state TEXT NOT NULL,
    branch_name TEXT,
    parent_commit TEXT,
    commit_sha TEXT,
    config_sha256 TEXT,
    workspace_path TEXT,
    method_card_id TEXT,
    experiment_kind TEXT,
    terminal_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(run_id, iteration_number)
);

CREATE TABLE IF NOT EXISTS transitions (
    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    rung TEXT NOT NULL,
    repair_number INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    command_sha256 TEXT,
    commit_sha TEXT,
    started_at TEXT,
    ended_at TEXT,
    wall_seconds REAL NOT NULL DEFAULT 0,
    exit_code INTEGER,
    signal INTEGER,
    error_type TEXT,
    error_summary TEXT,
    stdout_artifact_id TEXT,
    stderr_artifact_id TEXT
);

CREATE TABLE IF NOT EXISTS experiment_repairs (
    repair_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    repair_number INTEGER NOT NULL,
    phase TEXT NOT NULL,
    failure_status TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    evidence_json TEXT,
    previous_commit_sha TEXT,
    repaired_commit_sha TEXT,
    previous_config_sha256 TEXT,
    repaired_config_sha256 TEXT,
    effective_config_artifact_id TEXT REFERENCES artifacts(artifact_id),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(experiment_id, repair_number)
);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    attempt_id TEXT REFERENCES attempts(attempt_id),
    split TEXT NOT NULL,
    fold TEXT,
    seed INTEGER,
    evaluator_sha256 TEXT NOT NULL,
    gauc REAL NOT NULL,
    ndcg5 REAL NOT NULL,
    primary_score REAL NOT NULL,
    primary_units INTEGER NOT NULL,
    ci_low REAL,
    ci_high REAL,
    rows INTEGER NOT NULL,
    users INTEGER NOT NULL,
    diagnostics_artifact_id TEXT,
    UNIQUE(experiment_id, split, fold, seed)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    attempt_id TEXT REFERENCES attempts(attempt_id),
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_links (
    link_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    experiment_id TEXT REFERENCES experiments(experiment_id),
    attempt_id TEXT REFERENCES attempts(attempt_id),
    artifact_path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotions (
    promotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    previous_experiment_id TEXT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    primary_units INTEGER NOT NULL,
    checkpoint_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    prediction_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    submission_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    validator_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_promotions (
    promotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    previous_experiment_id TEXT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    primary_units INTEGER NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS convergence_transactions (
    experiment_id TEXT PRIMARY KEY REFERENCES experiments(experiment_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    outcome TEXT NOT NULL,
    delta_units INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    experiment_id TEXT REFERENCES experiments(experiment_id),
    role TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    request_id TEXT,
    request_artifact_id TEXT,
    response_artifact_id TEXT,
    schema_valid INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    wall_seconds REAL NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS resource_usage (
    resource_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    experiment_id TEXT REFERENCES experiments(experiment_id),
    attempt_id TEXT REFERENCES attempts(attempt_id),
    scope TEXT NOT NULL,
    wall_seconds REAL NOT NULL DEFAULT 0,
    cpu_user_seconds REAL NOT NULL DEFAULT 0,
    cpu_system_seconds REAL NOT NULL DEFAULT 0,
    peak_rss_bytes INTEGER NOT NULL DEFAULT 0,
    gpu_seconds REAL NOT NULL DEFAULT 0,
    llm_tokens INTEGER NOT NULL DEFAULT 0,
    resource_key TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS lessons (
    lesson_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    scope TEXT NOT NULL,
    lesson TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interventions (
    intervention_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_outbox (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    event_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_hash TEXT,
    event_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    exported_at TEXT
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
