"""SQLite connection and transaction helpers."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class Database:
    def __init__(self, path: str | Path, schema_path: str | Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema_path = Path(schema_path or Path(__file__).with_name("schema.sql"))

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(self.schema_path.read_text(encoding="utf-8"))
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """Apply additive, restart-safe migrations to databases from earlier builds."""
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(experiments)").fetchall()
        }
        for name, declaration in (
            ("workspace_path", "TEXT"),
            ("method_card_id", "TEXT"),
            ("experiment_kind", "TEXT"),
        ):
            if name not in columns:
                connection.execute(f"ALTER TABLE experiments ADD COLUMN {name} {declaration}")

        resource_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(resource_usage)").fetchall()
        }
        if "resource_key" not in resource_columns:
            connection.execute("ALTER TABLE resource_usage ADD COLUMN resource_key TEXT")
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS resource_usage_resource_key "
            "ON resource_usage(resource_key) WHERE resource_key IS NOT NULL"
        )
        link_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(artifact_links)").fetchall()
        }
        if "artifact_path" not in link_columns:
            connection.execute("ALTER TABLE artifact_links ADD COLUMN artifact_path TEXT")
        connection.execute(
            "UPDATE artifact_links SET artifact_path=(SELECT path FROM artifacts "
            "WHERE artifacts.artifact_id=artifact_links.artifact_id) WHERE artifact_path IS NULL"
        )
        connection.execute("DROP INDEX IF EXISTS artifact_links_provenance")
        connection.execute(
            "CREATE UNIQUE INDEX artifact_links_provenance ON artifact_links(artifact_id,"
            "IFNULL(experiment_id,''),IFNULL(attempt_id,''),IFNULL(artifact_path,''))"
        )
        for row in connection.execute(
            "SELECT artifact_id,experiment_id,attempt_id,path,created_at FROM artifacts"
        ).fetchall():
            provenance = (
                f"{row['artifact_id']}\0{row['experiment_id'] or ''}\0"
                f"{row['attempt_id'] or ''}\0{row['path']}"
            )
            link_id = "artifact-link-" + hashlib.sha256(provenance.encode("utf-8")).hexdigest()
            connection.execute(
                "INSERT OR IGNORE INTO artifact_links(link_id,artifact_id,experiment_id,attempt_id,"
                "artifact_path,created_at) VALUES(?,?,?,?,?,?)",
                (
                    link_id,
                    row["artifact_id"],
                    row["experiment_id"],
                    row["attempt_id"],
                    row["path"],
                    row["created_at"],
                ),
            )
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version,name,applied_at) VALUES(?,?,?)",
            (1, "initial_schema", now),
        )
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version,name,applied_at) VALUES(?,?,?)",
            (2, "durable_sessions_and_exactly_once_records", now),
        )
        connection.execute("PRAGMA user_version = 2")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
