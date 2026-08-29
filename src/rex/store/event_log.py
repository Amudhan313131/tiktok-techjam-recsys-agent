"""Append-only JSONL export and verification of the SQLite event outbox."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile

from rex.store.db import Database
from rex.store.repository import utc_now
from rex.data.manifest import canonical_json_bytes


def export_events(
    database: Database,
    run_id: str,
    destination: str | Path,
    *,
    include_exported: bool = False,
) -> int:
    """Deterministically rebuild a run's complete event log and atomically replace it.

    Rebuilding rather than appending closes the crash window where bytes reached the
    destination but SQLite did not record their export, which previously duplicated
    events after restart. ``include_exported`` remains for API compatibility.
    """
    del include_exported
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    with database.transaction() as connection:
        rows = connection.execute(
            "SELECT sequence,run_id,event_type,aggregate_id,payload_json,previous_hash,event_hash,"
            "created_at FROM event_outbox WHERE run_id=? ORDER BY sequence",
            (run_id,),
        ).fetchall()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Some filesystems do not allow fsync on directories; replacement is still atomic.
                pass
            connection.execute(
                "UPDATE event_outbox SET exported_at=? WHERE run_id=?",
                (utc_now(), run_id),
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return len(rows)


def verify_event_chain(path: str | Path) -> bool:
    previous = None
    seen = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        seen = True
        event = json.loads(line)
        if event["previous_hash"] != previous:
            return False
        body = {
            "run_id": event["run_id"],
            "event_type": event["event_type"],
            "aggregate_id": event["aggregate_id"],
            "payload": json.loads(event["payload_json"]),
            "previous_hash": event["previous_hash"],
        }
        if hashlib.sha256(canonical_json_bytes(body)).hexdigest() != event["event_hash"]:
            return False
        previous = event["event_hash"]
    return seen
