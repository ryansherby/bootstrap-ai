from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from configs import CONFIG


@dataclass(frozen=True)
class WorkflowFact:
    event_id: str
    event_type: str

    run_id: str
    epoch: str | None
    batch_id: str | None = None

    workflow_name: str | None = None
    process_name: str | None = None

    attributes: dict[str, Any] = field(default_factory=dict)


class WorkflowFactRepository(Protocol):
    async def record(self, fact: WorkflowFact) -> None:
        """Insert idempotently using event_id as the unique key."""
        ...


class SQLiteWorkflowFactRepository:
    """SQLite-backed fact repository for publish-only workflow facts."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    async def record(self, fact: WorkflowFact) -> None:
        await asyncio.to_thread(self._record_sync, fact)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_facts (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    epoch TEXT,
                    batch_id TEXT,
                    workflow_name TEXT NOT NULL,
                    process_name TEXT,
                    attributes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _record_sync(self, fact: WorkflowFact) -> None:
        attributes_json = json.dumps(fact.attributes, sort_keys=True, default=str)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workflow_facts (
                    event_id,
                    event_type,
                    run_id,
                    epoch,
                    batch_id,
                    workflow_name,
                    process_name,
                    attributes_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(event_id) DO UPDATE SET
                    event_type=excluded.event_type,
                    run_id=excluded.run_id,
                    epoch=excluded.epoch,
                    batch_id=excluded.batch_id,
                    workflow_name=excluded.workflow_name,
                    process_name=excluded.process_name,
                    attributes_json=excluded.attributes_json,
                    created_at=CURRENT_TIMESTAMP
                """,
                (
                    fact.event_id,
                    fact.event_type,
                    fact.run_id,
                    fact.epoch,
                    fact.batch_id,
                    fact.workflow_name,
                    fact.process_name,
                    attributes_json,
                ),
            )


def workflow_fact_repository_factory() -> SQLiteWorkflowFactRepository:
    return SQLiteWorkflowFactRepository(CONFIG.facts_db_path)