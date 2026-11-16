"""Transactional SQLite snapshots with retained edit history."""
import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4
from .models import Calibration


class EventRepository:
    SCHEMA_VERSION = 1

    def __init__(self, path):
        self.path = path

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.execute('PRAGMA foreign_keys=ON')
        return connection

    def save(self, calibration, session_id=None, history=()):
        # Validate before writing a transaction; unsupported future schemas stay untouched.
        Calibration.from_dict(calibration.to_dict())
        now = datetime.now(timezone.utc).isoformat()
        session_id = session_id or str(uuid4())
        with self._connect() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, self.SCHEMA_VERSION):
                raise ValueError('未対応のデータベース形式です')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calibrations(id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), created_at TEXT NOT NULL, data_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calibration_edits(id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), created_at TEXT NOT NULL, data_json TEXT NOT NULL);
            ''')
            db.execute(f'PRAGMA user_version={self.SCHEMA_VERSION}')
            db.execute('INSERT OR IGNORE INTO sessions VALUES(?,?,?)', (session_id, now, json.dumps(calibration.source, ensure_ascii=False)))
            db.execute('UPDATE sessions SET source_json=? WHERE id=?', (json.dumps(calibration.source, ensure_ascii=False), session_id))
            db.execute('INSERT INTO calibrations(session_id,created_at,data_json) VALUES(?,?,?)',
                       (session_id, now, json.dumps(calibration.to_dict(), ensure_ascii=False)))
            db.executemany('INSERT INTO calibration_edits(session_id,created_at,data_json) VALUES(?,?,?)',
                           [(session_id, now, json.dumps(x.to_dict(), ensure_ascii=False)) for x in history])
        return session_id

    def load(self):
        with self._connect() as db:
            if db.execute('PRAGMA user_version').fetchone()[0] != self.SCHEMA_VERSION:
                raise ValueError('未対応のデータベース形式です')
            row = db.execute('SELECT session_id,data_json FROM calibrations ORDER BY id DESC LIMIT 1').fetchone()
        if not row:
            raise ValueError('キャリブレーションがありません')
        return row[0], Calibration.from_dict(json.loads(row[1]))
