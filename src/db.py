"""SQLite case tracking for UPI Sentinel (analyst review workflow)."""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cases.db")

STATUSES = ["Open", "Confirmed Fraud", "False Positive", "Escalated"]


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with _connect() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS cases (
                txn_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                analyst_note TEXT,
                updated_at TEXT
            )"""
        )


def save_case(txn_id, status, note=""):
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    init_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as con:
        con.execute(
            """INSERT INTO cases (txn_id, status, analyst_note, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(txn_id) DO UPDATE SET
                 status=excluded.status,
                 analyst_note=excluded.analyst_note,
                 updated_at=excluded.updated_at""",
            (str(txn_id), status, note, now),
        )


def get_case(txn_id):
    init_db()
    with _connect() as con:
        row = con.execute(
            "SELECT txn_id, status, analyst_note, updated_at FROM cases WHERE txn_id = ?",
            (str(txn_id),),
        ).fetchone()
    if row is None:
        return None
    return {"txn_id": row[0], "status": row[1], "analyst_note": row[2], "updated_at": row[3]}


def get_all_cases():
    init_db()
    with _connect() as con:
        rows = con.execute(
            "SELECT txn_id, status, analyst_note, updated_at FROM cases ORDER BY updated_at DESC"
        ).fetchall()
    return [
        {"txn_id": r[0], "status": r[1], "analyst_note": r[2], "updated_at": r[3]} for r in rows
    ]


if __name__ == "__main__":
    init_db()
    save_case("TEST-1", "Open", "smoke test")
    print(get_case("TEST-1"))
    print(len(get_all_cases()), "case(s)")