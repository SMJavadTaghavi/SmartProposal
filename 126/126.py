from __future__ import annotations

import sqlite3
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class UserDecision:
    # Identifiers
    user_id: str
    document_id: str
    suggestion_id: str  # stable identifier for the suggestion/comment

    # Decision: ACCEPT / REJECT / EDIT
    decision: str

    # Original suggestion shown to the user (text)
    suggestion_text: str

    # If user edited or applied a change, store the applied change text and/or structured diff
    applied_change_text: Optional[str] = None
    applied_change_json: Optional[Dict[str, Any]] = None  # e.g., {"from":"...", "to":"...", "xpath":"..."} or any diff format

    # Optional metadata about where the suggestion applies (useful for ODT comment mapping)
    xpath: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None

    # Audit timestamps
    created_at_unix: int = 0


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS user_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    suggestion_id TEXT NOT NULL,

    decision TEXT NOT NULL CHECK (decision IN ('ACCEPT','REJECT','EDIT')),

    suggestion_text TEXT NOT NULL,

    applied_change_text TEXT,
    applied_change_json TEXT, -- JSON string (optional)

    xpath TEXT,
    start_offset INTEGER,
    end_offset INTEGER,

    created_at_unix INTEGER NOT NULL
);

-- Avoid duplicates: one decision per (user, document, suggestion).
CREATE UNIQUE INDEX IF NOT EXISTS ux_user_decisions
ON user_decisions (user_id, document_id, suggestion_id);

CREATE INDEX IF NOT EXISTS ix_user_decisions_doc
ON user_decisions (document_id);

CREATE INDEX IF NOT EXISTS ix_user_decisions_user
ON user_decisions (user_id);
"""


def init_db(db_path: str = "app.db") -> None:
    """Create tables and indexes."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def record_user_decision(dec: UserDecision, db_path: str = "app.db") -> int:
    """
    Insert or replace decision record.
    Returns the row id of the inserted record.
    """
    if dec.created_at_unix <= 0:
        dec.created_at_unix = int(time.time())

    applied_json_str = json.dumps(dec.applied_change_json, ensure_ascii=False) if dec.applied_change_json is not None else None

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # Insert with UPSERT so users can change their decision later.
        # We update decision + applied change fields + timestamps.
        cur = conn.execute(
            """
            INSERT INTO user_decisions (
                user_id, document_id, suggestion_id,
                decision, suggestion_text,
                applied_change_text, applied_change_json,
                xpath, start_offset, end_offset,
                created_at_unix
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, document_id, suggestion_id) DO UPDATE SET
                decision=excluded.decision,
                suggestion_text=excluded.suggestion_text,
                applied_change_text=excluded.applied_change_text,
                applied_change_json=excluded.applied_change_json,
                xpath=excluded.xpath,
                start_offset=excluded.start_offset,
                end_offset=excluded.end_offset,
                created_at_unix=excluded.created_at_unix
            """,
            (
                dec.user_id, dec.document_id, dec.suggestion_id,
                dec.decision, dec.suggestion_text,
                dec.applied_change_text, applied_json_str,
                dec.xpath, dec.start_offset, dec.end_offset,
                dec.created_at_unix,
            ),
        )
        conn.commit()

        # For SQLite, lastrowid is meaningful for INSERT, not for UPDATE via upsert.
        # We'll return the row id by selecting it using the unique key.
        row = conn.execute(
            """
            SELECT id FROM user_decisions
            WHERE user_id=? AND document_id=? AND suggestion_id=?
            """,
            (dec.user_id, dec.document_id, dec.suggestion_id),
        ).fetchone()
        return int(row[0]) if row else -1


def get_user_decision(user_id: str, document_id: str, suggestion_id: str, db_path: str = "app.db") -> Optional[Dict[str, Any]]:
    """Fetch a decision record by unique key."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT * FROM user_decisions
            WHERE user_id=? AND document_id=? AND suggestion_id=?
            """,
            (user_id, document_id, suggestion_id),
        ).fetchone()
        if not row:
            return None

        d = dict(row)
        # Decode JSON field
        if d.get("applied_change_json"):
            try:
                d["applied_change_json"] = json.loads(d["applied_change_json"])
            except Exception:
                # Keep raw string if decoding fails
                pass
        return d


# -------------------------
# Test: "ثبت صحیح در DB"
# -------------------------

if __name__ == "__main__":
    DB = "test_app.db"
    init_db(DB)

    decision = UserDecision(
        user_id="u1",
        document_id="doc42",
        suggestion_id="sugg-0007",
        decision="EDIT",
        suggestion_text="Replace informal wording with formal academic style.",
        applied_change_text="Changed 'خیلی خوب' to 'مطلوب'.",
        applied_change_json={"from": "خیلی خوب", "to": "مطلوب", "xpath": "/office:document-content/.../text:p[12]", "start": 5, "end": 12},
        xpath="/office:document-content/.../text:p[12]",
        start_offset=5,
        end_offset=12,
    )

    row_id = record_user_decision(decision, DB)
    fetched = get_user_decision("u1", "doc42", "sugg-0007", DB)

    ok = (
        row_id > 0 and
        fetched is not None and
        fetched["decision"] == "EDIT" and
        fetched["suggestion_text"] == decision.suggestion_text and
        fetched["applied_change_text"] == decision.applied_change_text and
        isinstance(fetched.get("applied_change_json"), dict) and
        fetched["applied_change_json"].get("to") == "مطلوب"
    )

    print("TEST_OK=", ok)
    if fetched:
        print(json.dumps(fetched, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ok else 1)
