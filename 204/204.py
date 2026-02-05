from __future__ import annotations

import os
import re
import sqlite3
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class ForgetRequestResult:
    ok: bool
    user_id: str
    deleted_db_rows: int
    deleted_files: int
    redacted_logs: int
    errors: List[str]


class RightToBeForgotten:
    """
    A practical "delete on request" module.
    It supports:
      - Hard delete from SQLite tables (configurable table list)
      - Deleting user-owned files from upload/artifact directories
      - Optional log redaction (best-effort) for plain-text logs
    """

    def __init__(
        self,
        sqlite_path: str = "app.db",
        user_id_column: str = "user_id",
        tables: Optional[List[str]] = None,
        upload_dirs: Optional[List[str]] = None,
        log_files: Optional[List[str]] = None,
    ) -> None:
        self.sqlite_path = sqlite_path
        self.user_id_column = user_id_column

        # Tables that may store user-related data (adjust to your schema).
        self.tables = tables or [
            "users",
            "documents",
            "proposals",
            "annotations",
            "feedback",
            "jobs",
            "audit_events",
            "sessions",
        ]

        # Directories that may contain user files (adjust to your project paths).
        self.upload_dirs = upload_dirs or [
            "uploads",
            "artifacts",
            "media",
            "storage",
        ]

        # Optional plain-text logs to redact (not always required/possible).
        self.log_files = log_files or [
            "app.log",
            "server.log",
        ]

    # -------------------------
    # Public API
    # -------------------------

    def delete_user_data(self, user_id: str) -> ForgetRequestResult:
        """
        Orchestrates deletion across DB + filesystem + optional log redaction.
        Returns a structured summary for auditing.
        """
        errors: List[str] = []
        deleted_db_rows = 0
        deleted_files = 0
        redacted_logs = 0

        # 1) Delete DB rows (hard delete).
        try:
            deleted_db_rows = self._delete_from_sqlite(user_id)
        except Exception as e:
            errors.append(f"DB delete failed: {type(e).__name__}: {e}")

        # 2) Delete user files (best-effort).
        try:
            deleted_files = self._delete_user_files(user_id)
        except Exception as e:
            errors.append(f"File delete failed: {type(e).__name__}: {e}")

        # 3) Redact user_id from logs (best-effort).
        try:
            redacted_logs = self._redact_logs(user_id)
        except Exception as e:
            errors.append(f"Log redaction failed: {type(e).__name__}: {e}")

        ok = len(errors) == 0
        return ForgetRequestResult(
            ok=ok,
            user_id=user_id,
            deleted_db_rows=deleted_db_rows,
            deleted_files=deleted_files,
            redacted_logs=redacted_logs,
            errors=errors,
        )

    # -------------------------
    # SQLite deletion
    # -------------------------

    def _delete_from_sqlite(self, user_id: str) -> int:
        """
        Hard delete rows from configured tables where user_id matches.
        Uses a transaction; skips tables/columns that do not exist.
        """
        if not os.path.exists(self.sqlite_path):
            return 0

        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        deleted_total = 0

        def table_exists(name: str) -> bool:
            cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
            return cur.fetchone() is not None

        def column_exists(table: str, column: str) -> bool:
            cur = conn.execute(f"PRAGMA table_info({table})")
            cols = [row[1] for row in cur.fetchall()]  # row[1] = name
            return column in cols

        try:
            conn.execute("BEGIN;")
            for table in self.tables:
                if not table_exists(table):
                    continue
                if not column_exists(table, self.user_id_column):
                    continue

                # Parameterized query to avoid injection.
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE {self.user_id_column} = ?",
                    (user_id,),
                )
                deleted_total += cur.rowcount if cur.rowcount is not None else 0

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return deleted_total

    # -------------------------
    # File deletion
    # -------------------------

    def _delete_user_files(self, user_id: str) -> int:
        """
        Deletes files that likely belong to the user.
        Heuristics:
          - File path contains `/user_id/` directory
          - OR file name contains user_id token
        This is best-effort; you should adapt to your storage naming convention.
        """
        deleted = 0
        token = self._safe_token(user_id)

        for base in self.upload_dirs:
            if not os.path.isdir(base):
                continue

            for root, dirs, files in os.walk(base):
                # If the path includes a folder named exactly user_id, nuke that subtree fast.
                parts = set(os.path.normpath(root).split(os.sep))
                if user_id in parts:
                    # Delete all files under this root, then try removing dirs bottom-up.
                    for r2, d2, f2 in os.walk(root):
                        for fn in f2:
                            fp = os.path.join(r2, fn)
                            if self._safe_unlink(fp):
                                deleted += 1
                    # Remove directories bottom-up
                    for r2, d2, f2 in os.walk(root, topdown=False):
                        for dn in d2:
                            self._safe_rmdir(os.path.join(r2, dn))
                    self._safe_rmdir(root)
                    continue

                # Otherwise, delete files whose names contain the user token.
                for fn in files:
                    if token and token in fn:
                        fp = os.path.join(root, fn)
                        if self._safe_unlink(fp):
                            deleted += 1

        return deleted

    # -------------------------
    # Log redaction
    # -------------------------

    def _redact_logs(self, user_id: str) -> int:
        """
        Best-effort log redaction for plain-text logs.
        Replaces exact occurrences of user_id with "[REDACTED_USER]".
        This is not guaranteed for structured logs or external log systems.
        """
        redacted_files = 0
        for lf in self.log_files:
            if not os.path.isfile(lf):
                continue
            try:
                with open(lf, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if user_id not in content:
                    continue
                content2 = content.replace(user_id, "[REDACTED_USER]")
                tmp = lf + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(content2)
                os.replace(tmp, lf)
                redacted_files += 1
            except Exception:
                # Do not fail the entire operation due to logs.
                continue
        return redacted_files

    # -------------------------
    # Utilities
    # -------------------------

    def _safe_unlink(self, path: str) -> bool:
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False
        except IsADirectoryError:
            return False
        except PermissionError:
            return False
        except Exception:
            return False

    def _safe_rmdir(self, path: str) -> bool:
        try:
            os.rmdir(path)
            return True
        except Exception:
            return False

    def _safe_token(self, user_id: str) -> str:
        """
        Produce a safe token for filename matching (avoid path separators, wildcards, etc.).
        """
        uid = (user_id or "").strip()
        if not uid:
            return ""
        uid = uid.replace(os.sep, "_")
        uid = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", uid)
        return uid


# -------------------------
# CLI usage
# -------------------------

if __name__ == "__main__":
    # Example:
    #   python right_to_be_forgotten.py user123
    import sys

    if len(sys.argv) < 2:
        print("Usage: python right_to_be_forgotten.py <user_id>")
        raise SystemExit(2)

    uid = sys.argv[1]
    rtb = RightToBeForgotten(
        sqlite_path=os.environ.get("APP_SQLITE_PATH", "app.db"),
        # Adjust these lists to your project if needed; kept generic by design.
    )
    res = rtb.delete_user_data(uid)
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
    raise SystemExit(0 if res.ok else 1)
