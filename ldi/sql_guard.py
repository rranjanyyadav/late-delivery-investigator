"""SQL safety for LLM-written queries. Three layers, each enough on its own
for some attacks, together covering all the ones we know of:

1. validate_sql()  - rejects anything that is not ONE SELECT statement, before
                     it reaches the database, with a message the LLM can act on.
2. connect_readonly() - the DuckDB file is opened read-only, so writes fail
                     even if a statement slipped past layer 1.
3. enable_external_access = false - no reading local files (read_csv), no
                     ATTACH, no COPY ... TO, no extension downloads.

Plus a row limit, so a runaway query cannot flood the UI or the LLM prompt.
"""
import re

import duckdb
import pandas as pd

from ldi.config import DB_PATH

MAX_ROWS = 1000

# Any of these words anywhere in the query (outside string literals) means it
# is not a plain read.
FORBIDDEN_WORDS = {
    "insert", "update", "delete", "merge", "upsert", "drop", "create", "alter",
    "truncate", "attach", "detach", "copy", "export", "import",
    "install", "load", "pragma", "set", "reset", "call", "checkpoint", "vacuum",
    "grant", "revoke", "begin", "commit", "rollback", "use", "glob",
}


class UnsafeSQLError(ValueError):
    """The query is not a single read-only SELECT."""


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)  # /* block */
    return re.sub(r"--[^\n]*", " ", sql)                   # -- line


def validate_sql(sql: str) -> str:
    """Return the cleaned query if it is a single SELECT, else raise UnsafeSQLError."""
    cleaned = _strip_comments(sql).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQLError("Empty query.")

    # Blank out string literals first, so a value like 'drop' or 'a;b' in a
    # WHERE clause is not mistaken for a keyword or a statement separator.
    masked = re.sub(r"'(?:[^']|'')*'", "''", cleaned).lower()

    if ";" in masked:
        raise UnsafeSQLError("Only one statement is allowed (found ';').")
    first_word = masked.split(None, 1)[0]
    if first_word not in ("select", "with"):
        raise UnsafeSQLError(f"Only SELECT queries are allowed, got '{first_word.upper()}'.")

    words = set(re.findall(r"[a-z_]+", masked))
    bad = sorted(words & FORBIDDEN_WORDS)
    if bad:
        raise UnsafeSQLError(f"Forbidden keyword(s) in a read-only query: {', '.join(bad)}.")
    # read_csv, read_parquet, read_json, read_text, ... all read files.
    readers = sorted(w for w in words if w.startswith("read_"))
    if readers:
        raise UnsafeSQLError(f"File-reading functions are not allowed: {', '.join(readers)}.")
    return cleaned


def connect_readonly(db_path=DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(
        str(db_path), read_only=True, config={"enable_external_access": False}
    )


def run_query(con, sql: str, params=None, max_rows: int = MAX_ROWS):
    """Execute and return (DataFrame, truncated). Only max_rows rows are
    fetched; truncated=True tells the caller (and the LLM) there were more."""
    cur = con.cursor()  # own cursor, so an unfinished result never blocks `con`
    try:
        cur.execute(sql, params or [])
        rows = cur.fetchmany(max_rows + 1)
        columns = [d[0] for d in cur.description]
    finally:
        cur.close()
    truncated = len(rows) > max_rows
    return pd.DataFrame(rows[:max_rows], columns=columns), truncated
