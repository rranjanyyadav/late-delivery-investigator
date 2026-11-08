"""Central place for file paths, so no other module hard-codes them."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(os.getenv("LDI_RAW_DIR", ROOT / "data" / "raw"))
DB_PATH = Path(os.getenv("LDI_DB_PATH", ROOT / "data" / "olist.duckdb"))
SQL_DIR = ROOT / "sql"
