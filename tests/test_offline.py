"""The suite must run without internet, API keys or the Kaggle data (CI has none)."""
import urllib.request

import pytest

from ldi.config import DB_PATH


def test_network_is_blocked_in_tests():
    # Proves the autouse no_network fixture in conftest.py is active.
    with pytest.raises(RuntimeError, match="Network access attempted"):
        urllib.request.urlopen("https://api.groq.com", timeout=5)


def test_tests_use_the_tiny_fixture_db_not_the_real_one(tiny_db):
    assert tiny_db != DB_PATH and tiny_db.name == "tiny.duckdb"
