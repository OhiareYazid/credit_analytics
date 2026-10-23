"""Thin helpers for running the repo's SQL files against DuckDB."""
import duckdb

import config as C


def connect(read_only: bool = False):
    return duckdb.connect(str(C.DB_PATH), read_only=read_only)


def run_sql(con, filename: str):
    """Execute a file from /sql and return the result as a DataFrame."""
    query = (C.SQL_DIR / filename).read_text()
    return con.sql(query).df()
