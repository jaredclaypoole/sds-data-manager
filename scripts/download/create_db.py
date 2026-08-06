"""Utility to create a local postgres database."""

import os
from pathlib import Path
from urllib.parse import urlparse

from psycopg import connect, sql

DB_URL = os.environ["DATABASE_URL"]
parsed = urlparse(DB_URL)
DB_NAME = Path(parsed.path).name
ADMIN_DB_URL = parsed._replace(path="/postgres").geturl()


def main():
    """Create a local postgres database."""
    conn = connect(ADMIN_DB_URL)
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))


if __name__ == "__main__":
    main()
