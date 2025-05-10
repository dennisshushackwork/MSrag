"""
Base Class for the database queries (Postgres).
Initializes the Postgres connection.
"""

# External imports:
import os
import logging
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load environmental variables & logging
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Postgres:

    def __init__(self):
        self.host = os.getenv("PG_HOST")
        self.port = int(os.getenv("PG_PORT"))
        self.user = os.getenv("PG_USER")
        self.pw = os.getenv("PG_PW")
        self.name = os.getenv("PG_DB")
        self.dsn = f"host={self.host} port={self.port} dbname={self.name} user={self.user} password={self.pw}"

        try:
            self.conn = psycopg2.connect(self.dsn)
            self.conn.autocommit = True
            logger.info("Connected to PostgresSQL database.")
        except Exception as e:
            logger.error(f"Error connecting to PostgresSQL: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()


