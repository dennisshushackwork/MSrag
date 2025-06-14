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

    def test_specific(self, chunk_id, emb):
        """"""
        query = """
        SELECT  1 - (chunk_emb <=> %s::vector) as score
        FROM Chunk
        Where chunk_id = %s"""
        with self.conn.cursor() as cur:
            cur.execute(query, (emb, chunk_id))
            print(cur.fetchone())

    def test(self, emb):
        """Returns the chunk by id. This is implemented for the failback strategy"""
        query = """
                  SELECT  1 - (chunk_emb <=> %s::vector) as score
                  FROM Chunk 
                  ORDER BY score DESC
                  """
        with self.conn.cursor() as cur:
            cur.execute(query, (emb,))
            chunk = cur.fetchall()
            print(chunk)
            print("-------")
            print(len(chunk))
            print("------")

    def count_chunk(self):
        """Returns the chunk by id. This is implemented for the failback strategy"""
        query = """
                  SELECT COUNT(chunk_id) FROM Chunk
                  """
        with self.conn.cursor() as cur:
            cur.execute(query)
            chunk = cur.fetchone()
            print(chunk)
