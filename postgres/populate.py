"""
Main queries to populate the database.
These include:
- Inserting documents,
- Inserting chunks,
- Building the Knowledge Graph (Inserting/Updating)
 relationships and entities.
"""
# External imports:
import logging
from typing import Optional, Dict
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from typing import List, Tuple

# Internal imports:
from postgres.base import Postgres

# Load environmental variables & logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PopulateQueries(Postgres):
    # Initializes the parent class (Postgres)
    def __init__(self):
        super().__init__()

# ----------------------------- Document specific queries (populate db) -------------------------- #
    def set_document(self, doc: str) -> int:
        """Inserts a document into the database and returns its ID."""
        query = """
            INSERT INTO Document (content)
            VALUES (%s)
            RETURNING document_id;
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (doc,))
            document_id = cur.fetchone()[0]
        self.conn.commit()
        return document_id

    def clear_documents(self) -> None:
        """Clears all the documents from the database."""
        query = """
        DELETE FROM DOCUMENT;
        """
        with self.conn.cursor() as cur:
            cur.execute(query)

# -------------------------- Chunk Specific Queries (populate db) -------------------------------- #
    def set_chunks(self, values) -> None:
        """Adds chunks into the database and returns them (without the embedding)."""
        with self.conn.cursor() as cur:
            query = """
                  INSERT INTO Chunk (chunk_document_id, chunk_text, chunk_tokens, chunk_type, chunk_embed)
                  VALUES %s
              """
            execute_values(cur, query, values, template=None, page_size=100)

    def set_chunks_with_positions(self, chunks) -> None:
        """
        Insert chunks with start/end positions (needed for evaluation chromadb). Needs adjustment. (use chunk_embed)
        """
        query = """
               INSERT INTO Chunk (chunk_document_id, chunk_text, chunk_tokens, chunk_type, chunk_embed, start_index, end_index) 
               VALUES %s
               """
        with self.conn.cursor() as cur:
            execute_values(cur, query, chunks, page_size=100)

    def load_chunks_in_batches(self, document_id: int, batch_size: Optional[int] = 10, offset: int = 0) -> List[tuple]:
        """ Loads chunks in batches for KG-Construction using pagination with LIMIT and OFFSET"""
        query = """
                 SELECT chunk_id, chunk_document_id, chunk_text, chunk_tokens, chunk_type
                 FROM Chunk
                 WHERE chunk_document_id = %s
                 ORDER BY chunk_id
                 LIMIT %s OFFSET %s
             """
        with self.conn.cursor() as cur:
            cur.execute(query, (document_id, batch_size, offset))
            chunks = cur.fetchall()
            logger.info(
                f"Loaded {len(chunks)} chunks for document {document_id} (offset: {offset}, limit: {batch_size})")
            return chunks

    def fetch_chunk_by_id(self, chunk_id: int) -> tuple:
        """Returns the chunk by id. This is implemented for the failback strategy"""
        query = """
                  SELECT chunk_id, chunk_document_id, chunk_text, chunk_tokens, chunk_type
                  FROM Chunk WHERE chunk_id = %s
                  """
        with self.conn.cursor() as cur:
            cur.execute(query, (chunk_id,))
            chunk = cur.fetchone()
            return chunk

    def clear_chunks_by_type(self, chunk_type: str) -> None:
        """Clears all chunks of a specific type from the database."""
        query = """
        DELETE FROM Chunk 
        WHERE chunk_type = %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (chunk_type,))

# ---------------------- Entity Specific Queries (populate db) ------------------------------------- #
    def upsert_entities_bulk(self, names: List[str]) -> List[Tuple[str, int]]:
        """
        Bulk upsert Entity rows by name, returns list of (name, entity_id).
        """
        if not names:
            return []

        values = [(name, True) for name in names]
        query = '''
             INSERT INTO Entity (entity_name, entity_embed)
             VALUES %s
             ON CONFLICT (entity_name) DO UPDATE
               SET entity_embed = TRUE
             RETURNING entity_name, entity_id;
         '''
        with self.conn.cursor() as cur:
            execute_values(cur, query, values, template=None, page_size=100)
            results = cur.fetchall()
        return results

    def insert_entity_documents_bulk(self, links: List[Tuple[int, int]]) -> None:
        """
        Bulk insert into EntityDocument join table: (entity_id, document_id).
        """
        if not links:
            return
        query = '''
             INSERT INTO EntityDocument (entity_id, document_id)
             VALUES %s
             ON CONFLICT DO NOTHING;
         '''
        with self.conn.cursor() as cur:
            execute_values(cur, query, links, page_size=100)
        self.conn.commit()

    def insert_entity_chunks_bulk(self, links: List[Tuple[int, int]]) -> None:
        """
        Bulk insert into EntityChunk join table: (entity_id, chunk_id).
        """
        if not links:
            return
        query = '''
             INSERT INTO EntityChunk (entity_id, chunk_id)
             VALUES %s
             ON CONFLICT DO NOTHING;
         '''
        with self.conn.cursor() as cur:
            execute_values(cur, query, links, page_size=100)
        self.conn.commit()

    # ---------------------- Relationship Specific Queries (populate db) ------------------------------------- #
    def upsert_relationships_bulk(self, values):
        """Insert the relationships into the database"""
        with self.conn.cursor() as cur:
            query = """
                     INSERT INTO Relationship (from_entity, to_entity, rel_description, rel_summary, rel_chunk_id, rel_document_id, rel_embed) 
                     VALUES %s
                     """
            try:
                execute_values(cur, query, values, template=None, page_size=100)
                logger.info(f"Bulk inserted {len(values)} relationships.")
            except Exception as e:
                logger.exception(f"Bulk insert relationships error: {e}")

    # -------------------------------------------------------------------------------------------------------- #
