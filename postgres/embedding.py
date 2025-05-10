"""
Main queries to perform the embedding in the database.
These include:
- Embedd the Entities
- Embedd the Relationships
- Embedd the Chunks
- Embedd the CommunityGroups
"""

# External imports:
import logging
from typing import List
from dotenv import load_dotenv
from psycopg2.extras import execute_values

# Internal imports:
#from api.postgres.base import Postgres

# Load environmental variables & logging
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EmbeddingQueries(Postgres):
    # Initializes the parent class (Postgres)
    def __init__(self):
        super().__init__()

    # ----------------------------- Chunk specific queries (embedding db) -------------------------- #
    def count_chunks_to_embed(self) -> int:
        """Counts the number of chunks to embed"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(chunk_id) as count FROM Chunk WHERE chunk_embed = true")
            row = cur.fetchone()
        return row[0] if row else 0

    def get_chunk_batches(self, batch_size: int, offset: int) -> List[tuple]:
        """
        Retrieves a batch of chunks that require embedding (where chunk_embed is True).
        Uses pagination with batch_size and offset for memory-efficient processing.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                        SELECT 
                            chunk_id, 
                            chunk_text, 
                            chunk_tokens,
                            chunk_emb
                            FROM Chunk 
                            WHERE chunk_embed = true
                            ORDER BY chunk_id
                            LIMIT %s OFFSET %s    
                        """, (batch_size, offset))
            return cur.fetchall()

    def update_emb_chunk(self, update_values: List[tuple]) -> None:
        """
        Bulk updates chunks with new embeddings.
        Expects update_values as a list of tuples: (chunk_id, emb)
        """
        if not update_values:
            return

        with self.conn.cursor() as cur:
            query = """
                       UPDATE Chunk AS c
                       SET chunk_emb = v.chunk_emb,
                           chunk_embed = false
                       FROM (VALUES %s) AS v(chunk_id, chunk_emb)
                       WHERE c.chunk_id = v.chunk_id
                   """
            try:
                execute_values(cur, query, update_values, template=None, page_size=100)
                logger.info(f"Bulk updated embeddings for {len(update_values)} chunks.")
            except Exception as e:
                logger.exception(f"Bulk update embeddings error: {e}")

    # ----------------------------- Entity specific queries (embedding db) -------------------------- #
    def count_entities_to_embed(self) -> int:
        """Counts the number of entities to embed"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(entity_id) as count FROM Entity WHERE entity_embed = true")
            row = cur.fetchone()
        return row[0] if row else 0

    def get_entities_batches(self, batch_size: int, offset: int) -> List[tuple]:
        """
        Retrieves a batch of entities that require embedding (where entity_embed is True).
        Uses pagination with batch_size and offset for memory-efficient processing.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                        SELECT 
                            entity_id, 
                            entity_name, 
                            entity_emb
                            FROM Entity 
                            WHERE entity_embed = true
                            ORDER BY entity_id
                            LIMIT %s OFFSET %s    
                        """, (batch_size, offset))
            return cur.fetchall()

    def update_entities_emb(self, update_values: List[tuple]) -> None:
        """Bulk updates entities with new embeddings. Expects update_values as a list of tuples: (id, emb)"""
        if not update_values:
            return

        with self.conn.cursor() as cur:
            query = """
                       UPDATE Entity AS e
                       SET entity_emb = v.entity_emb,
                           entity_embed = false
                       FROM (VALUES %s) AS v(entity_id, entity_emb)
                       WHERE e.entity_id = v.entity_id
                   """
            try:
                execute_values(cur, query, update_values, template=None, page_size=100)
                logger.info(f"Bulk updated embeddings for {len(update_values)} entities.")
            except Exception as e:
                logger.exception(f"Bulk update embeddings error: {e}")

    # ----------------------------- Relationship specific queries (embedding db) -------------------------- #
    def count_rels_to_embed(self) -> int:
        """Counts the number of entities to embed"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(rel_id) as count FROM Relationship WHERE rel_embed = true")
            row = cur.fetchone()
        return row[0] if row else 0

    def get_rels_batches(self, batch_size: int, offset: int) -> List[tuple]:
        """
        Retrieves a batch of relationships that require embedding (where rel_embed is True).
        Uses pagination with batch_size and offset for memory-efficient processing.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                          SELECT 
                              rel_id, 
                              rel_description, 
                              rel_emb,
                              rel_tokens
                              FROM Relationship 
                              WHERE rel_embed = true
                              ORDER BY rel_id
                              LIMIT %s OFFSET %s    
                          """, (batch_size, offset))
            return cur.fetchall()

    def update_rels_emb(self, update_values: List[tuple]) -> None:
        """Bulk updates relationships with new embeddings and token counts.
        Expects update_values as a list of tuples: (id, emb, tokens)"""
        if not update_values:
            return
        with self.conn.cursor() as cur:
            query = """
                         UPDATE RELATIONSHIP AS r
                         SET rel_emb = v.rel_emb,
                             rel_tokens = v.rel_tokens,
                             rel_embed = false
                         FROM (VALUES %s) AS v(rel_id, rel_emb, rel_tokens)
                         WHERE r.rel_id = v.rel_id
                     """
            try:
                execute_values(cur, query, update_values, template=None, page_size=100)
                logger.info(f"Bulk updated embeddings and token counts for {len(update_values)} relationships.")
            except Exception as e:
                logger.exception(f"Bulk update embeddings error: {e}")

    def calculate_rel_weights(self):
        """Calcualtes the weights of the relationships (similarity source and target)"""
        with self.conn.cursor() as cur:
            query = """
                    UPDATE Relationship r
                    SET rel_weight = 1 + FLOOR(9 * (1-(e1.entity_emb <=> e2.entity_emb)))::int
                    FROM Entity e1, Entity e2
                    WHERE r.from_entity = e1.entity_id
                      AND r.to_entity = e2.entity_id
                      AND r.rel_weight IS NULL
                      AND e1.entity_emb IS NOT NULL
                      AND e2.entity_emb IS NOT NULL;
                    """
            cur.execute(query)






