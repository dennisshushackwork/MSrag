"""
Updated ResolutionQueries class with additional methods for debugging
"""
# External imports
import json
import logging
import numpy as np
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from psycopg2.extras import execute_values, DictCursor

# Internal imports:
from postgres.base import Postgres

# Load environmental variables & logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResolutionQueries(Postgres):
    def __init__(self):
        super().__init__()

    # ---------------- Entity Resolution Queries -------------- #
    def count_entities(self) -> int:
        """Counts the number of chunks to embed"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(entity_id) as count FROM Entity")
            row = cur.fetchone()
        return row[0] if row else 0

    def load_entities_in_batches(self, offset: int, limit: int) -> Tuple[np.ndarray, List[int]]:
        """
        Load a batch of entity embeddings from database
        Fixed for pgvector VECTOR type with psycopg2
        """
        query = """
        SELECT entity_id, entity_emb as embedding
        FROM ENTITY
        ORDER BY entity_id
        LIMIT %s OFFSET %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (limit, offset))
            results = cur.fetchall()

            if not results:
                return np.array([]), []

            entity_ids = [row[0] for row in results]
            embeddings = [json.loads(row[1]) for row in results]

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # ADD THIS: Normalize the vectors
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            normalized_embeddings = embeddings_array / norms

            return normalized_embeddings, entity_ids

    def save_similarity_pairs(self, pairs) -> None:
        """Insert batch of entity similarities from EntityPair objects"""
        if not pairs:
            return
        # Convert EntityPair objects to tuples for database insertion
        values = [(pair.entity1_id, pair.entity2_id, pair.similarity) for pair in pairs]
        query = """
                INSERT INTO EntitySimilarity (from_entity, to_entity, similarity)
                VALUES %s
                """
        with self.conn.cursor() as cur:
            execute_values(cur, query, values)
        self.conn.commit()

    def count_similarity_pairs(self, min_similarity: float) -> int:
        """Count total similarity pairs above threshold"""
        query = "SELECT COUNT(*) FROM EntitySimilarity WHERE similarity >= %s"
        with self.conn.cursor() as cur:
            cur.execute(query, (min_similarity,))
            return cur.fetchone()[0]

    def load_similarity_pairs_batch(self, offset: int, limit: int, min_similarity: float) -> List[dict]:
        """Load batch of similarity pairs from database"""
        query = """
            SELECT from_entity as entity1_id, to_entity as entity2_id, similarity 
            FROM EntitySimilarity 
            WHERE similarity >= %s 
            ORDER BY from_entity, to_entity
            OFFSET %s LIMIT %s
        """
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (min_similarity, offset, limit))
            return [dict(row) for row in cur.fetchall()]

    def get_entity_names(self, entity_ids: List[int]) -> Dict[int, str]:
        """
        Get entity names from database for a list of entity IDs
        Processes in chunks to handle large batches efficiently
        """
        if not entity_ids:
            return {}

        # Remove duplicates while preserving order
        unique_entity_ids = list(dict.fromkeys(entity_ids))

        entity_names = {}
        chunk_size = 5000  # Safe chunk size to avoid database query limits

        try:
            # Process in chunks to handle large batches
            for i in range(0, len(unique_entity_ids), chunk_size):
                chunk = unique_entity_ids[i:i + chunk_size]

                # Create placeholders for the IN clause
                placeholders = ','.join(['%s'] * len(chunk))
                query = f"""
                    SELECT entity_id, entity_name 
                    FROM Entity 
                    WHERE entity_id IN ({placeholders})
                """

                with self.conn.cursor() as cur:
                    cur.execute(query, chunk)
                    results = cur.fetchall()

                    for row in results:
                        entity_id, entity_name = row
                        entity_names[entity_id] = entity_name if entity_name else ""

                logger.debug(f"Loaded {len([r for r in results])} entity names from chunk {i//chunk_size + 1}")

        except Exception as e:
            logger.error(f"Error fetching entity names: {e}")
            return {}

        # Log if some entities weren't found
        missing_ids = set(unique_entity_ids) - set(entity_names.keys())
        if missing_ids:
            logger.warning(f"Could not find names for {len(missing_ids)} entity IDs")

        logger.debug(f"Successfully loaded {len(entity_names)} entity names out of {len(unique_entity_ids)} requested")
        return entity_names
