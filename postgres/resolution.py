"""
This class is specifically designed for the entity Resolution workflow.
"""
# External imports
import logging
import numpy as np
from typing import List, Dict, Tuple
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

    def load_entity_batch(self, offset: int, limit: int) -> Tuple[np.ndarray, List[int]]:
        """
        Load a batch of entity embeddings from database
        Fixed for pgvector VECTOR type with psycopg2
        """
        query = """
        SELECT entity_id, entity_emb as embedding
        FROM Entity
        WHERE entity_emb IS NOT NULL
        ORDER BY entity_id
        LIMIT %s OFFSET %s
        """
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (limit, offset))
            results = cur.fetchall()

        if not results:
            return np.array([]), []

        entity_ids = [r['entity_id'] for r in results]
        embeddings_list = []

        for r in results:
            embedding_data = r['embedding']

            try:
                if isinstance(embedding_data, str):
                    # pgvector returns vectors as strings like '[0.1,0.2,0.3]'
                    # Remove brackets and parse
                    cleaned_str = embedding_data.strip('[]')
                    values = [float(x.strip()) for x in cleaned_str.split(',')]
                    embeddings_list.append(values)
                elif hasattr(embedding_data, '__iter__'):
                    # If it's already iterable (list, array, etc.)
                    embeddings_list.append([float(x) for x in embedding_data])
                else:
                    logger.error(f"Unexpected embedding type: {type(embedding_data)}")
                    continue

            except Exception as e:
                logger.error(f"Failed to parse embedding for entity {r['entity_id']}: {e}")
                logger.error(f"Embedding data: {embedding_data}")
                continue

        if not embeddings_list:
            logger.warning("No valid embeddings found in batch")
            return np.array([]), []

        embeddings = np.array(embeddings_list, dtype=np.float32)

        # Validate dimensions
        if embeddings.shape[1] != 256:
            logger.error(f"Expected 256 dimensions, got {embeddings.shape[1]}")

        logger.info(f"Successfully loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
        return embeddings, entity_ids

    def get_entity_id_batch(self, offset: int, limit: int) -> List[int]:
        """Get a batch of entity IDs that have embeddings"""
        query = """
        SELECT entity_id
        FROM Entity
        WHERE entity_emb IS NOT NULL
        ORDER BY entity_id
        LIMIT %s OFFSET %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (limit, offset))
            return [row[0] for row in cur.fetchall()]

    def clear_entity_similarities(self) -> None:
        """Clear all entity similarities for fresh calculation"""
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE EntitySimilarity")
        self.conn.commit()

    def calculate_similarities_for_batch_adv_2(self, entity_ids: List[int], min_semantic_similarity: float = 0.85,
                                             min_lexical_similarity: float = 0.05):
        if not entity_ids:
            logger.info("No entity IDs provided for similarity calculation. Returning empty list.")
            return []

            # Optimized query using CTEs for better performance
        query = """
               WITH source_entities AS (
                   SELECT entity_id, entity_name, entity_emb
                   FROM Entity 
                   WHERE entity_id = ANY(%s) 
                     AND entity_emb IS NOT NULL
               )
               SELECT
                   se.entity_id AS source_entity_id,
                   se.entity_name AS source_entity_name,
                   e.entity_id AS target_entity_id,
                   e.entity_name AS target_entity_name,
                   (1 - (se.entity_emb <=> e.entity_emb)) AS semantic_similarity,
                   similarity(se.entity_name, e.entity_name) AS lexical_similarity,
                   -- Optional: combined score
                   ((1 - (se.entity_emb <=> e.entity_emb)) * 0.7 + similarity(se.entity_name, e.entity_name) * 0.3) AS combined_score
               FROM source_entities se
               CROSS JOIN Entity e
               WHERE se.entity_id != e.entity_id
                 AND e.entity_emb IS NOT NULL
                 AND (1 - (se.entity_emb <=> e.entity_emb)) >= %s
                 AND similarity(se.entity_name, e.entity_name) >= %s
               ORDER BY
                   se.entity_id,
                   semantic_similarity DESC,
                   lexical_similarity DESC
           """

        results = []
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, (entity_ids, min_semantic_similarity, min_lexical_similarity))
                results = cur.fetchall()

            logger.info(
                f"Calculated combined similarity for {len(entity_ids)} source entities. "
                f"Found {len(results)} pairs with semantic >= {min_semantic_similarity} "
                f"and lexical >= {min_lexical_similarity}."
            )
        except Exception as e:
            logger.error(f"Error calculating combined entity similarities for batch: {e}")

        print(results)

    def calculate_similarities_for_batch_adv(self, entity_ids: List[int], min_semantic_similarity: float = 0.85,
                                         min_lexical_similarity: float = 0.05) -> List[Dict]:
        """
        Calculates a combined similarity score (semantic + lexical using FTS) between a batch of specified
        entities and all other entities, filtering for scores above minimum thresholds.

        Args:
            entity_ids (List[int]): A list of entity IDs for which to calculate similarities.
            min_semantic_similarity (float): The minimum cosine similarity score (default 0.90).
            min_lexical_similarity (float): Returns a number that indicates how similar the two arguments are.
            The range of the result is zero (indicating that the two strings are completely dissimilar)
            to one (indicating that the two strings are identical).
        """
        if not entity_ids:
            logger.info("No entity IDs provided for similarity calculation. Returning empty list.")
            return []

        query = """
            SELECT
                e1.entity_id AS source_entity_id,
                e1.entity_name AS source_entity_name,
                e2.entity_id AS target_entity_id,
                e2.entity_name AS target_entity_name,
                (1 - (e1.entity_emb <=> e2.entity_emb)) AS semantic_similarity,
                similarity(STRIP(e1.entity_name_tsv)::text, STRIP(e2.entity_name_tsv)::text) AS lexical_similarity
            FROM
                Entity e1
            JOIN
                Entity e2 ON e1.entity_id != e2.entity_id
            WHERE
                e1.entity_id = ANY(%s)
                AND e1.entity_emb IS NOT NULL
                AND e2.entity_emb IS NOT NULL
                AND (1 - (e1.entity_emb <=> e2.entity_emb)) >= %s
                AND similarity(STRIP(e1.entity_name_tsv)::text, STRIP(e2.entity_name_tsv)::text) >= %s
            ORDER BY
                source_entity_id,
                semantic_similarity DESC, -- Prioritize semantic similarity in sorting
                lexical_similarity DESC;
        """
        results = []
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                # Pass entity_ids, min_semantic_similarity, and min_lexical_similarity
                cur.execute(query, (entity_ids, min_semantic_similarity, min_lexical_similarity))
                results = cur.fetchall()
            logger.info(
                f"Calculated combined similarity for {len(entity_ids)} source entities. "
                f"Found {len(results)} pairs with semantic >= {min_semantic_similarity} "
                f"and FTS lexical >= {min_lexical_similarity}."
            )
        except Exception as e:
            logger.error(f"Error calculating combined entity similarities for batch: {e}")
        print(results)
        #return results


    def calculate_similarities_for_batch(self, entity_ids: List[int], min_similarity: float):
        """Calculates the similarity among a batch of entities. """

        if not entity_ids:
            logger.info("No entity IDs provided for similarity calculation. Returning empty list.")
            return []

        query = """
            SELECT
                e1.entity_id AS source_entity_id,
                e1.entity_name AS source_entity_name,
                e2.entity_id AS target_entity_id,
                e2.entity_name AS target_entity_name,
                (1 - (e1.entity_emb <=> e2.entity_emb)) AS cosine_similarity
            FROM
                Entity e1
            JOIN
                Entity e2 ON e1.entity_id != e2.entity_id
            WHERE
                e1.entity_id = ANY(%s)
                AND (1 - (e1.entity_emb <=> e2.entity_emb)) >= %s 
            ORDER BY
                source_entity_id,
                cosine_similarity DESC;
        """
        results = []
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                # Pass both the list of IDs and the min_similarity threshold
                cur.execute(query, (entity_ids, min_similarity))
                results = cur.fetchall()
            logger.info(f"Calculated similarity for {len(entity_ids)} source entities. Found {len(results)} similar pairs above or equal to {min_similarity}.")
        except Exception as e:
            logger.error(f"Error calculating entity similarities for batch: {e}")
        print(results)
        #return results

    def insert_entity_similarities_batch(self, similarities: List[Dict]) -> None:
        """Insert batch of entity similarities from database query results"""
        if not similarities:
            return
        query = """
                INSERT INTO EntitySimilarity (from_entity, to_entity, similarity)
                VALUES %s
                """
        with self.conn.cursor() as cur:
            execute_values(cur, query, similarities)
        self.conn.commit()

    def load_entities_in_batches(self, batch_size: int, offset: int) -> List[Dict]:
        """Returns all the entities in batches"""
        query = """
                SELECT es.from_entity, es.to_entity, es.similarity,
                e1.entity_name as from_name, e2.entity_name as to_name
                FROM EntitySimilarity es
                JOIN Entity e1 ON es.from_entity = e1.entity_id
                JOIN Entity e2 ON es.to_entity = e2.entity_id
                LIMIT %s OFFSET %s
                """
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (batch_size, offset))
            entities = cur.fetchall()
            return entities



