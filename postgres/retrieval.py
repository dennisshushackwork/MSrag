"""
Main queries to perform RAG on the database.
"""

# External imports:
import logging
from dotenv import load_dotenv
from psycopg2.extras import DictCursor

# Internal imports:
from postgres.base import Postgres

# Load environmental variables & logging
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RetrievalQueries(Postgres):
    # Initializes the parent class (Postgres)
    def __init__(self):
        super().__init__()

    def hybrid_search(self, query_text, query_embedding, chunk_type, rrf_k=60, candidate_pool_size=120, final_limit=60):
        """
        Performs a hybrid search combining semantic (vector) and keyword (full-text) search for RAG,
        optimized for performance and relevance on a very large database

        This function efficiently retrieves a larger pool of candidates from both semantic and
        keyword searches using indexed lookups, then combines and re-ranks these candidates
        using Reciprocal Rank Fusion (RRF) to provide a robust and relevant set of results.
        """
        query = f"""
                    -- CTE for Semantic Search (Vector Similarity using DiskANN)
                    -- This efficiently retrieves the top 'candidate_pool_size' chunks based on vector similarity.
                    WITH semantic_search AS (
                        SELECT
                            chunk_id,
                            chunk_document_id AS document_id,
                            chunk_text AS content,
                            chunk_tokens AS tokens,
                            -- Assign a rank based on vector similarity (smaller distance is higher rank).
                            -- ROW_NUMBER() OVER (ORDER BY chunk_emb <=> %(embedding)s) assigns 1 to the most similar.
                            ROW_NUMBER() OVER (
                                ORDER BY chunk_emb <=> %(embedding)s
                            ) AS sem_rank
                        FROM Chunk
                        WHERE chunk_type = %(chunk_type)s
                        ORDER BY chunk_emb <=> %(embedding)s
                        LIMIT %(candidate_pool_size)s -- Retrieve a larger pool of candidates for re-ranking
                    ),
                    -- CTE for Keyword Search (Full-Text Search using PostgreSQL's tsvector/tsquery)
                    -- This efficiently retrieves the top 'candidate_pool_size' chunks based on text relevance.
                    keyword_search AS (
                        SELECT
                            chunk_id,
                            chunk_document_id AS document_id,
                            chunk_text AS content,
                            chunk_tokens AS tokens,
                            -- Assign a rank based on full-text search relevance (higher ts_rank_cd is higher rank).
                            -- ROW_NUMBER() OVER (ORDER BY ts_rank_cd(...) DESC) assigns 1 to the most relevant.
                            ROW_NUMBER() OVER (
                                ORDER BY ts_rank_cd(
                                    chunk_text_tsv,
                                    plainto_tsquery('english', %(query)s)
                                ) DESC
                            ) AS key_rank
                        FROM Chunk
                        WHERE
                            chunk_type = %(chunk_type)s
                            AND chunk_text_tsv @@ plainto_tsquery('english', %(query)s) -- Efficient filtering using GIN index
                        ORDER BY
                            ts_rank_cd(
                                chunk_text_tsv,
                                plainto_tsquery('english', %(query)s)
                            ) DESC
                        LIMIT %(candidate_pool_size)s -- Retrieve a larger pool of candidates for re-ranking
                    )
                    -- Combine results from both searches using FULL OUTER JOIN and re-rank with RRF
                    SELECT
                        -- Use COALESCE to ensure we get values even if a chunk is found by only one search type.
                        COALESCE(s.document_id, k.document_id) AS document_id,
                        COALESCE(s.content, k.content) AS content,
                        COALESCE(s.tokens, k.tokens) AS tokens,
                        -- Calculate combined score using Reciprocal Rank Fusion (RRF).
                        -- The score is 1 / (k + rank). Higher score means higher relevance.
                        -- If a chunk is not found in one search, its rank for that search is effectively
                        -- set to beyond the candidate pool size (%(candidate_pool_size)s + 1), ensuring
                        -- it gets a very low score contribution from that specific search.
                        (1.0 / (%(rrf_k)s + COALESCE(s.sem_rank, %(candidate_pool_size)s + 1)))
                        + (1.0 / (%(rrf_k)s + COALESCE(k.key_rank, %(candidate_pool_size)s + 1)))
                        AS score
                    FROM semantic_search AS s
                    FULL OUTER JOIN keyword_search AS k USING (chunk_id) -- Join based on chunk_id
                    ORDER BY score DESC -- Order by the combined RRF score (highest first)
                    LIMIT %(final_limit)s; -- Limit the final combined and re-ranked results returned
                """
        with self.conn.cursor(cursor_factory=DictCursor) as cur:  # Use DictCursor for dictionary-like access to results
            params = {
                "chunk_type": chunk_type,
                "embedding": query_embedding,
                "query": query_text,
                "rrf_k": rrf_k,
                "candidate_pool_size": candidate_pool_size,
                "final_limit": final_limit
            }
            cur.execute(query, params)
            results = cur.fetchall()
            logger.info(
                f"Found {len(results)} results for query: '{query_text[:70]}{'...' if len(query_text) > 70 else ''}'")
            return results

    def find_similar_entities(self, entities: dict, top_k: int) -> list:
        """Returns the top-k Similar entities"""

        # Holds the entity ids:
        results = []
        # Get similar entities query:
        query = """
                 SELECT 
                     entity_id
                 FROM Entity
                 ORDER BY entity_emb <=> %s::vector
                 LIMIT %s;
                 """
        with self.conn.cursor() as cur:
            for entity_name, embedding in entities.items():
                cur.execute(query, (embedding, top_k))
                similar_entities = cur.fetchall()
                results.append(similar_entities)

        # Merge the results:
        merged_ids = set() # Use a set to store unique IDs
        for inner_list in results:
            for id_tuple in inner_list:
                # Extract the integer ID from the tuple (e.g., (448,) -> 448)
                merged_ids.add(id_tuple[0])
        final_unique_ids = list(merged_ids)
        return final_unique_ids


