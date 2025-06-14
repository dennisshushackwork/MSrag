"""
Main queries to perform RAG on the database.
"""

# External imports:
import logging
from dotenv import load_dotenv
from psycopg2.extras import DictCursor
from typing import List, Dict, Optional

# Internal imports:
from postgres.base import Postgres
from emb.re_ranker import Qwen3Reranker

# Load environmental variables & logging
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RetrievalQueries(Postgres):
    def __init__(self):
        """Initializes the parent class (Postgres) and the Qwen3 reranker."""
        super().__init__()
        # Initialize the Qwen3 reranker (singleton pattern)
        self.qwen3_reranker = Qwen3Reranker()
        logger.info("RetrievalQueries initialized with Qwen3Reranker")

    # ------------------ Standalone Reranker Functions ------------------ #
    @staticmethod
    def rerank_with_model(query: str, documents: List[Dict], model, top_k: int = 5) -> List[Dict]:
        """
        Legacy reranker function for any model with a .predict() method.
        Kept for backward compatibility.
        Note: This is deprecated - use rerank_with_qwen3() instead.
        """
        if not documents:
            return []
        # Create pairs of [query, document_content] for the model
        # Note: Fixed the print(doc) bug - should be doc['content'] or similar
        sentence_pairs = [[query, doc.get('content', str(doc))] for doc in documents]
        # Predict the relevance scores
        scores = model.predict(sentence_pairs)
        # Add scores to each document
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score
        # Sort documents by the new rerank score in descending order
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        # Return the top_k documents
        return reranked_docs[:top_k]

    def rerank_with_qwen3(self, query: str, documents: List[Dict], top_k: int = 5,
                         instruction: Optional[str] = None) -> List[Dict]:
        """
        Reranks documents using the Qwen3 reranker model.

        Args:
            query: Search query string
            documents: List of document dictionaries from semantic_search
            top_k: Number of top documents to return
            instruction: Optional task instruction for better performance

        Returns:
            List of reranked documents with 'rerank_score' field added
        """
        if not documents:
            return []

        # Default instruction for RAG scenarios
        if instruction is None:
            instruction = "Given a user question, find the most relevant document passages that contain the answer"

        logger.info(f"Reranking {len(documents)} documents with Qwen3 for query: '{query[:50]}...'")

        # Use the Qwen3 reranker with 'content' as the content key (matches your DB schema)
        reranked_docs = self.qwen3_reranker.rerank_with_model(
            query=query,
            documents=documents,
            top_k=top_k,
            instruction=instruction,
            content_key='content'  # This matches the 'content' field from your semantic_search
        )

        logger.info(f"Reranking complete. Top score: {reranked_docs[0]['rerank_score']:.4f}")
        return reranked_docs

    # ------------------------------ RAG Approach: Semantic Only ------------------------------- #

    def semantic_search(self, query_embedding: str, chunk_type: str, limit: int = 60) -> List[Dict]:
        """
        Performs a standard semantic search using only vector similarity.
        This function retrieves the top 'limit' chunks based on the cosine
        distance (vector similarity) between the query embedding and the stored
        chunk embeddings.
        """
        query = """
            SELECT
                chunk_id,
                chunk_document_id AS document_id,
                chunk_text AS content,
                chunk_tokens AS tokens,
                chunk_type AS type,
                start_index AS start,
                end_index AS end,
                1 - (chunk_emb <=> %s::vector) as score
            FROM Chunk
            WHERE chunk_type = %s
            ORDER BY score DESC
            LIMIT %s;
        """
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (query_embedding, chunk_type, limit))
            results = cur.fetchall()
            logger.info(f"Found {len(results)} results for semantic search.")
            # Convert DictRow to regular dict for easier handling
            return [dict(row) for row in results]

    # ------------------------------ RAG Approach: Semantic + Reranking ------------------------------- #

    def semantic_search_with_reranking(self, query: str, query_embedding: str, chunk_type: str,
                                     initial_limit: int = 100, final_top_k: int = 10,
                                     instruction: Optional[str] = None) -> List[Dict]:
        """
        Enhanced RAG approach: Semantic search followed by Qwen3 reranking.

        Args:
            query: Original text query (for reranking)
            query_embedding: Vector embedding of the query (for initial search)
            chunk_type: Type of chunks to search
            initial_limit: Number of candidates to retrieve from semantic search
            final_top_k: Number of final results after reranking
            instruction: Optional task instruction for the reranker

        Returns:
            List of top reranked documents
        """
        logger.info(f"Starting semantic search + reranking for query: '{query[:50]}...'")

        # Step 1: Get initial candidates using semantic search
        semantic_results = self.semantic_search(
            query_embedding=query_embedding,
            chunk_type=chunk_type,
            limit=initial_limit
        )

        if not semantic_results:
            logger.warning("No results found in semantic search")
            return []

        logger.info(f"Retrieved {len(semantic_results)} candidates from semantic search")

        # Step 2: Rerank using Qwen3
        reranked_results = self.rerank_with_qwen3(
            query=query,
            documents=semantic_results,
            top_k=final_top_k,
            instruction=instruction
        )

        logger.info(f"Final reranked results: {len(reranked_results)} documents")
        return reranked_results

    # ------------------------------ RAG Approach (wait for this one) ------------------------------- #
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
                        chunk_type AS type,
                        start_index AS start,
                        end_index AS end, 
                        -- Use the same score calculation as your working semantic_search
                        1 - (chunk_emb <=> %(embedding)s::vector) as similarity_score,
                        -- Assign a rank based on vector similarity (higher similarity is higher rank).
                        ROW_NUMBER() OVER (
                            ORDER BY (1 - (chunk_emb <=> %(embedding)s::vector)) DESC
                        ) AS sem_rank
                    FROM Chunk
                    WHERE chunk_type = %(chunk_type)s
                    ORDER BY similarity_score DESC
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
                        chunk_type AS type,
                        start_index AS start,
                        end_index AS end, 
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
                    COALESCE(s.chunk_id, k.chunk_id) AS chunk_id,
                    COALESCE(s.document_id, k.document_id) AS document_id,
                    COALESCE(s.content, k.content) AS content,
                    COALESCE(s.tokens, k.tokens) AS tokens,
                    COALESCE(s.type, k.type) AS type,
                    COALESCE(s.start, k.start) AS start,
                    COALESCE(s.end, k.end) AS end,
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
            # Convert DictRow to regular dict for easier handling
            return [dict(row) for row in results]


     # ------------------------------ Hybrid RAG Approach: Semantic + Syntactic + Reranking ------------------------------- #

    def hybrid_search_with_reranking(self, query: str, query_embedding: str, chunk_type: str,
                                     initial_limit: int = 20, final_top_k: int = 10,
                                     instruction: Optional[str] = None) -> List[Dict]:
        """
        Enhanced RAG approach: Semantic search followed by Qwen3 reranking.

        Args:
            query: Original text query (for reranking)
            query_embedding: Vector embedding of the query (for initial search)
            chunk_type: Type of chunks to search
            initial_limit: Number of candidates to retrieve from semantic search
            final_top_k: Number of final results after reranking
            instruction: Optional task instruction for the reranker

        Returns:
            List of top reranked documents
        """
        logger.info(f"Starting semantic search + reranking for query: '{query[:50]}...'")

        # Step 1: Get initial candidates using semantic search
        hybrid_results = self.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            chunk_type=chunk_type,
            final_limit=initial_limit
        )

        if not hybrid_results:
            logger.warning("No results found in hybrid search")
            return []

        logger.info(f"Retrieved {len(hybrid_results)} candidates from semantic search")

        # Step 2: Rerank using Qwen3
        reranked_results = self.rerank_with_qwen3(
            query=query,
            documents=hybrid_results,
            top_k=final_top_k,
            instruction=instruction
        )

        logger.info(f"Final reranked results: {len(reranked_results)} documents")
        return reranked_results

    # ------------------------------ Graph-RAG Approach ------------------------------- #
    def find_similar_entities(self, entities: dict, top_k: int) -> list:
        """Returns the top-k Similar entities"""
        print(top_k)
        # Holds the entity ids:
        results = []
        # Get similar entities query:
        query = """
                 SELECT 
                     entity_id
                 FROM Entity
                 ORDER BY 1-(entity_emb <=> %s::vector) DESC
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

    def extract_rel_given_ids(self, query_embedding: List[float], rel_ids: List[int]):
        """Returns the relationships ranked based on vector similarity to the query
           Returns a list of tuples of form (rel, tokens, similarity)"""
        query = """
        SELECT 
            r.rel_description, 
            r.rel_tokens,
            1-(r.rel_emb <=> %s::vector) AS cosine_similarity
        FROM Relationship r
        WHERE r.rel_id = ANY(%s)
        ORDER BY cosine_similarity DESC
        """
        results = []
        with self.conn.cursor() as cur:
            # Execute query:
            cur.execute(query, (query_embedding, rel_ids))
            results = cur.fetchall()
        return results