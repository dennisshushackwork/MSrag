""" my cluster
Advanced Cluster Semantic Chunker using NLTK for sentence splitting and Gemma tokenizer for token counting.
Optimized with Numba for fast dynamic programming.

This chunker creates semantically coherent chunks by:
1. Splitting text into proper sentences using NLTK's sophisticated sentence boundary detection
2. Generating embeddings for each sentence using your custom Embedder (Alibaba-NLP/gte-multilingual-base)
3. Building a similarity matrix between all sentence pairs using cosine similarity
4. Using Numba-optimized dynamic programming to find the optimal clustering that maximizes
   semantic coherence while respecting token size limits
5. Post-processing to merge small adjacent chunks when beneficial
6. Using Gemma tokenizer for accurate token counting that matches your LLM's tokenization

Unlike simple token or recursive chunkers, this approach preserves semantic boundaries by keeping
related sentences together, resulting in more coherent chunks for better RAG performance.
"""
import re
from chunkers.base import BaseChunker
from typing import List, Tuple
from numba import njit
import numpy as np
from emb.gemma_tokenizer import GemmaSingletonTokenizer
from emb.embedder import Embedder
import logging
from functools import lru_cache

# NLTK imports
import nltk
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

@njit
def _calculate_reward_numba(matrix: np.ndarray, start: int, end: int) -> float:
    """
    Numba-optimized reward calculation for clustering sentences from start to end.

    The reward is the sum of all pairwise similarities within the cluster,
    with a size bonus to encourage larger, coherent chunks.

    Args:
        matrix: Similarity matrix
        start: Start index of cluster
        end: End index of cluster (inclusive)

    Returns:
        Total similarity reward for this cluster with size bonus
    """
    # Calculate sum of submatrix efficiently
    total_reward = 0.0
    for i in range(start, end + 1):
        for j in range(start, end + 1):
            total_reward += matrix[i, j]

    # Add size bonus to encourage larger clusters
    cluster_size = end - start + 1
    size_bonus = np.log(cluster_size) * 0.2  # Encourage larger clusters

    return total_reward + size_bonus

@njit
def _optimal_segmentation_numba(
    similarity_matrix: np.ndarray,
    sentence_tokens: np.ndarray,
    cumulative_tokens: np.ndarray,
    max_chunk_size: int,
    mean_similarity: float
) -> np.ndarray:
    """
    Numba-optimized dynamic programming for optimal segmentation.

    Args:
        similarity_matrix: NxN similarity matrix between sentences
        sentence_tokens: Array of token counts for each sentence
        cumulative_tokens: Cumulative sum of token counts for O(1) range queries
        max_chunk_size: Maximum tokens per chunk
        mean_similarity: Mean similarity to subtract for normalization

    Returns:
        Array where segmentation[i] = start index of cluster ending at i
    """
    # Normalize matrix by subtracting mean
    normalized_matrix = similarity_matrix - mean_similarity

    # Zero out diagonal (remove self-similarity)
    n = normalized_matrix.shape[0]
    for i in range(n):
        normalized_matrix[i, i] = 0.0

    # DP arrays
    dp = np.zeros(n, dtype=np.float64)  # dp[i] = maximum reward for segmenting 0..i
    segmentation = np.zeros(n, dtype=np.int32)  # segmentation[i] = start of cluster ending at i

    # Dynamic programming
    for i in range(n):
        # Try different cluster sizes, but respect token limits
        for j in range(i + 1):  # j is the start of potential cluster ending at i
            # Fast token count using precomputed cumulative sums
            if j == 0:
                cluster_tokens = cumulative_tokens[i]
            else:
                cluster_tokens = cumulative_tokens[i] - cumulative_tokens[j - 1]

            if cluster_tokens <= max_chunk_size:
                # Calculate reward for cluster from j to i
                reward = _calculate_reward_numba(normalized_matrix, j, i)

                # Add reward from previous optimal segmentation
                adjusted_reward = reward
                if j > 0:
                    adjusted_reward += dp[j - 1]

                # Update if this segmentation is better
                if adjusted_reward > dp[i]:
                    dp[i] = adjusted_reward
                    segmentation[i] = j

    return segmentation

@njit
def _get_cluster_tokens_numba(cumulative_tokens: np.ndarray, start: int, end: int) -> int:
    """
    Numba-optimized function to get token count for cluster from start to end (inclusive) in O(1) time.

    Args:
        cumulative_tokens: Precomputed cumulative token sums
        start: Start sentence index
        end: End sentence index (inclusive)

    Returns:
        Total token count for the cluster
    """
    if start == 0:
        return int(cumulative_tokens[end])
    return int(cumulative_tokens[end] - cumulative_tokens[start - 1])

class ImprovedClusterSemanticChunker(BaseChunker):
    """
    Advanced semantic chunker that creates coherent chunks by clustering semantically similar sentences.

    This chunker solves the problem of maintaining semantic coherence in text chunks by:
    - Using NLTK for proper sentence boundary detection (handles abbreviations, decimals, etc.)
    - Generating embeddings for each sentence to capture semantic meaning
    - Finding optimal sentence clusters that maximize internal semantic similarity
    - Using dynamic programming to ensure global optimization while respecting token limits
    - Leveraging Numba for high-performance computation on large documents

    The result is chunks where sentences within each chunk are semantically related,
    leading to better context preservation for RAG applications compared to simple
    token-based or recursive splitting methods.
    """

    def __init__(
        self,
        doc_id: int,
        embedder: Embedder = None,
        max_chunk_size: int = 400,
        **kwargs
    ):
        """
        Initialize the ImprovedClusterSemanticChunker.

        Args:
            doc_id: Document ID for tracking chunks
            embedder: Custom embedder instance (creates new one if None)
            max_chunk_size: Maximum tokens per final chunk
            **kwargs: Additional arguments
        """
        # Initialize custom embedder (singleton pattern)
        self.embedder = embedder or Embedder()

        # Initialize Gemma tokenizer for accurate token counting
        self.tokenizer = GemmaSingletonTokenizer()

        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt_try')
        except LookupError:
                nltk.download('punkt_tab')

        self._doc_id = doc_id
        self._chunk_size = max_chunk_size

        logger.info(f"Initialized ImprovedClusterSemanticChunker with max_chunk_size={max_chunk_size}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens using Gemma tokenizer for accuracy."""
        return self.tokenizer.count_tokens(text)

    def _precompute_token_data(self, sentences: List[str]) -> None:
        """
        Pre-compute token counts and cumulative sums for fast O(1) cluster token counting.

        This eliminates the need to repeatedly tokenize cluster text during DP optimization.

        Args:
            sentences: List of sentence strings
        """
        logger.info("Pre-computing token counts for optimization...")

        # Store sentences for caching
        self.sentences = sentences

        # Pre-compute token count for each sentence using Gemma tokenizer
        self.sentence_tokens = np.array([self._count_tokens(s) for s in sentences], dtype=np.int32)

        # Create cumulative sum for O(1) range queries
        self.cumulative_tokens = np.cumsum(self.sentence_tokens, dtype=np.int32)

        logger.info(f"Pre-computed token data for {len(sentences)} sentences")

    def _get_cluster_tokens(self, start: int, end: int) -> int:
        """
        Get token count for cluster from start to end (inclusive) in O(1) time.

        Uses pre-computed cumulative sums instead of re-tokenizing cluster text.

        Args:
            start: Start sentence index
            end: End sentence index (inclusive)

        Returns:
            Total token count for the cluster
        """
        return _get_cluster_tokens_numba(self.cumulative_tokens, start, end)

    @lru_cache(maxsize=20000)
    def _get_cluster_text(self, start: int, end: int) -> str:
        """
        Get cluster text with LRU caching to avoid repeated string operations.

        Args:
            start: Start sentence index
            end: End sentence index (inclusive)

        Returns:
            Combined text for the cluster
        """
        return ' '.join(self.sentences[start:end+1])

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into proper sentences using NLTK's sophisticated sentence boundary detection.

        NLTK's sent_tokenize handles complex cases like:
        - Abbreviations (Dr., Mr., etc.)
        - Decimal numbers (3.14, 2.5)
        - Multiple punctuation marks
        - Quotes and parentheses

        Args:
            text: Input text to split

        Returns:
            List of clean sentences
        """
        # Clean up the text first
        text = re.sub(r'\s+', ' ', text.strip())

        # Use NLTK for proper sentence tokenization
        sentences = sent_tokenize(text)

        # Clean up each sentence
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Only add non-empty sentences
                cleaned_sentences.append(sentence)

        logger.info(f"Split text into {len(cleaned_sentences)} sentences using NLTK")
        return cleaned_sentences

    def _get_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Calculate similarity matrix between all sentence pairs using your custom embedder.

        Uses the Alibaba-NLP/gte-multilingual-base model to generate embeddings and compute
        cosine similarity between all sentence pairs. This creates a semantic map of how
        related each sentence is to every other sentence.

        Args:
            sentences: List of sentence strings

        Returns:
            NxN similarity matrix where N is number of sentences
        """
        BATCH_SIZE = 100  # Process in batches to manage memory
        N = len(sentences)

        logger.info(f"Calculating similarity matrix for {N} sentences using custom embedder")

        # Generate embeddings in batches
        embedding_matrix = None
        for i in range(0, N, BATCH_SIZE):
            batch_sentences = sentences[i:i+BATCH_SIZE]

            # Use custom embedder to generate embeddings (256-dimensional from Alibaba-NLP model)
            embeddings = self.embedder.embed_texts(batch_sentences, are_queries=False)

            # Convert embeddings list to numpy array
            batch_embedding_matrix = np.array(embeddings)

            # Append to main embedding matrix
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        # Calculate cosine similarity matrix
        # Since embeddings are already normalized, dot product gives cosine similarity
        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        logger.info(f"Generated similarity matrix with shape {similarity_matrix.shape}")
        return similarity_matrix

    def _optimal_segmentation(self, matrix: np.ndarray, sentences: List[str]) -> List[Tuple[int, int]]:
        """
        Find optimal segmentation using Numba-optimized dynamic programming with token-aware clustering.

        This algorithm finds the segmentation that maximizes semantic coherence
        while respecting token limits for each cluster. It uses dynamic programming
        to ensure the globally optimal solution rather than greedy local decisions.

        Args:
            matrix: Similarity matrix between sentences
            sentences: List of sentence strings for token counting

        Returns:
            List of (start, end) tuples representing optimal clusters
        """
        # Calculate mean similarity for normalization
        n = matrix.shape[0]
        upper_triangular_indices = np.triu_indices(n, k=1)
        mean_similarity = np.mean(matrix[upper_triangular_indices])

        logger.info(f"Starting Numba-optimized DP with mean similarity: {mean_similarity:.4f}")

        # Use Numba-optimized segmentation
        segmentation = _optimal_segmentation_numba(
            matrix,
            self.sentence_tokens,
            self.cumulative_tokens,
            self._chunk_size,
            mean_similarity
        )

        # Reconstruct optimal clusters from segmentation array
        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        logger.info(f"Found {len(clusters)} optimal clusters using Numba optimization")
        return clusters

    def _merge_small_chunks(self, clusters: List[Tuple[int, int]], sentences: List[str]) -> List[Tuple[int, int]]:
        """
        Post-process clusters to merge small adjacent chunks when possible.

        This helps eliminate tiny chunks by combining them with neighbors
        while respecting the maximum chunk size limit. Small chunks can hurt
        RAG performance by providing insufficient context.

        Args:
            clusters: List of (start, end) cluster tuples
            sentences: List of sentence strings

        Returns:
            List of merged cluster tuples
        """
        if len(clusters) <= 1:
            return clusters

        merged = []
        current_cluster = clusters[0]

        for next_cluster in clusters[1:]:
            # Check if we can merge current and next clusters
            combined_start = current_cluster[0]
            combined_end = next_cluster[1]
            combined_tokens = self._get_cluster_tokens(combined_start, combined_end)

            # Also check if either cluster is small (good candidate for merging)
            current_tokens = self._get_cluster_tokens(current_cluster[0], current_cluster[1])
            next_tokens = self._get_cluster_tokens(next_cluster[0], next_cluster[1])

            should_merge = (
                combined_tokens <= self._chunk_size and
                (current_tokens < 60 or next_tokens < 60)  # Merge if either is small
            )

            if should_merge:
                # Merge clusters
                current_cluster = (combined_start, combined_end)
                logger.debug(f"Merged small chunks: {current_tokens} + {next_tokens} = {combined_tokens} tokens")
            else:
                # Keep current cluster and move to next
                merged.append(current_cluster)
                current_cluster = next_cluster

        # Add the final cluster
        merged.append(current_cluster)

        logger.info(f"Post-processing: {len(clusters)} -> {len(merged)} chunks after merging")
        return merged

    def split_text(self, text: str) -> List[tuple]:
        """
        Split text into semantically coherent chunks using Numba-optimized dynamic programming.

        This is the main method that orchestrates the entire semantic chunking process:
        1. Split text into sentences using NLTK's robust sentence boundary detection
        2. Pre-compute token data using Gemma tokenizer for fast DP optimization
        3. Generate embeddings for each sentence using your custom Alibaba-NLP embedder
        4. Calculate semantic similarity matrix between all sentence pairs
        5. Find optimal clustering using Numba-optimized dynamic programming
        6. Post-process to merge small chunks for better context
        7. Combine clusters into final chunks with accurate token counts

        Args:
            text: Input text to chunk

        Returns:
            List of tuples in format: (doc_id, chunk_text, token_count, "semantic", True)
        """
        logger.info("Starting Numba-optimized semantic chunking process")

        # Step 1: Split text into proper sentences using NLTK
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            # If only one sentence, return as-is
            if sentences:
                token_count = self._count_tokens(sentences[0])
                return [(self._doc_id, sentences[0], token_count, "semantic", True)]
            else:
                return []

        # Step 2: Pre-compute token data for optimization
        self._precompute_token_data(sentences)

        # Step 3: Calculate similarity matrix using custom embedder
        similarity_matrix = self._get_similarity_matrix(sentences)

        # Step 4: Find optimal clusters using Numba-optimized DP
        clusters = self._optimal_segmentation(similarity_matrix, sentences)

        # Step 5: Post-process to merge small chunks
        merged_clusters = self._merge_small_chunks(clusters, sentences)

        # Step 6: Combine clusters into final chunks
        result = []
        for i, (start, end) in enumerate(merged_clusters):
            # Combine sentences in cluster
            chunk_text = self._get_cluster_text(start, end)
            token_count = self._get_cluster_tokens(start, end)

            # Add to result in required format (changed from "cluster" to "semantic" for clarity)
            result.append((self._doc_id, chunk_text, token_count, "semantic", True))

            logger.debug(f"Final Chunk {i+1}: sentences {start}-{end}, {token_count} tokens")

        logger.info(f"Created {len(result)} final semantic chunks using Numba optimization")
        return result


# Example usage and testing
if __name__ == "__main__":
    # Initialize the improved semantic chunker
    chunker = ImprovedClusterSemanticChunker(
        doc_id=1,
        max_chunk_size=400
    )

    # Test with sample text that has clear semantic groupings
    test_text = """


        """

    # Generate semantic chunks
    chunks = chunker.split_text(test_text)

    print("=== Semantic Clustering Results (with Gemma Tokenizer) ===")
    print(f"Generated {len(chunks)} semantic chunks:")
    print("=" * 80)

    for i, (doc_id, chunk_text, token_count, unit_type, is_processed) in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Doc ID: {doc_id}")
        print(f"  Token Count: {token_count}")
        print(f"  Unit Type: {unit_type}")
        print(f"  Is Processed: {is_processed}")
        print(f"  Text: {chunk_text}")
        print("-" * 40)

    print(f"\nTotal chunks: {len(chunks)}")
    total_tokens = sum(token_count for _, _, token_count, _, _ in chunks)
    print(f"Total tokens: {total_tokens}")
    avg_tokens = total_tokens / len(chunks) if chunks else 0
    print(f"Average tokens per chunk: {avg_tokens:.1f}")