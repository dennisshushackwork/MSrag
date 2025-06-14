"""
Cluster Semantic Chunker with Position Tracking

This implementation follows the main author's implementation pattern, adapted to use
your custom embedder_mlr_test and tokenizer while maintaining the same logic flow.
Optimized with Numba for high-performance computation.

Enhanced with precise position tracking to map cluster chunks back to their
exact locations in the original document.

Key Features:
- Uses RecursiveTokenChunker to create initial small chunks with position tracking
- Generates embeddings for these chunks using your custom embedder_mlr_test
- Builds similarity matrix between all chunk pairs
- Uses dynamic programming to find optimal clustering
- Maps cluster boundaries back to original document positions
- Returns final merged chunks with accurate start/end indices
"""
from chunkers.base import BaseChunker
from chunkers.recursive_chunker import PositionTrackingRecursiveChunker, ChunkWithPosition
from typing import List, Tuple
import numpy as np
from emb.embedder_old import Embedder
from emb.gemma_tokenizer import GemmaSingletonTokenizer
import logging
from numba import njit

logger = logging.getLogger(__name__)


@njit
def _calculate_reward_numba(matrix: np.ndarray, start: int, end: int) -> float:
    """
    Numba-optimized reward calculation for clustering chunks from start to end.

    This function calculates the total similarity reward for clustering chunks
    together by summing all pairwise similarities within the cluster.

    Args:
        matrix: Similarity matrix between chunks
        start: Start index of cluster (inclusive)
        end: End index of cluster (inclusive)

    Returns:
        Total similarity reward for this cluster
    """
    total_reward = 0.0
    for i in range(start, end + 1):
        for j in range(start, end + 1):
            total_reward += matrix[i, j]
    return total_reward


@njit
def _optimal_segmentation_numba(
    matrix: np.ndarray,
    max_cluster_size: int
) -> np.ndarray:
    """
    Numba-optimized dynamic programming for optimal segmentation.

    This implements the core dynamic programming algorithm to find the optimal
    way to cluster consecutive chunks to maximize total similarity while
    respecting cluster size constraints.

    

    Args:
        matrix: Normalized similarity matrix between chunks
        max_cluster_size: Maximum number of chunks per cluster

    Returns:
        Array where segmentation[i] = start index of cluster ending at i
    """
    n = matrix.shape[0]

    # dp[i] = maximum reward for segmenting chunks 0..i
    dp = np.zeros(n, dtype=np.float64)

    # segmentation[i] = start of cluster ending at i
    segmentation = np.zeros(n, dtype=np.int32)

    # Dynamic programming to find optimal segmentation
    for i in range(n):
        for size in range(1, max_cluster_size + 1):
            if i - size + 1 >= 0:
                # Calculate reward for cluster from (i-size+1) to i
                reward = _calculate_reward_numba(matrix, i - size + 1, i)

                # Add reward from previous optimal segmentation
                adjusted_reward = reward
                if i - size >= 0:
                    adjusted_reward += dp[i - size]

                # Update if this segmentation is better
                if adjusted_reward > dp[i]:
                    dp[i] = adjusted_reward
                    segmentation[i] = i - size + 1

    return segmentation


class ClusterSemanticChunker(BaseChunker):
    """
    Cluster Semantic Chunker with Position Tracking

    This implementation follows the main author's implementation pattern, enhanced
    with precise position tracking capabilities.

    The chunking process:
    1. Uses PositionTrackingRecursiveChunker to create initial small chunks with positions
    2. Generates embeddings for these chunks using your custom embedder_mlr_test
    3. Builds similarity matrix between all chunk pairs
    4. Uses dynamic programming to find optimal clustering
    5. Maps cluster boundaries back to original document positions
    6. Returns final merged chunks with accurate start/end indices
    """

    def __init__(
        self,
        doc_id: int,
        embedding_function=None,
        max_chunk_size: int = 400,
        min_chunk_size: int = 50,
        **kwargs
    ):
        """
        Initialize the ClusterSemanticChunker.

        Args:
            doc_id: Document ID for tracking chunks
            embedding_function: Custom embedding function (uses your embedder_mlr_test if None)
            max_chunk_size: Maximum size of final chunks in tokens
            min_chunk_size: Size of initial chunks for clustering
            **kwargs: Additional arguments
        """
        # Initialize your recursive chunker for initial splitting with position tracking
        self.splitter = PositionTrackingRecursiveChunker(
            doc_id=doc_id,
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        # Initialize your custom embedder_mlr_test and tokenizer
        self.embedder = Embedder()
        self.tokenizer = GemmaSingletonTokenizer()

        # Set up embedding function
        if embedding_function is None:
            # Use your custom embedder_mlr_test
            self.embedding_function = self._custom_embedding_function
        else:
            self.embedding_function = embedding_function

        self._doc_id = doc_id
        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size // min_chunk_size

        logger.info(f"Initialized ClusterSemanticChunker with position tracking - max_chunk_size={max_chunk_size}, min_chunk_size={min_chunk_size}")

    def _custom_embedding_function(self, texts: List[str]) -> List[List[float]]:
        """
        Wrapper for your custom embedder_mlr_test to match the expected interface.

        This method adapts your embedder_mlr_test to work with the clustering algorithm,
        ensuring embeddings are generated for similarity calculation.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors for similarity calculation
        """
        return self.embedder.embed_texts(texts, are_queries=False)

    def _get_similarity_matrix(self, embedding_function, sentences: List[str]) -> np.ndarray:
        """
        Calculate similarity matrix between all sentence pairs.

        This method generates embeddings for all chunks and computes their
        pairwise cosine similarities, forming the basis for clustering decisions.

        Args:
            embedding_function: Function to generate embeddings
            sentences: List of sentence/chunk strings

        Returns:
            NxN similarity matrix where entry (i,j) is similarity between chunks i and j
        """
        BATCH_SIZE = 500  # Process in batches to manage memory
        N = len(sentences)

        logger.info(f"Calculating similarity matrix for {N} chunks")

        # Generate embeddings in batches to handle large documents
        embedding_matrix = None
        for i in range(0, N, BATCH_SIZE):
            batch_sentences = sentences[i:i+BATCH_SIZE]
            embeddings = embedding_function(batch_sentences)

            # Convert embeddings list to numpy array
            batch_embedding_matrix = np.array(embeddings)

            # Append to main embedding matrix
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        # Calculate cosine similarity matrix (embeddings are already normalized)
        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        logger.info(f"Generated similarity matrix with shape {similarity_matrix.shape}")
        return similarity_matrix

    def _calculate_reward(self, matrix: np.ndarray, start: int, end: int) -> float:
        """
        Calculate reward for clustering chunks from start to end using Numba optimization.

        This method leverages the Numba-optimized function for fast reward calculation,
        which is critical for the dynamic programming algorithm's performance.

        Args:
            matrix: Similarity matrix between chunks
            start: Start index of cluster (inclusive)
            end: End index of cluster (inclusive)

        Returns:
            Total similarity reward for this cluster
        """
        return _calculate_reward_numba(matrix, start, end)

    def _optimal_segmentation(
        self,
        matrix: np.ndarray,
        max_cluster_size: int,
        window_size: int = 3
    ) -> List[tuple]:
        """
        Find optimal segmentation using Numba-optimized dynamic programming.

        This method implements the core clustering algorithm that determines the
        optimal way to group consecutive chunks to maximize semantic coherence
        while respecting size constraints.

        Args:
            matrix: Similarity matrix between chunks
            max_cluster_size: Maximum number of chunks per cluster
            window_size: Window size for local density calculation (not used in simple version)

        Returns:
            List of (start, end) tuples representing optimal clusters
        """
        # Normalize matrix by subtracting mean to focus on above-average similarities
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value
        np.fill_diagonal(matrix, 0)  # Remove self-similarity

        n = matrix.shape[0]
        logger.info(f"Starting Numba-optimized DP with mean similarity: {mean_value:.4f}")

        # Use Numba-optimized segmentation for performance
        segmentation = _optimal_segmentation_numba(matrix, max_cluster_size)

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

    def _merge_small_chunks(self, clusters: List[tuple], chunks: List[ChunkWithPosition], min_final_size: int = 80) -> List[tuple]:
        """
        Post-process clusters to merge small final chunks when possible.

        This method combines adjacent clusters if they're small and the combination
        doesn't exceed the maximum chunk size, helping to avoid tiny final chunks.

        Args:
            clusters: List of (start, end) cluster tuples
            chunks: List of ChunkWithPosition objects with token counts
            min_final_size: Minimum tokens for final chunks before considering merge

        Returns:
            List of merged cluster tuples
        """
        if len(clusters) <= 1:
            return clusters

        merged = []
        current_cluster = clusters[0]

        for next_cluster in clusters[1:]:
            # Calculate token counts for current and next clusters
            current_tokens = sum(chunks[i].token_count for i in range(current_cluster[0], current_cluster[1] + 1))
            next_tokens = sum(chunks[i].token_count for i in range(next_cluster[0], next_cluster[1] + 1))
            combined_tokens = current_tokens + next_tokens

            # Check if we should merge these clusters
            should_merge = (
                combined_tokens <= self._chunk_size and
                (current_tokens < min_final_size or next_tokens < min_final_size)
            )

            if should_merge:
                # Merge clusters by extending the end boundary
                current_cluster = (current_cluster[0], next_cluster[1])
                logger.debug(f"Merged small chunks: {current_tokens} + {next_tokens} = {combined_tokens} tokens")
            else:
                # Keep current cluster and move to next
                merged.append(current_cluster)
                current_cluster = next_cluster

        # Add the final cluster
        merged.append(current_cluster)

        logger.info(f"Post-processing: {len(clusters)} -> {len(merged)} chunks after merging small ones")
        return merged

    def split_text(self, text: str) -> List[tuple]:
        """
        Split text into semantically coherent chunks with precise position tracking.

        This is the main method that orchestrates the entire clustering process:
        1. Uses PositionTrackingRecursiveChunker to create initial small chunks with positions
        2. Generates embeddings for these chunks using your custom embedder_mlr_test
        3. Finds optimal clustering using dynamic programming
        4. Maps cluster boundaries back to original document positions
        5. Post-processes to merge very small final chunks
        6. Returns merged chunks with accurate position information

        Args:
            text: Input text to chunk

        Returns:
            List of tuples in format: (doc_id, chunk_text, token_count, "cluster", True, start_index, end_index)
        """
        logger.info("Starting cluster semantic chunking with position tracking")

        # Step 1: Split text into initial small chunks with position tracking
        initial_chunks = self.splitter.split_text_with_positions(text)

        # Extract just the text from the chunks for clustering analysis
        sentences = [chunk.text for chunk in initial_chunks]

        logger.info(f"Created {len(sentences)} initial chunks for clustering")

        if len(sentences) <= 1:
            # If only one chunk, return as-is but change type to "cluster"
            if initial_chunks:
                chunk = initial_chunks[0]
                return [(self._doc_id, chunk.text, chunk.token_count, "cluster", True,
                        chunk.start_index, chunk.end_index)]
            else:
                return []

        # Step 2: Calculate similarity matrix using custom embedder_mlr_test
        similarity_matrix = self._get_similarity_matrix(self.embedding_function, sentences)

        # Step 3: Find optimal clusters using dynamic programming
        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size=self.max_cluster)

        # Step 4: Post-process to merge very small chunks
        merged_clusters = self._merge_small_chunks(clusters, initial_chunks, min_final_size=80)

        # Step 5: Create final chunks with precise position mapping
        result = []
        for i, (start_idx, end_idx) in enumerate(merged_clusters):
            # Get the chunks in this cluster
            cluster_chunks = initial_chunks[start_idx:end_idx + 1]

            # Combine the text from all chunks in the cluster
            chunk_text = ' '.join(chunk.text for chunk in cluster_chunks)

            # Calculate total token count
            token_count = sum(chunk.token_count for chunk in cluster_chunks)

            # Determine start and end positions in original document
            # Use the first chunk's start and last chunk's end for precise mapping
            chunks_with_positions = [c for c in cluster_chunks if c.start_index is not None]

            if chunks_with_positions:
                # Map to original document positions
                start_index = chunks_with_positions[0].start_index
                end_index = chunks_with_positions[-1].end_index

                logger.debug(f"Cluster {i+1}: chunks {start_idx}-{end_idx}, {token_count} tokens, positions {start_index}-{end_index}")
            else:
                # Fallback: estimate positions or use None
                logger.warning(f"Cluster {i+1}: No position info available for chunks {start_idx}-{end_idx}")
                start_index = None
                end_index = None

            # Always create result tuple with consistent format
            result.append((
                self._doc_id,
                chunk_text,
                token_count,
                "cluster",
                True,
                start_index,
                end_index
            ))

        logger.info(f"Created {len(result)} final cluster chunks with position tracking")
        return result


# Example usage and testing
if __name__ == "__main__":
    # Initialize the cluster semantic chunker with position tracking
    chunker = ClusterSemanticChunker(
        doc_id=1,
        max_chunk_size=400,
        min_chunk_size=50
    )

    # Test with sample text that demonstrates semantic clustering
    test_text = """
    Good evening. If I were smart, I'd go home now. Mr. Speaker, Madam Vice President, 
    members of Congress, my fellow Americans. In January 1941, Franklin Roosevelt came 
    to this chamber to speak to the nation. And he said, "I address you at a moment 
    unprecedented in the history of the Union". Hitler was on the march. War was raging 
    in Europe. President Roosevelt's purpose was to wake up Congress and alert the 
    American people that this was no ordinary time.
    Good evening. If I were smart, I'd go home now. Mr. Speaker, Madam Vice President, 
    members of Congress, my fellow Americans. In January 1941, Franklin Roosevelt came 
    to this chamber to speak to the nation. And he said, "I address you at a moment 
    unprecedented in the history of the Union". Hitler was on the march. War was raging 
    in Europe. President Roosevelt's purpose was to wake up Congress and alert the 
    American people that this was no ordinary time.
    Good evening. If I were smart, I'd go home now. Mr. Speaker, Madam Vice President, 
    members of Congress, my fellow Americans. In January 1941, Franklin Roosevelt came 
    to this chamber to speak to the nation. And he said, "I address you at a moment 
    unprecedented in the history of the Union". Hitler was on the march. War was raging 
    in Europe. President Roosevelt's purpose was to wake up Congress and alert the 
    American people that this was no ordinary time.
    """

    # Generate cluster chunks with position tracking
    chunks = chunker.split_text(test_text)

    print("=== Cluster Semantic Chunking with Position Tracking Results ===")
    print(f"Generated {len(chunks)} cluster chunks:")
    print("=" * 80)

    for i, chunk_data in enumerate(chunks):
        print(chunk_data)
