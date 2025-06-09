"""
Cluster Semantic Chunker following the main author's implementation pattern.
Adapted to use your custom embedder and tokenizer while maintaining the same logic flow.
Optimized with Numba for high-performance computation.
"""
from chunkers.base import BaseChunker
from chunkers.recursive_chunker import RecursiveTokenChunker  # Your recursive chunker
from typing import List
import numpy as np
from emb.embedder import Embedder
from emb.gemma_tokenizer import GemmaSingletonTokenizer
import logging
from numba import njit

logger = logging.getLogger(__name__)


@njit
def _calculate_reward_numba(matrix: np.ndarray, start: int, end: int) -> float:
    """
    Numba-optimized reward calculation for clustering chunks from start to end.

    Args:
        matrix: Similarity matrix
        start: Start index of cluster
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

    Args:
        matrix: Normalized similarity matrix
        max_cluster_size: Maximum number of chunks per cluster

    Returns:
        Array where segmentation[i] = start index of cluster ending at i
    """
    n = matrix.shape[0]
    dp = np.zeros(n, dtype=np.float64)  # dp[i] = maximum reward for segmenting 0..i
    segmentation = np.zeros(n, dtype=np.int32)  # segmentation[i] = start of cluster ending at i

    # Dynamic programming
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
    Cluster Semantic Chunker following the main author's implementation pattern.

    This implementation:
    1. Uses RecursiveTokenChunker to create initial small chunks (like sentences)
    2. Generates embeddings for these chunks using your custom embedder
    3. Builds similarity matrix between all chunk pairs
    4. Uses dynamic programming to find optimal clustering
    5. Returns final merged chunks
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
            embedding_function: Custom embedding function (uses your embedder if None)
            max_chunk_size: Maximum size of final chunks in tokens
            min_chunk_size: Size of initial chunks for clustering
            **kwargs: Additional arguments
        """
        # Initialize your recursive chunker for initial splitting
        self.splitter = RecursiveTokenChunker(
            doc_id=doc_id,
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        # Initialize your custom embedder and tokenizer
        self.embedder = Embedder()
        self.tokenizer = GemmaSingletonTokenizer()

        # Set up embedding function
        if embedding_function is None:
            # Use your custom embedder
            self.embedding_function = self._custom_embedding_function
        else:
            self.embedding_function = embedding_function

        self._doc_id = doc_id
        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size // min_chunk_size

        logger.info(f"Initialized ClusterSemanticChunker with Numba optimization - max_chunk_size={max_chunk_size}, min_chunk_size={min_chunk_size}")

    def _custom_embedding_function(self, texts: List[str]) -> List[List[float]]:
        """
        Wrapper for your custom embedder to match the expected interface.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        return self.embedder.embed_texts(texts, are_queries=False)

    def _get_similarity_matrix(self, embedding_function, sentences: List[str]) -> np.ndarray:
        """
        Calculate similarity matrix between all sentence pairs.

        Args:
            embedding_function: Function to generate embeddings
            sentences: List of sentence/chunk strings

        Returns:
            NxN similarity matrix
        """
        BATCH_SIZE = 500  # Process in batches to manage memory
        N = len(sentences)

        logger.info(f"Calculating similarity matrix for {N} chunks")

        # Generate embeddings in batches
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

        Args:
            matrix: Similarity matrix
            start: Start index of cluster
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

        Args:
            matrix: Similarity matrix between chunks
            max_cluster_size: Maximum number of chunks per cluster
            window_size: Window size for local density calculation (not used in simple version)

        Returns:
            List of (start, end) tuples representing optimal clusters
        """
        # Normalize matrix by subtracting mean
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value
        np.fill_diagonal(matrix, 0)  # Remove self-similarity

        n = matrix.shape[0]
        logger.info(f"Starting Numba-optimized DP with mean similarity: {mean_value:.4f}")

        # Use Numba-optimized segmentation
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

    def _merge_small_chunks(self, clusters: List[tuple], sentences: List[str], min_final_size: int = 80) -> List[tuple]:
        """
        Post-process clusters to merge small final chunks when possible.

        Args:
            clusters: List of (start, end) cluster tuples
            sentences: List of sentence strings
            min_final_size: Minimum tokens for final chunks

        Returns:
            List of merged cluster tuples
        """
        if len(clusters) <= 1:
            return clusters

        merged = []
        current_cluster = clusters[0]

        for next_cluster in clusters[1:]:
            # Calculate token counts
            current_text = ' '.join(sentences[current_cluster[0]:current_cluster[1]+1])
            current_tokens = self.tokenizer.count_tokens(current_text)

            next_text = ' '.join(sentences[next_cluster[0]:next_cluster[1]+1])
            next_tokens = self.tokenizer.count_tokens(next_text)

            # Check if we can merge
            combined_text = ' '.join(sentences[current_cluster[0]:next_cluster[1]+1])
            combined_tokens = self.tokenizer.count_tokens(combined_text)

            should_merge = (
                combined_tokens <= self._chunk_size and
                (current_tokens < min_final_size or next_tokens < min_final_size)
            )

            if should_merge:
                # Merge clusters
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
        Split text into semantically coherent chunks.

        This is the main method that:
        1. Uses RecursiveTokenChunker to create initial small chunks
        2. Generates embeddings for these chunks
        3. Finds optimal clustering using dynamic programming
        4. Post-processes to merge very small final chunks
        5. Returns merged chunks in the required format

        Args:
            text: Input text to chunk

        Returns:
            List of tuples in format: (doc_id, chunk_text, token_count, "cluster", True)
        """
        logger.info("Starting Numba-optimized cluster semantic chunking process")

        # Step 1: Split text into initial small chunks using recursive chunker
        initial_chunks = self.splitter.split_text(text)

        # Extract just the text from the tuples returned by recursive chunker
        sentences = [chunk_text for _, chunk_text, _, _, _ in initial_chunks]

        logger.info(f"Created {len(sentences)} initial chunks for clustering")

        if len(sentences) <= 1:
            # If only one chunk, return as-is but change type to "cluster"
            if sentences:
                token_count = self.tokenizer.count_tokens(sentences[0])
                return [(self._doc_id, sentences[0], token_count, "cluster", True)]
            else:
                return []

        # Step 2: Calculate similarity matrix using custom embedder
        similarity_matrix = self._get_similarity_matrix(self.embedding_function, sentences)

        # Step 3: Find optimal clusters using dynamic programming
        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size=self.max_cluster)

        # Step 4: Post-process to merge very small chunks
        merged_clusters = self._merge_small_chunks(clusters, sentences, min_final_size=80)

        # Step 5: Combine clusters into final chunks
        result = []
        for i, (start, end) in enumerate(merged_clusters):
            # Combine sentences in cluster
            chunk_text = ' '.join(sentences[start:end+1])
            token_count = self.tokenizer.count_tokens(chunk_text)

            # Add to result in required format
            result.append((self._doc_id, chunk_text, token_count, "cluster", True))

            logger.debug(f"Final Cluster {i+1}: chunks {start}-{end}, {token_count} tokens")

        logger.info(f"Created {len(result)} final cluster chunks using Numba optimization")
        return result


# Example usage and testing
if __name__ == "__main__":
    # Initialize the cluster semantic chunker
    chunker = ClusterSemanticChunker(
        doc_id=1,
        max_chunk_size=400,
        min_chunk_size=50
    )

    # Test with sample text
    test_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    
    Machine learning is a subset of artificial intelligence that focuses on 
    algorithms that can learn and make decisions from data. Deep learning, 
    a subset of machine learning, uses neural networks with multiple layers 
    to model and understand complex patterns in data.
    
    Natural language processing (NLP) is another important area of AI that 
    deals with the interaction between computers and human language. It involves 
    developing algorithms that can understand, interpret, and generate human language.
    """

    # Generate cluster chunks
    chunks = chunker.split_text(test_text)

    print("=== Cluster Semantic Chunking Results ===")
    print(f"Generated {len(chunks)} cluster chunks:")
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