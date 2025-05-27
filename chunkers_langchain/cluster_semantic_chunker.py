from base_chunker import BaseChunker
from typing import List
import numpy as np
from recursive_token_chunker import RecursiveTokenChunker
from emb.embedder import Embedder


class ClusterSemanticChunker(BaseChunker):
    def __init__(self, embedder=None, max_chunk_size=400, min_chunk_size=50):
        """
        Initialize the ClusterSemanticChunker with custom embedder.

        Args:
            embedder: Your custom Embedder instance. If None, creates a new one.
            max_chunk_size: Maximum size of final chunks in tokens
            min_chunk_size: Minimum size for initial splits in tokens
        """
        # Initialize embedder
        if embedder is None:
            embedder = Embedder()
        self.embedder = embedder

        # Create the recursive token chunker using your custom embedder
        self.splitter = RecursiveTokenChunker.from_custom_embedder(
            embedder=embedder,
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size // min_chunk_size

    def _get_similarity_matrix(self, sentences):
        """
        Calculate similarity matrix using your custom embedder.
        Processes sentences in batches to handle memory efficiently.
        """
        BATCH_SIZE = 500
        N = len(sentences)
        embedding_matrix = None

        for i in range(0, N, BATCH_SIZE):
            batch_sentences = sentences[i:i + BATCH_SIZE]

            # Use your custom embedder to get embeddings
            embeddings = self.embedder.embed_texts(batch_sentences, are_queries=False)

            # Convert embeddings list of lists to numpy array
            batch_embedding_matrix = np.array(embeddings)

            # Append the batch embedding matrix to the main embedding matrix
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        # Calculate cosine similarity matrix
        # Since embeddings are already normalized, dot product gives cosine similarity
        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
        return similarity_matrix

    def _calculate_reward(self, matrix, start, end):
        """Calculate reward for a cluster segment."""
        sub_matrix = matrix[start:end + 1, start:end + 1]
        return np.sum(sub_matrix)

    def _optimal_segmentation(self, matrix, max_cluster_size, window_size=3):
        """
        Find optimal segmentation using dynamic programming.
        """
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value  # Normalize the matrix
        np.fill_diagonal(matrix, 0)  # Set diagonal to 0 to avoid trivial solutions

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                if i - size + 1 >= 0:
                    reward = self._calculate_reward(matrix, i - size + 1, i)
                    adjusted_reward = reward

                    if i - size >= 0:
                        adjusted_reward += dp[i - size]

                    if adjusted_reward > dp[i]:
                        dp[i] = adjusted_reward
                        segmentation[i] = i - size + 1

        # Reconstruct clusters
        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        return clusters

    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks
        """
        # Step 1: Split text into initial small chunks
        sentences = self.splitter.split_text(text)

        if len(sentences) <= 1:
            return sentences

        # Step 2: Get similarity matrix using custom embedder
        similarity_matrix = self._get_similarity_matrix(sentences)

        # Step 3: Find optimal segmentation
        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size=self.max_cluster)

        # Step 4: Combine sentences within each cluster
        docs = []
        for start, end in clusters:
            chunk_text = ' '.join(sentences[start:end + 1])
            # Verify chunk size with embedder
            token_count = self.embedder.count_tokens(chunk_text)
            if token_count <= self._chunk_size:
                docs.append(chunk_text)
            else:
                # If chunk is too large, split it further
                sub_chunks = self.splitter.split_text(chunk_text)
                docs.extend(sub_chunks)

        return docs


# Example usage and testing
if __name__ == "__main__":
    # Create custom embedder
    embedder = Embedder()

    # Create cluster semantic chunker
    chunker = ClusterSemanticChunker(
        embedder=embedder,
        max_chunk_size=1000,  # 1000 tokens max per chunk
        min_chunk_size=100  # 100 tokens for initial splits
    )

    # Sample text for testing
    sample_text = """
    Barack Hussein Obama II (born August 4, 1961) is an American politician who was the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African American president in American history. Obama previously served as a U.S. senator representing Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.

    Born in Honolulu, Hawaii, Obama graduated from Columbia University in 1983 with a Bachelor of Arts degree in political science and later worked as a community organizer in Chicago. In 1988, Obama enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review. He became a civil rights attorney and an academic, teaching constitutional law at the University of Chicago Law School from 1992 to 2004.

    In 1996, Obama was elected to represent the 13th district in the Illinois Senate, a position he held until 2004, when he successfully ran for the U.S. Senate. In the 2008 presidential election, after a close primary campaign against Hillary Clinton, he was nominated by the Democratic Party for president. Obama selected Joe Biden as his running mate and defeated Republican nominee John McCain and his running mate Sarah Palin.

    Obama was awarded the 2009 Nobel Peace Prize for efforts in international diplomacy, a decision which drew both criticism and praise. During his first term, his administration responded to the 2008 financial crisis with measures including the American Recovery and Reinvestment Act of 2009, a major stimulus package to guide the economy in recovering from the Great Recession.
    """

    print("🧠 Cluster Semantic Chunking with Custom Embedder")
    print("=" * 60)
    print(f"📄 Input text: {embedder.count_tokens(sample_text)} tokens")

    # Split the text
    chunks = chunker.split_text(sample_text)

    print(f"✅ Created {len(chunks)} semantic chunks:")

    total_tokens = 0
    for i, chunk in enumerate(chunks):
        token_count = embedder.count_tokens(chunk)
        total_tokens += token_count
        print(f"\nChunk {i + 1} ({token_count} tokens):")
        print("-" * 40)
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

    print(f"\n📊 Summary:")
    print(f"  • Total chunks: {len(chunks)}")
    print(f"  • Total tokens: {total_tokens}")
    print(f"  • Average tokens per chunk: {total_tokens / len(chunks):.1f}")
    print(f"  • Max chunk size setting: {chunker._chunk_size} tokens")