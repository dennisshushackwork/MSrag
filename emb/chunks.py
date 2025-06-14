# External imports
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Internal Imports:
from emb.embedder import Qwen3Embedder
from postgres.embedding import EmbeddingQueries

# Initialisation of logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChunkEmbedder:
    """
    This class embeds chunks from the database concurrently using multiple threads.
    """
    def __init__(self):
        self.max_tokens_per_batch = 7000  # Maximum allowed tokens per embedding call.
        self.batch_size = 100             # Batch size for chunks.
        self.max_workers = 4

    def batch_embed_texts(self, embedder: Qwen3Embedder, texts_and_token_count: List[tuple]) -> List:
        """
        Batches texts so that each embedding call doesn't exceed max_tokens_per_batch.
        `texts_and_token_count` is a list of tuples: (text, token_count).
        """
        batched_results = []
        current_batch = []
        current_token_count = 0

        for text, token_count in texts_and_token_count:
            if current_batch and (current_token_count + token_count > self.max_tokens_per_batch):
                logger.info("Embedding a batch with %d texts (%d tokens)", len(current_batch), current_token_count)
                print(current_batch)
                batch_result = embedder.embed_texts(current_batch)
                batched_results.extend(batch_result)
                current_batch = []
                current_token_count = 0
            current_batch.append(text)
            current_token_count += token_count

        if current_batch:
            logger.info("Embedding final batch with %d texts (%d tokens)", len(current_batch), current_token_count)
            batch_result = embedder.embed_texts(current_batch)
            batched_results.extend(batch_result)
        return batched_results

    def get_chunks_in_batches(self):
        """
        Yields batches of chunks from the database.
        """
        offset = 0
        while True:
            with EmbeddingQueries() as db:
                batch = db.get_chunk_batches(self.batch_size, offset)
            if not batch:
                break
            yield batch
            offset += self.batch_size

    def process_single_batch(self, batch: List[tuple]) -> int:
        """
        Process a single batch of chunks:
          - Creates its own EmbeddingService instance.
          - Embeds the batch.
          - Updates the database.
        Returns the number of processed chunks.
        """
        # Each thread instantiates its own embedder_mlr_test.
        embedder = Qwen3Embedder()
        texts_and_tokens = [(c[1], c[2]) for c in batch]  # c[1]=text, c[2]=token_count.
        chunk_ids = [c[0] for c in batch]                   # c[0]=chunk_id.
        logger.info("Thread starting processing %d chunks.", len(batch))
        emb_results = self.batch_embed_texts(embedder, texts_and_tokens)
        update_tuples = list(zip(chunk_ids, emb_results))
        with EmbeddingQueries() as db:
            db.update_emb_chunk(update_tuples)
        logger.info("Thread finished processing %d chunks.", len(batch))
        return len(batch)

    def process_chunk_emb_batches(self) -> None:
        """
        Processes all chunks marked for embedding using multiple threads.
        """
        with EmbeddingQueries() as db:
            chunk_count = db.count_chunks_to_embed()

        if chunk_count == 0:
            logger.info("No chunks require embedding.")
            return
        logger.info("Found %d chunks that need embedding.", chunk_count)
        total_processed_count = 0
        batch_count = 0

        # Use ThreadPoolExecutor to run two threads concurrently.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for batch in self.get_chunks_in_batches():
                batch_count += 1
                future = executor.submit(self.process_single_batch, batch)
                futures.append(future)

            for future in as_completed(futures):
                processed = future.result()
                total_processed_count += processed
                logger.info("Processed %d/%d chunks so far.", total_processed_count, chunk_count)

        logger.info("Completed embedding process. Processed %d chunks in %d batches.", total_processed_count, batch_count)


if __name__ == "__main__":
    emb = ChunkEmbedder()
    emb.process_chunk_emb_batches()


