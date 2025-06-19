"""
Entity Embedder: This class is optimised to embed all entities inside the Postgres Database
using multiple threads for improved performance.
"""

# External Imports:
import logging
import time  # Used for adding delays between retries
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Internal Imports:
from emb.embedder import Qwen3Embedder
from postgres.embedding import EmbeddingQueries

# Setting the logger:
# Configures a basic logger to output timestamped messages to the console.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EntityEmbedder:
    """
    This class embeds entities from a database concurrently using multiple threads.
    It includes robust error handling and a retry mechanism for failed batches.
    """

    def __init__(self):
        """Initializes the EntityEmbedder with configuration parameters."""
        self.max_tokens_per_batch = 10000  # Max tokens per API call to the embedding model.
        self.batch_size = 500  # Number of entities to fetch from the database at once.
        self.max_workers = 3  # Number of concurrent threads for processing batches.

    def batch_embed_texts(self, embedder: Qwen3Embedder, texts_and_token_count: List[tuple]) -> List:
        """
        Dynamically creates batches of texts for embedding to stay within token limits.

        This method ensures that each call to the embedding model API does not exceed
        the `max_tokens_per_batch` limit, which prevents API errors.

        Args:
            embedder (Qwen3Embedder): The embedding model instance.
            texts_and_token_count (List[tuple]): A list of tuples, where each tuple
                                                 contains the text to embed and its token count.

        Returns:
            List: A list containing all the generated embedding vectors.
        """
        batched_results = []
        current_batch = []
        current_token_count = 0

        for text, token_count in texts_and_token_count:
            # If adding the next text would exceed the token limit, process the current batch first.
            if current_batch and (current_token_count + token_count > self.max_tokens_per_batch):
                logger.info("Embedding a batch with %d texts (%d tokens)", len(current_batch), current_token_count)
                # Call the embedding model API.
                batch_result = embedder.embed_texts_custom_dim(texts=current_batch, embedding_dim=256, are_queries=False)
                batched_results.extend(batch_result)
                # Reset the batch after processing.
                current_batch = []
                current_token_count = 0

            # Add the current text to the new or existing batch.
            current_batch.append(text)
            current_token_count += token_count

        # After the loop, process any remaining texts in the final batch.
        if current_batch:
            logger.info("Embedding final batch with %d texts (%d tokens)", len(current_batch), current_token_count)
            batch_result = embedder.embed_texts_custom_dim(current_batch, embedding_dim=256, are_queries=False)
            batched_results.extend(batch_result)

        return batched_results

    def get_entities_in_batches(self):
        """
        A generator that fetches and yields batches of entities from the database.

        This method retrieves entities incrementally to avoid loading the entire dataset
        into memory at once.

        Yields:
            List[tuple]: A batch of entities from the database.
        """
        offset = 0
        while True:
            # Open a new database connection for each batch to ensure thread safety.
            with EmbeddingQueries() as db:
                batch = db.get_entities_batches(self.batch_size, offset)
            # If the database returns an empty batch, we have processed all entities.
            if not batch:
                break
            yield batch
            offset += self.batch_size

    def process_single_batch(self, batch: List[tuple]) -> int:
        """
        Processes a single batch of entities.

        This function is executed by each worker thread. It handles embedding the texts
        and updating the database for one batch. It is designed to raise exceptions
        on failure, which are then caught by the calling method.

        Args:
            batch (List[tuple]): A list of entity records to process.

        Returns:
            int: The number of entities successfully processed in the batch.

        Raises:
            Exception: Propagates any exception from the embedding or database update steps.
        """
        # Each thread instantiates its own embedder to avoid sharing resources across threads.
        embedder = Qwen3Embedder()
        entities_to_embed = [e for e in batch]
        update_tuples = []  # Stores (entity_id, embedding_vector) for bulk update.

        logger.info("Thread starting processing %d entities.", len(entities_to_embed))

        if entities_to_embed:
            # Extract names and count tokens for batching.
            names = [e[1] for e in entities_to_embed]
            names_tokens = [(name, embedder.count_tokens(name)) for name in names]

            # Get embedding results for the current batch's texts.
            emb_results = self.batch_embed_texts(embedder, list(zip(names, [t[1] for t in names_tokens])))

            # Prepare the data for the bulk database update.
            for idx, entity in enumerate(entities_to_embed):
                update_tuples.append((entity[0], emb_results[idx]))

        # Use a new database connection to perform the bulk update for this batch.
        with EmbeddingQueries() as db:
            if update_tuples:
                db.update_entities_emb(update_tuples)

        logger.info("Thread finished processing %d entities.", len(entities_to_embed))
        return len(entities_to_embed)

    def process_emb_batches(self) -> int:
        """
        Processes all entities using a thread pool.

        This method fetches all entities needing embeddings and distributes the work
        across multiple threads. It catches and counts any errors from the threads.

        Returns:
            int: The total count of batches that failed to process.
        """
        error_count = 0
        with EmbeddingQueries() as db:
            entity_count = db.count_entities_to_embed()

        if entity_count == 0:
            logger.info("No entities require embedding.")
            return 0  # Return 0 errors if there's nothing to do.

        logger.info("Found %d entities that need embedding.", entity_count)
        total_processed_count = 0

        # Create a thread pool to manage worker threads.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches to the executor and store the future objects.
            futures = [executor.submit(self.process_single_batch, batch) for batch in self.get_entities_in_batches()]
            batch_count = len(futures)

            # Process futures as they complete.
            for future in as_completed(futures):
                try:
                    # future.result() will re-raise any exception caught in the worker thread.
                    processed = future.result()
                    total_processed_count += processed
                except Exception as e:
                    # If an exception occurs, log it and increment the error counter.
                    logger.exception(f"A batch failed to process due to an error: {e}")
                    error_count += 1

                logger.info("Processed %d/%d entities so far.", total_processed_count, entity_count)

        logger.info("Completed embedding pass. Processed %d entities in %d batches with %d errors.",
                    total_processed_count, batch_count, error_count)
        return error_count

    def run_with_retries(self, max_retries: int = 2):
        """
        Orchestrates the embedding process with a robust retry loop.

        This method will attempt the full embedding process and, if errors occur,
        will retry up to `max_retries` times.

        Args:
            max_retries (int): The maximum number of times to retry after the initial run fails.
        """
        logger.info("Starting embedding workflow with a maximum of %d retries.", max_retries)

        attempt = 0
        # The loop runs for the initial attempt (0) plus the number of retries.
        while attempt <= max_retries:
            logger.info("--- Starting attempt #%d ---", attempt + 1)

            errors_found = self.process_emb_batches()

            # If a run completes with zero errors, the workflow is successful.
            if errors_found == 0:
                logger.info("Workflow successful. No errors found.")
                return  # Exit the function on success.

            logger.warning("Attempt #%d finished with %d errors.", attempt + 1, errors_found)
            attempt += 1

            # If we haven't exceeded the retry limit, wait before trying again.
            if attempt <= max_retries:
                logger.info("Waiting for 10 seconds before retrying...")
                time.sleep(10)

        # If the loop finishes without a successful run, log the final failure.
        logger.error(
            "Embedding workflow failed after %d retries. Please check the logs for persistent errors.",
            max_retries
        )


if __name__ == "__main__":
    # Create an instance of the embedder.
    kg = EntityEmbedder()
    # Call the robust workflow method to start the process.
    kg.run_with_retries(max_retries=100)