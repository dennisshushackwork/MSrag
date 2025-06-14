"""
Relationship Embedder: This class is optimised to embed all
Relationships inside the Postgres Database using multiple threads for improved performance.
"""
# External Imports:
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Internal Imports:
from emb.embedder_old import Embedder
from postgres.embedding import EmbeddingQueries

# Setting the logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RelEmbedder:
    """This class embeds the entities inside the database concurrently using multiple threads."""
    def __init__(self):
        self.max_tokens_per_batch = 8000  # Maximum allowed tokens per embedding call
        self.batch_size = 100  # Batches of entities (100 max)
        self.max_workers = 4  # Number of concurrent threads

    def batch_embed_texts(self, embedder: Embedder, texts_and_token_count: List[tuple]) -> List:
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

    def get_rels_in_batches(self):
        """
        Yields batches of entities from the database.
        """
        offset = 0
        while True:
            with EmbeddingQueries() as db:
                batch = db.get_rels_batches(self.batch_size, offset)
            if not batch:
                break
            yield batch
            offset += self.batch_size

    def process_single_batch(self, batch: List[tuple]) -> int:
        """
        Process a single batch of entities:
          - Creates its own EmbeddingService instance.
          - Embeds the batch.
          - Updates the database.
        Returns the number of processed entities.
        """
        # Each thread instantiates its own embedder_mlr_test
        embedder = Embedder()

        rels_to_embed = [r for r in batch]
        update_tuples = []  # Each tuple: (rel_id, new_emb, token_count)

        logger.info("Thread starting processing %d relationships that need embedding.",
                    len(rels_to_embed))

        # Process entities that need embedding
        if rels_to_embed:
            # Get the entity names:
            descriptions = [r[1] for r in rels_to_embed]
            description_tokens = [(description, embedder.count_tokens(description)) for description in descriptions]

            # Batch the embedding calls for efficiency
            emb_results = self.batch_embed_texts(embedder, list(zip(descriptions, [t[1] for t in description_tokens])))

            # Build update tuples
            for idx, rel in enumerate(rels_to_embed):
                update_tuples.append((rel[0], emb_results[idx], description_tokens[idx][1]))

        # Update the database in bulk
        with EmbeddingQueries() as db:
            if update_tuples:
                db.update_rels_emb(update_tuples)

        logger.info("Thread finished processing %d relationships.", len(rels_to_embed))
        return len(rels_to_embed)

    def process_emb_batches(self) -> None:
        """
        Processes all entities marked for embedding using multiple threads.
        """
        # Get the number of entities to embed:
        with EmbeddingQueries() as db:
            rel_count = db.count_rels_to_embed()

        if rel_count == 0:
            logger.info("No relationships require embedding.")
            return

        logger.info("Found %d relationships that need embedding.", rel_count)
        total_processed_count = 0
        batch_count = 0

        # Use ThreadPoolExecutor to run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for batch in self.get_rels_in_batches():
                batch_count += 1
                future = executor.submit(self.process_single_batch, batch)
                futures.append(future)

            for future in as_completed(futures):
                processed = future.result()
                total_processed_count += processed
                logger.info("Processed %d/%d relationships so far.", total_processed_count, rel_count)

        logger.info("Completed embedding process. Processed %d relationships in %d batches.",
                   total_processed_count, batch_count)

        logger.info("Calculating the weights of the relationships.")
        with EmbeddingQueries() as db:
            db.calculate_rel_weights()


if __name__ == "__main__":
    relembedder = RelEmbedder()
    relembedder.process_emb_batches()