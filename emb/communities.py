"""
Embedding script for communities.
"""
# External imports
import logging
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Internal Imports:
from emb.embedder_old import Embedder
from postgres.embedding import EmbeddingQueries

# Initialisation of logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CommunityGroupEmbedder:
    """
    This class embeds community summaries from the database concurrently using multiple threads.
    """
    def __init__(self):
        # Maximum allowed tokens per embedding API call, preventing exceeding model context limits.
        self.max_tokens_per_batch = 7000
        # Number of community summaries to fetch and process in a single database query.
        self.batch_size = 100
        # The maximum number of worker threads to use for concurrent embedding tasks.
        self.max_workers = 4

    def batch_embed_texts(self, embedder: Embedder, texts_and_token_count: List[tuple]) -> List:
        """
        Batches texts (community summaries) such that each embedding call does not exceed
        `self.max_tokens_per_batch`.
        `texts_and_token_count` is a list of tuples: (text, token_count).
        """
        batched_results = []
        current_batch = []
        current_token_count = 0

        for text, token_count in texts_and_token_count:
            # If adding the next text would exceed the token limit, embed the current batch.
            if current_batch and (current_token_count + token_count > self.max_tokens_per_batch):
                logger.info("Embedding a batch with %d texts (%d tokens)", len(current_batch), current_token_count)
                # Calls the embedder_mlr_test to get embeddings for the current batch of texts.
                batch_result = embedder.embed_texts(current_batch)
                batched_results.extend(batch_result)
                # Resets for the next batch.
                current_batch = []
                current_token_count = 0
            # Adds the current text and its token count to the batch.
            current_batch.append(text)
            current_token_count += token_count

        # Embeds any remaining texts in the last batch.
        if current_batch:
            logger.info("Embedding final batch with %d texts (%d tokens)", len(current_batch), current_token_count)
            batch_result = embedder.embed_texts(current_batch)
            batched_results.extend(batch_result)
        return batched_results

    def get_community_groups_in_batches(self):
        """
        Yields batches of community groups (with their summaries) from the database
        that need embedding or token counting.
        """
        offset = 0
        while True:
            with EmbeddingQueries() as db:
                # Fetches a batch of community groups where `community_embed` is FALSE or `community_tokens` is NULL.
                batch = db.get_community_group_batches(self.batch_size, offset)
            if not batch:
                # Breaks the loop if no more community groups are found.
                break
            yield batch
            offset += self.batch_size

    def process_single_batch(self, batch: List[Tuple[int, str, int]]) -> int:
        """
        Processes a single batch of community group summaries:
        - Instantiates its own `Embedder` for thread safety.
        - Counts tokens if `community_tokens` is NULL.
        - Embeds the summaries.
        - Updates the database with the generated embeddings and sets `community_embed` to FALSE.
        Returns the number of processed community groups in this batch.
        """
        embedder = Embedder()
        summaries_to_embed = []
        updates_for_tokens = []

        for community_id, summary_text, current_tokens in batch:
            if current_tokens is None:
                # If tokens are not yet counted, count them first.
                token_count = embedder.count_tokens(summary_text)
                updates_for_tokens.append((token_count, community_id))
                summaries_to_embed.append((community_id, summary_text, token_count))
            else:
                summaries_to_embed.append((community_id, summary_text, current_tokens))

        # Update token counts in DB first if any were calculated
        if updates_for_tokens:
            with EmbeddingQueries() as db:
                db.update_community_group_tokens(updates_for_tokens)
            logger.info(f"Updated token counts for {len(updates_for_tokens)} community groups.")

        # Prepare for embedding (filter out items with NULL summary or summary too long if needed)
        valid_summaries_for_embedding = []
        community_ids_for_embedding_result_mapping = [] # To map embeddings back to correct community IDs
        for comm_id, summary_text, token_count in summaries_to_embed:
            if summary_text is None or token_count is None:
                logger.warning(f"Community {comm_id} has NULL summary or token count. Skipping embedding.")
                continue
            # Optionally add a check for token_count > self.max_tokens_per_batch here if you want to skip embedding
            # for overly long summaries rather than batching.
            valid_summaries_for_embedding.append((summary_text, token_count))
            community_ids_for_embedding_result_mapping.append(comm_id)

        if not valid_summaries_for_embedding:
            logger.info("No valid community summaries to embed in this batch.")
            return 0

        logger.info("Thread starting processing %d community group summaries for embedding.", len(valid_summaries_for_embedding))

        # Generates embeddings for the batch of community summaries.
        emb_results = self.batch_embed_texts(embedder, valid_summaries_for_embedding)

        # Combines community IDs with their generated embeddings for database update.
        # Ensure that emb_results order matches community_ids_for_embedding_result_mapping order
        update_embeddings_tuples = list(zip(community_ids_for_embedding_result_mapping, emb_results))

        with EmbeddingQueries() as db:
            # Updates the database with the new embeddings.
            db.update_emb_community_group(update_embeddings_tuples)
        logger.info("Thread finished processing %d community group summaries.", len(batch))
        return len(batch)

    def process_community_group_emb_batches(self) -> None:
        """
        Processes all community group summaries that require embedding or token counting using multiple threads.
        """
        with EmbeddingQueries() as db:
            # Counts the total number of community groups awaiting embedding or token counting.
            community_group_count = db.count_community_groups_to_embed()

        if community_group_count == 0:
            logger.info("No community groups require embedding.")
            return
        logger.info("Found %d community groups that need embedding or token counting.", community_group_count)

        total_processed_count = 0
        batch_count = 0

        # Uses ThreadPoolExecutor to parallelize the embedding process across multiple threads.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            # Iterates through batches of community groups from the database.
            for batch in self.get_community_groups_in_batches():
                batch_count += 1
                # Submits each batch to a worker thread for processing.
                future = executor.submit(self.process_single_batch, batch)
                futures.append(future)

            # Waits for each thread to complete and aggregates the results.
            for future in as_completed(futures):
                try:
                    processed = future.result()
                    total_processed_count += processed
                    logger.info("Processed %d/%d community groups so far.", total_processed_count, community_group_count)
                except Exception as exc:
                    logger.error(f"Batch processing generated an exception: {exc}")

        logger.info("Completed embedding process. Processed %d community groups in %d batches.", total_processed_count, batch_count)


if __name__ == "__main__":
    logger.info("--- Starting Community Group Embedding Script ---")
    community_group_embedder = CommunityGroupEmbedder()
    community_group_embedder.process_community_group_emb_batches()
    logger.info("--- Community Group Embedding Script Finished ---")