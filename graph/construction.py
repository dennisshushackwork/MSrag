"""
This class is designed to construct the Knowledge Graph (KG) from
the input text chunks. Its main goal is to:
1. Extract all entities and relationships.
2. Save those entities and relationships inside the Postgres database.
3. Embedd all the entities and relationships inside the Postgres database.
4. Provide a fallback mechanism for chunk processing.
"""

# External Imports:
import time
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Internal Imports:
from postgres.populate import PopulateQueries
from llm.prompts.construction.extract import Extractor
from emb.entities import EntityEmbedder
from emb.relationships import RelEmbedder

# Configure Logging & environmental variables:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphConstruction:
    """Builds the Graph from the chunks (extracts relationships and derived entities). Saves them inside the postgres database"""
    def __init__(self, batch_size: int = 10, max_workers: int = 5):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = 3 # Defines the number of retries allowed (failed chunk)
        self.retry_delay = 5 # Defines the retry delay
        self.failed_chunks = set()  # Store failed chunk IDs

    @staticmethod
    def extract_entities_from_relationships(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts unique entities from relationship source and target fields.
        Returns a list of entity dictionaries.
        """
        entity_dict = {}

        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            if source and source not in entity_dict:
                entity_dict[source] = {"name": source}
            if target and target not in entity_dict:
                entity_dict[target] = {"name": target}
        return list(entity_dict.values())


    @staticmethod
    def upsert_entities(entities: List[Dict[str, Any]], document_id: Optional[int] = None,
                        chunk_id: Optional[int] = None) -> List[tuple]:
        """
        Bulk upsert entities and link to document and chunk via join tables.
        Returns list of (entity_name, entity_id).
        """
        if not entities:
            logger.info("No entities to upsert")
            return []

        # Deduplicate names
        unique_names = {e['name'] for e in entities}
        names = list(unique_names)

        # 1) bulk upsert names
        with PopulateQueries() as db:
            name_id_pairs = db.upsert_entities_bulk(names)

            # 2) bulk link to document and chunk
            if document_id:
                doc_links = [(eid, document_id) for _, eid in name_id_pairs]
                db.insert_entity_documents_bulk(doc_links)
            if chunk_id:
                chunk_links = [(eid, chunk_id) for _, eid in name_id_pairs]
                db.insert_entity_chunks_bulk(chunk_links)
        logger.info(f"Upserted and linked {len(name_id_pairs)} entities for chunk {chunk_id}")
        return name_id_pairs

    @staticmethod
    def upsert_relationships(entities: List[tuple], relationships: List[Dict[str, Any]], chunk_id: int):
        """Upserts the relationships into the database. Entities = List(name, id)"""
        # Values to insert into the database:
        values = []

        # Setting the source and target to the respective entity_id
        for rel in relationships:
            source_id = None
            target_id = None

            # Find entity IDs for source and target
            for entity in entities:
                if rel["source"] == entity[0]:
                    source_id = entity[1]
                if rel["target"] == entity[0]:
                    target_id = entity[1]

            # Only add relationship if both source and target IDs were found
            if source_id is not None and target_id is not None:
                values.append((
                    source_id,
                    target_id,
                    rel["description"],
                    rel["summary"],
                    chunk_id,
                    True
                ))

        # Inserting the relationships into the database:
        with PopulateQueries() as db:
            db.upsert_relationships_bulk(values)
        logger.info(f"Inserted relationships of chunk with id {chunk_id}")

    def process_chunk(self, chunk: tuple, model: str) -> bool:
        """
        Extracts relationships from a chunk using the LLM,
        derives entities from those relationships,
        and inserts them into the database.
        Returns True if successful, False if failed.
        # Chunk is of the form:
        # chunk = (chunk_id, chunk_document_id, chunk_text, chunk_tokens, chunk_type)
        """
        chunk_id = chunk[0]
        try:
            logger.info(f"Processing chunk with chunk_id: {chunk_id}")
            extractor = Extractor(chunk[2], model=model)
            extraction = extractor.call_and_extract()

            # Get relationships from extraction
            chunk_relationships = extraction.get("relationships", [])

            if not chunk_relationships:
                logger.info(f"No relationships found for chunk_id: {chunk_id}")
                return True  # This is not a failure, just no relationships found

            # Extract entities from relationships
            chunk_entities = self.extract_entities_from_relationships(chunk_relationships)

            # Upsert entities
            entities = self.upsert_entities(chunk_entities, document_id=chunk[1], chunk_id=chunk_id)

            # Upsert the relationships
            self.upsert_relationships(entities, chunk_relationships, chunk_id=chunk_id)
            return True
        except Exception as e:
            logger.exception(f"Error processing chunk {chunk_id}: {e}")
            self.failed_chunks.add(chunk_id)
            return False

    def process_document_chunks(self, document_id: int, model: str) -> None:
        """
        Processes all chunks related to the document_id in batches.
        After processing all chunks, it attempts to retry any failed chunks.
        """
        offset = 0
        total_processed = 0
        self.failed_chunks.clear()  # Reset failed chunks for this document

        # First pass: process all chunks
        while True:
            # Get next batch of chunks
            with PopulateQueries() as db:
                chunks = db.load_chunks_in_batches(document_id, self.batch_size, offset)

            # If no more chunks, we're done
            if not chunks:
                break

            # Process this batch
            logger.info(f"Processing batch of {len(chunks)} chunks for document {document_id} (offset: {offset})")
            self.process_chunks_concurrently(chunks, model)

            # Update counts
            total_processed += len(chunks)
            offset += self.batch_size

            # Log progress
            logger.info(f"Processed {total_processed} chunks for document {document_id}")

        # Second pass: retry failed chunks
        if self.failed_chunks:
            logger.info(f"Retrying {len(self.failed_chunks)} failed chunks for document {document_id}")
            self.retry_failed_chunks(document_id, model)

        logger.info(f"Completed processing all chunks for document {document_id}")

        # After processing all batches for the document, embed the entities
        logger.info(f"Embedding entities for document {document_id}")
        kg_embedder = EntityEmbedder()
        kg_embedder.process_emb_batches()
        logger.info(f"Embedding Relationships for document {document_id}")
        rel_embedder = RelEmbedder()
        rel_embedder.process_emb_batches()
        logger.info(f"Done the embeddings for relationships and entities!")


    def process_chunks_concurrently(self, chunks: List[tuple], model: str) -> None:
        """
        Processes multiple chunks concurrently:
        1. Performs LLM calls to extract relationships
        2. Extracts entities from relationships
        3. Inserts/Updates the entities and relationships in the database
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk, model) for chunk in chunks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Error in processing thread: {e}")



    def retry_failed_chunks(self, document_id: int, model: str) -> None:
        """
        Retries processing of failed chunks with exponential backoff.
        """
        retry_count = 0
        while self.failed_chunks and retry_count < self.max_retries:
            retry_count += 1
            logger.info(f"Retry attempt {retry_count} for {len(self.failed_chunks)} failed chunks")

            # Get all failed chunks for this retry attempt
            failed_chunk_ids = list(self.failed_chunks)
            self.failed_chunks.clear()  # Reset for this retry

            # Fetch the actual chunk data for the failed chunk IDs
            chunks_to_retry = []
            for chunk_id in failed_chunk_ids:
                with PopulateQueries() as db:
                    chunk = db.fetch_chunk_by_id(chunk_id)
                    if chunk:
                        chunks_to_retry.append(chunk)

            if not chunks_to_retry:
                logger.warning("Could not fetch any chunks to retry")
                break

            # Process these chunks with increased delay between retries
            for chunk in chunks_to_retry:
                # Exponential backoff
                delay = self.retry_delay * (2 ** (retry_count - 1))
                time.sleep(delay)
                success = self.process_chunk(chunk, model)
                if not success:
                    logger.warning(f"Chunk {chunk[0]} failed again on retry {retry_count}")

            logger.info(f"After retry {retry_count}, {len(self.failed_chunks)} chunks still failed")

        # Log any chunks that still failed after all retries
        if self.failed_chunks:
            logger.error(
                f"After {self.max_retries} retries, {len(self.failed_chunks)} chunks still failed: {self.failed_chunks}")
