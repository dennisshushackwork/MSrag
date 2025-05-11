"""
This is a Service Class for Chunking.
It chunks the data and saves it in the Postgres Database.
"""

# External Imports:
import logging
from typing import Literal, List

# Internal Imports:
from chunkers.recursivechunker import RecursiveChunker
from chunkers.sentencechunker import SentenceTokenChunker
from chunkers.tokenchunker import TokenChunker
from emb.chunks import ChunkEmbedder
from postgres.populate import PopulateQueries

# Load the logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChunkingService:
    def __init__(self, doc: str, chunk_type: Literal["token", "recursive"], chunk_size: int = 400,
                 chunk_overlap: int = 100, doc_id: int = 1):
        self.doc = doc
        self.doc_id = doc_id
        self.chunk_type = chunk_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunk_from_document(self) -> None:
        """Creates the chunks from the document and saves them in the database."""
        logger.info("Creating the chunks...")
        if self.chunk_type == "token":
            chunker = TokenChunker(self.doc_id, self.chunk_size, self.chunk_overlap)
            chunks = chunker.process_document(document=self.doc)
        elif self.chunk_type == "sentence":
            chunker = SentenceTokenChunker(self.doc_id, self.chunk_size, self.chunk_overlap)
            chunks = chunker.process_document(document=self.doc)
        elif self.chunk_type == "recursive":
            chunker = RecursiveChunker(self.doc_id, self.chunk_size, self.chunk_overlap)
            chunks = chunker.process_document(document=self.doc)
        else:
            chunks = []
        logger.info("Inserting the chunks into the database!")

        # Inserts the chunk into the database:
        self.insert_chunks_into_db(chunks)
        # Creates the embedding for the chunks:
        embedder = ChunkEmbedder()
        embedder.process_chunk_emb_batches()

    @staticmethod
    def insert_chunks_into_db(chunks: List[tuple]) -> None:
        """Insert the chunks into the database"""
        with PopulateQueries() as db:
            db.set_chunks(chunks)
        logger.info("Chunks have been added to the database!")



