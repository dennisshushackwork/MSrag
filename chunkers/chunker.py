"""
This is a Service Class for Chunking.
It chunks the data and saves it in the Postgres Database.
"""
# External Imports:
import logging
from typing import Literal, List

# Internal Imports:
from chunkers.token_chunker import FixedTokenChunker
from chunkers.recursive_chunker import PositionTrackingRecursiveChunker
from chunkers.cluster_simple import ClusterSemanticChunker
from emb.chunks import ChunkEmbedder
from postgres.populate import PopulateQueries

# Load the logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkingService:
    def __init__(self, doc: str, chunk_type: Literal["token", "recursive", "cluster"], chunk_size: int = 400,
                 chunk_overlap: int = 100, doc_id: int = 1):
        self.doc = doc
        self.doc_id = doc_id
        self.chunk_type = chunk_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunk_from_document(self):
        """Creates the chunks from the document and saves them in the database."""
        logger.info("Creating the chunks...")
        if self.chunk_type == "token":
            # Splits the document based on token-splitting.
            chunker = FixedTokenChunker(self.doc_id, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            chunks = chunker.split_text(self.doc)
        elif self.chunk_type == "recursive":
            chunker = PositionTrackingRecursiveChunker(self.doc_id, self.chunk_size, self.chunk_overlap)
            chunks = chunker.split_text(self.doc)
        elif self.chunk_type == "cluster":
            chunker = ClusterSemanticChunker(doc_id=self.doc_id, max_chunk_size=self.chunk_size)
            chunks = chunker.split_text(self.doc)
        else:
            chunks = []


        # Inserts the chunk into the database:
        self.insert_chunks_into_db(chunks)
        # Creates the embedding for the chunks:
        embedder = ChunkEmbedder()
        embedder.process_chunk_emb_batches()

    @staticmethod
    def insert_chunks_into_db(chunks: List[tuple]) -> None:
        """Insert the chunks into the database"""
        with PopulateQueries() as db:
            db.set_chunks_with_positions(chunks)
        logger.info("Chunks have been added to the database!")



