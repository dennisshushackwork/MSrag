"""
This pipeline is specifically designed for the data pipeline. It does the following:
1. Parses the document.
2. Creates the chunks.
3. Extracts the entities and relationships.
4. Builds the Knowledge Graph and Performs Entity Resolution.
5. Performs the Leiden Algorithm to find Communities.
"""
# External Imports:
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal, Tuple

# Internal Imports:
from parser.parser import DoclingParser
from chunkers.chunker import ChunkingService
from graph.construction import GraphConstruction
from postgres.populate import PopulateQueries

# Load the environmental variables & Configure the logger:
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Class for the entire data flow.
    """
    @staticmethod
    def parse_document(document_path: str) -> str:
        """This method parses the document."""
        logger.info(f"Processing {document_path}")
        parser = DoclingParser()
        doc = parser.parse(Path(document_path))
        logger.info("Document has been parsed!")
        return doc

    @staticmethod
    def insert_document_into_db(doc: str) -> Tuple[str, int]:
        """This method inserts the document into the database."""
        with PopulateQueries() as db:
            doc_id = db.set_document(doc)
        return doc, doc_id

    @staticmethod
    def create_chunks(doc: str, doc_id: int, chunk_type: Literal["token", "recursive", "sentence"],
                      chunk_size: int = 400, chunk_overlap: int = 200) -> None:
        """This method creates chunks from the parsed document"""
        chunker = ChunkingService(doc=doc, chunk_type=chunk_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                  doc_id=doc_id)
        chunker.create_chunk_from_document()

    @staticmethod
    def create_knowledge_graph(doc_id: int, model: str) -> None:
        kg_constructor = GraphConstruction()
        kg_constructor.process_document_chunks(document_id=doc_id, model=model)

    @staticmethod
    def entity_resolution(model):
        """Performs entity resolution"""
        # resolution = EntityResolution(model=model)
        # resolution.resolve_entities()
        pass

if __name__ == "__main__":
    pipeline = DataPipeline()
    doc = pipeline.parse_document("/Users/dennis/Desktop/obama.pdf")
    doc, doc_id = pipeline.insert_document_into_db(doc)
    pipeline.create_chunks(doc_id=doc_id, doc=doc, chunk_type="recursive")
    pipeline.create_knowledge_graph(doc_id, model="openai")

