"""
Entity Resolution System for GraphRAG
This module handles the complete entity resolution workflow including:
1. Similarity calculation between entities using FAISS for fast performance
    leveraging product quantization
2. Graph-based grouping using connected components.
3. Levenshtein distance filtering
4. LLM-based merge decisions
5. Database updates with proper constraint handling
"""

# External imports:
import os
import faiss
import logging
import numpy as np
import networkx as nx
from typing import List, Set, Tuple
from tqdm import tqdm
from itertools import combinations
from dotenv import load_dotenv
from Levenshtein import distance as levenshtein_distance

# Internal imports:
from postgres.resolution import ResolutionQueries

# Load environmental variables & logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSSimilarityCalculator:
    """
    Calculates entity similarities using FAISS with Product Quantization
    """
    def __init__(self,
                 dimension: int = 256,
                 batch_size: int = 5_000_000,
                 similarity_threshold: float = float(os.environ.get('SIMILARITY_THRESHOLD'), 0.85),
                 ):
        self.dimension = dimension
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.index = None
        self.entity_id_map = []

    def get_entity_count(self):










class EntityResolution:
    """This class performs the entity resolution workflow"""
    def __init__(self, model: str = "openai", batch_size: int = 10_000):
        self.model = model # LLM-Model to perform the resolution
        self.batch_size = batch_size
        self.networkx_batch_size = 100_000
        self.min_similarity = float(os.getenv("MIN_SIMILARITY", 0.90))
        self.levenshtein_threshold = float(os.getenv("LEVENSHTEIN_THRESHOLD", 0.70))
        self.max_group_size = int(os.getenv("MAX_GROUP_SIZE", 100))
        self.total_entities = None

    def calculate_min_similarity(self) -> None:
        """Calculates the similarity between the entities and stores those who have > 0.85"""
        logger.info("Starting entity similarity calculation using DiskANN index in batches...")

        # Clear existing similarities:
        with ResolutionQueries() as db:
            db.clear_entity_similarities()

            # Get total entity count
            self.total_entities = db.count_entities()
            logger.info(f"Total entities to process: {self.total_entities}")

            # Process in batches
            for offset in tqdm(range(0, self.total_entities, self.batch_size), desc="Processing batches"):
                # Get batch of entity IDs
                entity_ids = db.get_entity_id_batch(offset, self.batch_size)

                if not entity_ids:
                    continue

                # Calculate similarities using database query
                similarities = db.calculate_similarities_for_batch(entity_ids, min_similarity=self.min_similarity)

                # Insert similarities (both directions)
                if similarities:
                    db.insert_entity_similarities_batch(similarities)
                logger.info(f"Processed {min(offset + self.batch_size, self.total_entities)}/{self.total_entities} entities")

    def find_entity_groups(self) -> List[Set[int]]:
        """
        Use graph-based approach to find connected components of similar entities
        """
        logger.info("Finding entity groups using connected components...")

        # Builds the networkx Graph in batches
        G = nx.Graph()

        entity_offset = 0
        while True:
            # Get all similar entity pairs
            with ResolutionQueries() as db:
                similar_pairs = db.load_entities_in_batches(batch_size=self.networkx_batch_size, offset=entity_offset)
                if not similar_pairs:
                    logger.info("All entities have been loaded from the database.")
                    break

                for pair in similar_pairs:
                    G.add_edge(pair['from_entity'], pair['to_entity'],
                               weight=pair['similarity'],
                               from_name=pair['from_name'],
                               to_name=pair['to_name'])

                entity_offset += self.batch_size
                logger.info(f"Entities loaded: {entity_offset} of {self.total_entities}")

        # Find connected components
        components = list(nx.connected_components(G))
        logger.info(f"Found {len(components)} connected components")

        # Filter by size
        valid_components = [comp for comp in components if 2 <= len(comp) <= self.max_group_size]
        logger.info(f"Valid components (2-{self.max_group_size} entities): {len(valid_components)}")
        for component in valid_components:
            print(component)

if __name__ == "__main__":
    entity_res = EntityResolution()
    entity_res.calculate_min_similarity()
    #entity_res.find_entity_groups()















