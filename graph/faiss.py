"""
FAISS-based Entity Similarity Calculator for GraphRAG
This module provides efficient similarity calculation using FAISS with Product Quantization.
It follows the guidelines from facebook to find the optimal Index:
https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
"""

# External imports:
import os
import faiss
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

# Internal imports:
from postgres.resolution import ResolutionQueries

# Set the logger:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSSimilarityCalculator:
    """
    Calculates entity similarities using FAISS with Product Quantization
    """
    def __init__(self,
                 dimension: int = 256, # Vector dimensions
                 batch_size: int = 5_000_000, # Batch size (needs to be high for training)
                 similarity_threshold: float = 0.90, # High threshold for similarity scoring
                 target_recall: float = 0.90, # The target recall we are trying to achieve
                 ):

        self.dimension = dimension
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.target_recall = target_recall
        self.index = None
        self.entity_id_map = []  # Maps FAISS index position to entity_id
        self.entity_count = None
        self.nprobe = 1
        self.nlist = 1

    def get_entity_count(self):
        """Gets the entity count of the database"""
        with ResolutionQueries() as db:
            self.entity_count = db.count_entities()
        return self.entity_count

    def get_nprobe(self):
        """Determines the nprobe of the index (the amount of cells we visit)"""
        return int(np.sqrt(self.nlist))

    @staticmethod
    def load_entity_batch(offset: int, limit: int) -> Tuple[np.ndarray, List[int]]:
        """Loads the entities in batches from the database"""
        with ResolutionQueries() as db:
            embeddings, entity_ids = db.load_entity_batch(offset=offset, limit=limit)
        return embeddings, entity_ids

    def choose_index_type(self, total_vectors: int) -> faiss.Index:
        """
        Creates the correct index based on the dataset size:
        https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        """
        logger.info(f"Choosing index type for {total_vectors:,} vectors")

        if total_vectors < 2_000_000:
            self.nlist = int(8 * np.sqrt(total_vectors))
            M = 16  # Number of subquantizers
            nbits = 11  # Bits per subquantizer based on the literature 2^11 = 2048

            # OPQ transformation
            opq_matrix = faiss.OPQMatrix(self.dimension, M, self.dimension)

            # Create IVF index with PQ
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, M, nbits)
            index = faiss.IndexPreTransform(opq_matrix, index)

            # NLIST AND NPROBE:
            self.nprobe = self.get_nprobe()

            logger.info(f"Created IVF{self.nlist}_OPQ{M}_PQ{M}x{nbits} index (k={M * (2 ** nbits)})")
            logger.info(f"Memory per vector: {M * nbits / 8} bytes")
            logger.info(f"Default nprobe: {self.nprobe} (searching {self.nprobe /self.nlist * 100:.1f}% of cells)")

        elif total_vectors < 10_000_000:
            # For 2M-10M vectors: IVF65536 with PQ (min required 1966080 vectors for training)
            self.nlist = 65536
            self.nprobe = self.get_nprobe()
            hnsw_m = 32 # Given by facebook
            M = 16
            nbits = 11

            quantizer = faiss.IndexHNSWFlat(self.dimension, hnsw_m)
            # https://github.com/neondatabase-labs/pg_embedding/issues/47:
            quantizer.hnsw.efConstruction = 64 # efConstruction = 2 x m
            quantizer.hnsw.efSearch =
            logger.info(f"Using HNSW{hnsw_m} quantizer for fast centroid assignment")
            # OPQ transformation
            opq_matrix = faiss.OPQMatrix(self.dimension, M, self.dimension)

            # Create IVF index with PQ
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, M, nbits)
            index = faiss.IndexPreTransform(opq_matrix, index)

            self.nprobe = self._calculate_nprobe(self.nlist, target_recall=self.target_recall)

            logger.info(f"Created IVF{self.nlist}_HNSW{hnsw_m}_OPQ{M}_PQ{M}x{nbits} index (k={M * (2 ** nbits)})")
            logger.info(f"Default nprobe: {self.nprobe} (searching {self.nprobe/self.nlist*100:.1f}% of cells)")
