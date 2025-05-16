"""
This class is specifically designed for Entity Resolution (merging similar entities togheter).
1.

"""
# External imports:
import ast
import faiss
import numpy as np
import rustworkx as rx


# Internal imports:
from postgres.resolution import ResolutionQueries


class IndexType:
    """
    Defines the index type used, based on the dataset size.
    https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    """
    FLAT = "Flat" # < 10K entities, exact search
    IVF = "IVF" # < 1M entities, basic clustering
    IVF_HNSW = "IVF_HNSW" # 1M-10M entities
    IVF_HNSW_LARGE = "IVF_HNSW_LARGE"  # 10M-100M entities
    IVF_HNSW_XLARGE = "IVF_HNSW_XLARGE"  # 100M-1B entities


class PQEntityIndexer:
    """
    FAISS IVF-PQ indexer for entity embeddings.
    Automatically selects an optimal index configuration based on dataset size,
    primarily using L2 distance and Product Quantization for memory efficiency.

    Key features:
      - Tiered index strategy (Flat, IVF with k-means, IVF with HNSW).
      - Product Quantization (PQ) for vector compression in IVF indexes.
      - Robust training data collection.
      - Customizable parameters (dimension, PQ settings, thresholds).
    """
    def __init__(self,
                 dim: int = 768, # Dimensions of the embedding vectors
                 d_threshold: float = 0.7, # Threshold for connected components (how similar they must be)
                 m: int = 16, # Number of sub-quantizers

                 ):
