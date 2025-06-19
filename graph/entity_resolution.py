"""
Entity Resolution System for GraphRAG
This module handles the complete entity resolution workflow including:
1. Similarity calculation between entities using FAISS for fast performance
    leveraging product quantization.
2. Graph-based grouping using connected components.
3. Levenshtein distance filtering
4. LLM-based merge decisions
5. Database updates with proper constraint handling
"""

# External imports:
import os
import re
import time
import faiss
import logging
import numpy as np
from dataclasses import dataclass
import networkx as nx
from typing import List, Tuple, Dict
from rapidfuzz import fuzz
from tqdm import tqdm
from itertools import combinations
from dotenv import load_dotenv

# Internal imports:
from postgres.resolution_new import ResolutionQueries
from llm.llm import LLMClient

# Load environmental variables & logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EntityGroup:
    """Represents a group of similar entities"""
    entity_ids: List[int]
    representative_id: int
    confidence: float

@dataclass
class EntityPair:
    """Represents a pair of entities with their similarity score"""
    entity1_id: int
    entity2_id: int
    similarity: float

@dataclass
class EntityResolutionResult:
    """Represents the result of LLM-based entity resolution"""
    should_merge: bool
    resolved_name: str
    confidence: str
    reasoning: str

class FAISSSimilarityCalculator:
    """
    Calculates entity similarities using FAISS with Product Quantization
    Simplified approach for better GPU compatibility with pre-normalized vectors
    """
    def __init__(self,
                 dimension: int = 256, # Vector dimensions of the input vectors
                 batch_size_train: int = 5_000_00,  # Max Batch size possible
                 batch_size_add: int = 100_000, # Batch size for adding the vectors
                 search_batch_size: int = 10000,  # Batch size for similarity search
                 similarity_threshold: float = 0.90,  # Adjusted for cosine similarity
                 top_k: int = 30,  # Top-K extracted from similarity search
                 ):
        self.dimension = dimension
        self.batch_size_train = batch_size_train # Batch size for training the index
        self.batch_size_add = batch_size_add
        self.similarity_threshold = similarity_threshold
        self.search_batch_size = search_batch_size
        self.top_k = top_k
        self.total_vectors = self.get_total_vectors()
        self.M = None
        self.nlist = None
        self.nprobe = None
        self.nbits = None
        self.index = None
        self.entity_id_mapping = {}

    def get_total_vectors(self) -> int:
        """This returns the total number of vectors in the dataset"""
        with ResolutionQueries() as db:
            total_vectors = db.count_entities()
        return total_vectors

    def get_nprobe(self, nlist: int) -> int:
        """Returns the nprobe value based on the nlist value"""
        return int(np.sqrt(nlist))

    def get_index(self):
        """Gets the index based on dataset count"""
        if self.total_vectors <= 50000:
            self.nlist = 128 # Choosing a safe, smaller value
            self.M = 32  # PQ subquantizers
            self.nbits = 8

        elif self.total_vectors < 1_000_000:
            # For <1M: 4*sqrt(N) to 16*sqrt(N)
            self.nlist = int(8 * np.sqrt(self.total_vectors))
            self.M = 32  # Number of subquantizers
            self.nbits = 8   # 8 bits = 256 centroids per subquantizer

        elif self.total_vectors < 10_000_000:
            # For 1M-10M: Use larger nlist for better partitioning
            self.nlist = int(16 * np.sqrt(self.total_vectors))
            self.nlist = min(self.nlist, 65536)  # Cap at 65536 for memory efficiency
            self.M = 32  # More subquantizers for better precision
            self.nbits = 8

        elif self.total_vectors < 100_000_000:
            # For 10M-100M: Even larger nlist
            self.nlist = min(int(20 * np.sqrt(self.total_vectors)), 262144)  # Cap at 262144
            self.M = 64  # Maximum subquantizers for best precision
            self.nbits = 8

        else:
            # For 100M+: Very large nlist
            self.nlist = min(int(32 * np.sqrt(self.total_vectors)), 1048576)  # Cap at 1M
            self.M = 64
            self.nbits = 8

        # Calculate nprobe
        self.nprobe = self.get_nprobe(self.nlist)

        # Use index_factory like the working script for simplicity
        index_string = f"IVF{self.nlist},PQ{self.M}"
        logger.info(f"Creating index with factory string: {index_string}")

        # For normalized vectors, use INNER_PRODUCT for cosine similarity
        index = faiss.index_factory(self.dimension, index_string, faiss.METRIC_INNER_PRODUCT)

        # Calculate memory usage
        memory_per_vector = self.M * self.nbits / 8
        total_memory_mb = (self.total_vectors * memory_per_vector) / (1024 * 1024)
        logger.info(f"Index configuration:")
        logger.info(f"  - Type: {index_string}")
        logger.info(f"  - Metric: INNER_PRODUCT (for cosine similarity)")
        logger.info(f"  - Number of cells: {self.nlist} MB")
        logger.info(f"  - Centroids per subquantizer: {2**self.nbits}")
        logger.info(f"  - Memory per vector: {memory_per_vector:.1f} bytes")
        logger.info(f"  - Estimated total memory: {total_memory_mb:.1f} MB")
        logger.info(f"  - nprobe: {self.nprobe} ({self.nprobe/self.nlist*100:.1f}% of cells)")

        return index

    @staticmethod
    def load_entity_batch(offset: int, limit: int) -> Tuple[np.ndarray, List[int]]:
        """Loads the entities in batches from the database"""
        with ResolutionQueries() as db:
            embeddings, entity_ids = db.load_entities_in_batches(offset=offset, limit=limit)
        return embeddings, entity_ids

    def _add_all_vectors_to_index(self):
        """Add all entity vectors to the trained index in batches"""
        offset = 0
        faiss_index = 0

        with tqdm(total=self.total_vectors, desc="Adding vectors to index") as pbar:
            while offset < self.total_vectors:
                # Load batch
                vectors, entity_ids = self.load_entity_batch(offset, self.batch_size_add)
                if len(vectors) == 0:
                    break
                # Add to index
                self.index.add(vectors.astype(np.float32))
                # Update entity ID mapping
                for i, entity_id in enumerate(entity_ids):
                    self.entity_id_mapping[faiss_index + i] = entity_id
                faiss_index += len(vectors)
                offset += self.batch_size_add
                pbar.update(len(vectors))

    def train_index(self):
        """
        Simplified GPU training approach similar to the working script
        """

        # Get total entity count and create index

        self.index = self.get_index()

        # Calculate training vectors needed
        min_training_vectors = 30 * self.nlist
        max_training_vectors = min(256 * self.nlist, self.total_vectors)
        training_vectors_needed = min(max_training_vectors, max(min_training_vectors, self.batch_size_train))

        logger.info(f"Loading {training_vectors_needed:,} training vectors from {self.total_vectors:,} total entities")

        # Load training data
        training_vectors, _ = self.load_entity_batch(offset=0, limit=training_vectors_needed)
        logger.info(f"Loaded {len(training_vectors):,} training vectors of dimension {training_vectors.shape[1]}")

        # Simple GPU training like the working script
        logger.info("Moving index to GPU for training")
        # Single GPU usage:
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        logger.info("Training index on GPU...")
        start_time = time.time()
        self.index.train(training_vectors.astype(np.float32))
        training_time = time.time() - start_time
        logger.info(f"GPU training completed in {training_time:.2f} seconds")

        # Adding all vectors to the index in batches
        logger.info("Adding all vectors to the index...")
        self._add_all_vectors_to_index()

    def _save_similarity_pairs_to_db(self, pairs: List[EntityPair]):
        """Save similarity pairs to database"""
        if not pairs:
            return

        logger.info(f"Saving {len(pairs)} similarity pairs to database")
        with ResolutionQueries() as db:
            db.save_similarity_pairs(pairs)

    def calculate_similarity(self):
        """
        This method calculates the cosine similarity among all vectors inside of the database (entities).
        Keeps those values, which are above similarity_threshold. Does this in batches.
        """
        if self.index is None:
            raise ValueError("Index is not trained yet")

        logger.info(f"Starting similarity calculation with threshold {self.similarity_threshold}")
        similar_pairs = []
        batch_count = 0
        # Process entities in batches
        offset = 0

        with tqdm(total=self.total_vectors, desc="Calculating similarities") as pbar:
            while offset < self.total_vectors:
                # Load query batch
                current_batch_size = min(self.search_batch_size, self.total_vectors - offset)
                query_vectors, query_entity_ids = self.load_entity_batch(offset, current_batch_size)

                if len(query_vectors) == 0:
                    break

                # Search for top-k similar vectors
                similarities, indices = self.index.search(
                    query_vectors.astype(np.float32),
                    self.top_k
                )

                # Process results and filter by threshold (Simplified Version)
                for query_id, scores, indices in zip(query_entity_ids, similarities, indices):
                    for score, faiss_id in zip(scores, indices):

                        # 1. Is the score high enough?
                        # 2. Is the FAISS index valid?
                        if score < self.similarity_threshold:
                            continue

                        neighbor_id = self.entity_id_mapping.get(faiss_id)

                        # 1. Is the neighbor a valid entity?
                        # 2. Is it not a self-match (e.g., A finding A)?
                        # 3. Is it not a duplicate pair (we already have B,A)?
                        if neighbor_id is not None and neighbor_id != query_id and query_id < neighbor_id:
                            similar_pairs.append(
                                EntityPair(
                                    entity1_id=query_id,
                                    entity2_id=neighbor_id,
                                    similarity=float(score)
                                ))
                batch_count += 1
                offset += current_batch_size

                 # Periodically save to database to manage memory
                if len(similar_pairs) >= 10000:  # Save every 100k pairs
                    self._save_similarity_pairs_to_db(similar_pairs)
                    similar_pairs = []

        # Save remaining pairs
        if similar_pairs:
            self._save_similarity_pairs_to_db(similar_pairs)

        logger.info("Similarity calculation completed and saved to database")
        return []  # Return empty since we're saving directly to DB


class GraphGrouper:
    """Groups entities using graph-based connected components by loading similarity pairs from database"""

    def __init__(self, batch_size: int = 100000):
        """
        Initialize GraphGrouper

        Args:
            batch_size: Number of similarity pairs to load per batch from database
        """
        self.batch_size = batch_size

        # Regular expression patterns for identifying various non-mergeable entity types
        # Date patterns (e.g., "January 2022", "01/15/2023")
        self.date_pattern = re.compile(
            r'^(january|february|march|april|may|june|july|august|september|october|november|december)(\s+\d{1,2}(,?\s+\d{4})?|\s+\d{4})',
            re.IGNORECASE)
        self.short_date_pattern = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})')
        # Year pattern (e.g., "2023")
        self.year_pattern = re.compile(r'^\d{4}$')
        # Monetary value pattern (e.g., "$10 million", "5.2 billion")
        self.monetary_pattern = re.compile(
            r'^(\$?\d+(\.\d+)?\s*(million|billion|trillion|thousand|m|b|t|k)|\d+(\.\d+)?\s*(million|billion|trillion|thousand))',
            re.IGNORECASE)
        # Percentage pattern (e.g., "25 percent", "10.5%")
        self.percentage_pattern = re.compile(r'^(\d+(\.\d+)?\s*percent|\d+(\.\d+)?%)', re.IGNORECASE)
        # Simple number pattern (e.g., "123")
        self.number_pattern = re.compile(r'^\d+$')
        # Decimal number pattern (e.g., "123.45")
        self.decimal_pattern = re.compile(r'^\d+\.\d+$')
        # Ratio pattern (e.g., "16:9")
        self.ratio_pattern = re.compile(r'^\d+:\d+$')
        # Measurement pattern (e.g., "10km", "150mg", "2hrs")
        self.measurement_pattern = re.compile(
            r'^\d+(\.\d+)?\s*(km|m|cm|mm|mi|ft|in|kg|g|mg|lb|oz|l|ml|gal|hrs?|min|sec)$',
            re.IGNORECASE)

    def is_mergeable_entity(self, entity_name: str) -> bool:
        """
        Determines if an entity should be considered for merging.
        Returns False for non-mergeable entities such as dates, monetary values,
        percentages, and bare numbers, which should not be merged with similar entities.

        Args:
            entity_name: The name of the entity to check

        Returns:
            bool: True if entity can be merged, False if it should be excluded
        """
        if not entity_name or not entity_name.strip():
            return False

        # Convert to lowercase for pattern matching
        name_lower = entity_name.strip().lower()

        # Check against patterns that should not be merged
        # Returns False if the entity matches any of the unmergeable patterns
        if (self.date_pattern.match(name_lower) or
                self.short_date_pattern.match(name_lower) or
                self.year_pattern.match(name_lower) or
                self.monetary_pattern.match(name_lower) or
                self.percentage_pattern.match(name_lower) or
                self.number_pattern.match(name_lower) or
                self.decimal_pattern.match(name_lower) or
                self.ratio_pattern.match(name_lower) or
                self.measurement_pattern.match(name_lower)
        ):
            logger.debug(f"Entity '{entity_name}' marked as non-mergeable (matches pattern)")
            return False
        return True

    def _filter_mergeable_entities(self, entity_names: Dict[int, str]) -> Dict[int, str]:
        """
        Filter out non-mergeable entities from the entity names dictionary

        Args:
            entity_names: Dict mapping entity_id -> entity_name

        Returns:
            Dict with only mergeable entities
        """
        filtered_entities = {}
        excluded_count = 0

        for entity_id, entity_name in entity_names.items():
            if self.is_mergeable_entity(entity_name):
                filtered_entities[entity_id] = entity_name
            else:
                excluded_count += 1
                logger.debug(f"Excluding non-mergeable entity: {entity_name}")

        if excluded_count > 0:
            logger.info(f"Filtered out {excluded_count} non-mergeable entities (dates, numbers, etc.)")

        return filtered_entities

    def _load_similarity_pairs_from_db(self, similarity_threshold: float = 0.85) -> List[EntityPair]:
        """Load all similarity pairs from database in batches"""
        logger.info(f"Loading similarity pairs from database with threshold >= {similarity_threshold}")

        all_pairs = []
        offset = 0

        with ResolutionQueries() as db:
            # Get total count first
            total_pairs = db.count_similarity_pairs(similarity_threshold)
            logger.info(f"Total similarity pairs to load: {total_pairs}")

            if total_pairs == 0:
                return []

            with tqdm(total=total_pairs, desc="Loading similarity pairs") as pbar:
                while True:
                    # Load batch of similarity pairs
                    batch_pairs = db.load_similarity_pairs_batch(
                        offset=offset,
                        limit=self.batch_size,
                        min_similarity=similarity_threshold
                    )

                    if not batch_pairs:
                        break

                    # Convert to EntityPair objects
                    for pair_data in batch_pairs:
                        all_pairs.append(EntityPair(
                            entity1_id=pair_data['entity1_id'],
                            entity2_id=pair_data['entity2_id'],
                            similarity=pair_data['similarity']
                        ))

                    offset += self.batch_size
                    pbar.update(len(batch_pairs))

                    # Break if we got fewer results than batch size (last batch)
                    if len(batch_pairs) < self.batch_size:
                        break

        logger.info(f"Loaded {len(all_pairs)} similarity pairs from database")
        return all_pairs

    def _build_graph_from_pairs(self, pairs: List[EntityPair]) -> nx.Graph:
        """Build NetworkX graph from entity pairs"""
        logger.info(f"Building graph from {len(pairs)} similarity pairs")

        G = nx.Graph()

        # Add edges with weights in batches for memory efficiency
        batch_size = 50000
        for i in tqdm(range(0, len(pairs), batch_size), desc="Adding edges to graph"):
            batch_pairs = pairs[i:i + batch_size]

            # Add edges for this batch
            edges_with_weights = [
                (pair.entity1_id, pair.entity2_id, {'weight': pair.similarity})
                for pair in batch_pairs
            ]
            G.add_edges_from(edges_with_weights)

        logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def group_entities(self, similarity_threshold: float = 0.90, filter_non_mergeable: bool = True) -> List[EntityGroup]:
        """
        Group entities into connected components by loading data from database

        Args:
            similarity_threshold: Minimum similarity threshold for pairs to include
            filter_non_mergeable: Whether to filter out non-mergeable entities (dates, numbers, etc.)

        Returns:
            List of EntityGroup objects representing connected components
        """
        logger.info("Starting entity grouping process")

        # Step 1: Load similarity pairs from database
        pairs = self._load_similarity_pairs_from_db(similarity_threshold)

        if not pairs:
            logger.info("No similarity pairs found in database")
            return []

        # Step 2: Filter out non-mergeable entities if requested
        if filter_non_mergeable:
            logger.info("Filtering non-mergeable entities from similarity pairs")

            # Get all entity IDs from pairs
            entity_ids = set()
            for pair in pairs:
                entity_ids.update([pair.entity1_id, pair.entity2_id])

            # Load entity names for filtering
            with ResolutionQueries() as db:
                entity_names = db.get_entity_names(list(entity_ids))

            # Filter out pairs containing non-mergeable entities
            filtered_pairs = []
            excluded_pairs = 0

            for pair in pairs:
                name1 = entity_names.get(pair.entity1_id, "")
                name2 = entity_names.get(pair.entity2_id, "")

                if (self.is_mergeable_entity(name1) and self.is_mergeable_entity(name2)):
                    filtered_pairs.append(pair)
                else:
                    excluded_pairs += 1

            logger.info(f"Filtered similarity pairs: {len(pairs)} → {len(filtered_pairs)} (excluded {excluded_pairs} pairs with non-mergeable entities)")
            pairs = filtered_pairs

        if not pairs:
            logger.info("No similarity pairs remaining after filtering")
            return []

        # Step 3: Build NetworkX graph
        G = self._build_graph_from_pairs(pairs)

        if G.number_of_nodes() == 0:
            logger.info("No nodes in graph")
            return []

        # Step 4: Find connected components
        logger.info("Finding connected components")
        components = list(nx.connected_components(G))
        logger.info(f"Found {len(components)} connected components")

        # Step 5: Create entity groups
        groups = []
        for i, component in enumerate(tqdm(components, desc="Processing components")):
            if len(component) > 1:  # Only groups with multiple entities
                entity_ids = list(component)

                # Choose representative (highest degree centrality)
                subgraph = G.subgraph(component)
                centrality = nx.degree_centrality(subgraph)
                representative_id = max(centrality.keys(), key=lambda x: centrality[x])

                # Calculate average confidence (mean of edge weights in component)
                edge_weights = [G[u][v]['weight'] for u, v in subgraph.edges()]
                confidence = sum(edge_weights) / len(edge_weights) if edge_weights else 0.0

                group = EntityGroup(
                    entity_ids=entity_ids,
                    representative_id=representative_id,
                    confidence=confidence
                )
                groups.append(group)

                # Log group details
                logger.debug(f"Group {i+1}: {len(entity_ids)} entities, "
                           f"representative: {representative_id}, "
                           f"confidence: {confidence:.3f}")

        logger.info(f"Created {len(groups)} entity groups from {len(components)} components")

        # Log summary statistics
        if groups:
            group_sizes = [len(group.entity_ids) for group in groups]
            logger.info(f"Group size statistics:")
            logger.info(f"  - Total entities in groups: {sum(group_sizes)}")
            logger.info(f"  - Average group size: {np.mean(group_sizes):.2f}")
            logger.info(f"  - Largest group size: {max(group_sizes)}")
            logger.info(f"  - Smallest group size: {min(group_sizes)}")

            confidences = [group.confidence for group in groups]
            logger.info(f"Confidence statistics:")
            logger.info(f"  - Average confidence: {np.mean(confidences):.3f}")
            logger.info(f"  - Highest confidence: {max(confidences):.3f}")
            logger.info(f"  - Lowest confidence: {min(confidences):.3f}")

        return groups

    def refine_groups_with_string_similarity(self,
                                             groups: List[EntityGroup],
                                             string_similarity_threshold: float = 90.0,
                                             min_group_size: int = 2,
                                             batch_size: int = 10000,
                                             filter_non_mergeable: bool = True) -> List[EntityGroup]:
        """
        Refine entity groups using RapidFuzz string similarity with graph-based clustering
        Uses RapidFuzz (MIT licensed, fast C++ implementation)
        Processes groups in batches for memory efficiency

        Args:
            groups: Original groups from connected components
            string_similarity_threshold: Minimum string similarity to stay in same group (0-100)
            min_group_size: Minimum entities per refined group
            batch_size: Number of groups to process per batch (default 10000)
            filter_non_mergeable: Whether to filter out non-mergeable entities

        Returns:
            List of refined EntityGroup objects split by string similarity
        """
        if not groups:
            return groups

        logger.info(f"Refining {len(groups)} groups using graph-based string similarity (threshold={string_similarity_threshold}, batch_size={batch_size}, filter_non_mergeable={filter_non_mergeable})")

        refined_groups = []
        total_splits = 0

        # Process groups in batches
        for batch_start in tqdm(range(0, len(groups), batch_size), desc="Processing group batches"):
            batch_end = min(batch_start + batch_size, len(groups))
            group_batch = groups[batch_start:batch_end]

            # Collect all entity IDs for this batch
            all_entity_ids = []
            for group in group_batch:
                all_entity_ids.extend(group.entity_ids)

            # Load all entity names for this batch in one query
            batch_entity_names = self._get_entity_names_for_batch(all_entity_ids)

            # Process each group in the batch
            for i, group in enumerate(group_batch):
                # Small groups don't need refinement
                if len(group.entity_ids) <= 2:
                    refined_groups.append(group)
                    continue

                # Extract names for this specific group
                entity_names = {
                    entity_id: batch_entity_names.get(entity_id, f"Entity_{entity_id}")
                    for entity_id in group.entity_ids
                    if entity_id in batch_entity_names
                }

                # Filter out non-mergeable entities if requested
                if filter_non_mergeable:
                    entity_names = self._filter_mergeable_entities(entity_names)

                if len(entity_names) < 2:
                    # If no mergeable entities or only one left, skip refinement
                    refined_groups.append(group)
                    continue

                # Create subgroups based on graph-based string similarity clustering
                subgroups = self._cluster_by_string_similarity(
                    entity_names,
                    string_similarity_threshold
                )

                # Convert subgroups back to EntityGroup objects
                group_objects = []
                for subgroup_entity_ids in subgroups:
                    if len(subgroup_entity_ids) >= min_group_size:
                        group_objects.append(EntityGroup(
                            entity_ids=subgroup_entity_ids,
                            representative_id=subgroup_entity_ids[0],
                            confidence=group.confidence
                        ))

                if len(group_objects) > 1:
                    logger.info(f"Batch {batch_start//batch_size + 1}, Group {i+1}: "
                               f"Split {len(group.entity_ids)} entities into {len(group_objects)} subgroups")
                    total_splits += len(group_objects) - 1
                    refined_groups.extend(group_objects)
                else:
                    refined_groups.append(group)

        logger.info(f"Graph-based string similarity refinement: {len(groups)} → {len(refined_groups)} groups (+{total_splits} splits)")
        return refined_groups

    def _get_entity_names_for_batch(self, entity_ids: List[int]) -> Dict[int, str]:
        """Get entity names from database for a batch of entity IDs"""
        if not entity_ids:
            return {}

        # Remove duplicates while preserving order
        unique_entity_ids = list(dict.fromkeys(entity_ids))

        try:
            with ResolutionQueries() as db:
                return db.get_entity_names(unique_entity_ids)
        except Exception as e:
            logger.error(f"Failed to get entity names for batch: {e}")
            return {}

    def _cluster_by_string_similarity(self,
                                      entity_names: Dict[int, str],
                                      threshold: float) -> List[List[int]]:
        """
        Cluster entities by string similarity using a robust graph-based approach.
        This correctly handles transitive relationships (A-B, B-C -> {A,B,C}).

        Args:
            entity_names: Dict mapping entity_id -> entity_name
            threshold: Minimum similarity to be in same cluster (0-100)

        Returns:
            List of lists, each containing entity_ids for a refined cluster.
        """
        if len(entity_names) < 2:
            return [list(entity_names.keys())]

        # Get all unique pairs using combinations
        node_ids = list(entity_names.keys())
        all_pairs = combinations(node_ids, 2)

        # Build a temporary graph for this small group
        G_refine = nx.Graph()
        G_refine.add_nodes_from(node_ids)

        # Add edges between similar entities
        for id1, id2 in all_pairs:
            name1 = entity_names[id1].strip()
            name2 = entity_names[id2].strip()

            # Calculate string similarity using RapidFuzz
            similarity = fuzz.ratio(name1, name2)

            if similarity >= threshold:
                G_refine.add_edge(id1, id2, weight=similarity)

        # Find connected components in the refinement graph
        # Each component is a correctly formed cluster
        refined_clusters = list(nx.connected_components(G_refine))

        # Convert set components to lists of integers
        clusters = [list(component) for component in refined_clusters]

        return clusters

@dataclass
class EntityResolutionResult:
    """Represents the result of LLM-based entity resolution"""
    should_merge: bool
    resolved_name: str
    confidence: str
    reasoning: str

class EntityResolution:
    """
    Orchestrates the complete entity resolution workflow:
    1. FAISS similarity calculation
    2. Graph-based grouping
    3. String similarity refinement
    4. LLM-based merge validation
    5. Database updates
    """

    def __init__(self,
                 vector_similarity_threshold: float = 0.85,
                 graph_similarity_threshold: float = 0.90,
                 string_similarity_threshold: float = 90.0,
                 min_group_size: int = 2,
                 llm_provider: str = "openai",
                 llm_temperature: float = 0.1,
                 batch_size: int = 10000):
        """
        Initialize EntityResolution orchestrator

        Args:
            vector_similarity_threshold: Threshold for FAISS similarity calculation
            graph_similarity_threshold: Threshold for initial graph grouping
            string_similarity_threshold: Threshold for string similarity refinement
            min_group_size: Minimum entities per group
            llm_provider: LLM provider to use ('openai', 'deepseek')
            llm_temperature: Temperature for LLM requests
            batch_size: Batch size for processing
        """
        self.vector_similarity_threshold = vector_similarity_threshold
        self.graph_similarity_threshold = graph_similarity_threshold
        self.string_similarity_threshold = string_similarity_threshold
        self.min_group_size = min_group_size
        self.llm_provider = llm_provider
        self.llm_temperature = llm_temperature
        self.batch_size = batch_size

        # Initialize components
        self.faiss_calculator = FAISSSimilarityCalculator(
            similarity_threshold=vector_similarity_threshold
        )
        self.grouper = GraphGrouper(batch_size=batch_size)

        logger.info(f"EntityResolution initialized with provider: {llm_provider}")

    def run_complete_resolution(self):
        """
        Run the complete entity resolution pipeline

        Args:
            skip_faiss: Skip FAISS training and similarity calculation (use existing data)
            skip_llm_validation: Skip LLM validation step

        Returns:
            List of final resolved entity groups
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE ENTITY RESOLUTION PIPELINE")
        logger.info("=" * 60)

        # Step 1: FAISS similarity calculation
        logger.info("Step 1: Training FAISS index and calculating similarities")
        self.faiss_calculator.train_index()
        self.faiss_calculator.calculate_similarity()
        del self.faiss_calculator.index

        # Step 2: Initial graph-based grouping
        logger.info("Step 2: Creating initial entity groups from vector similarities")
        initial_groups = self.grouper.group_entities(
            similarity_threshold=self.graph_similarity_threshold,
            filter_non_mergeable=True
        )

        if not initial_groups:
            logger.warning("No initial groups found. Ending pipeline.")
            return []

        logger.info(f"Created {len(initial_groups)} initial groups")

        # Step 3: String similarity refinement
        logger.info("Step 3: Refining groups with string similarity")
        refined_groups = self.grouper.refine_groups_with_string_similarity(
            groups=initial_groups,
            string_similarity_threshold=self.string_similarity_threshold,
            min_group_size=self.min_group_size,
            batch_size=self.batch_size,
            filter_non_mergeable=True
        )

        logger.info(f"Refined to {len(refined_groups)} groups")

        # Step 4: LLM validation and final resolution
        logger.info("Step 4: LLM validation and final entity resolution")
        final_groups = self._llm_validate_and_resolve(refined_groups)

        # Step 5: Update database with final resolutions
        logger.info("Step 5: Updating database with resolved entities")
        self._update_database_with_resolutions(final_groups)

        logger.info("=" * 60)
        logger.info("ENTITY RESOLUTION PIPELINE COMPLETED")
        logger.info(f"Final result: {len(final_groups)} resolved entity groups")
        logger.info("=" * 60)

        return final_groups

    def _llm_validate_and_resolve(self, groups: List[EntityGroup]) -> List[EntityGroup]:
        """
        Use LLM to validate entity groups and determine final resolutions

        Args:
            groups: List of entity groups to validate

        Returns:
            List of validated and resolved entity groups
        """
        if not groups:
            return groups

        logger.info(f"Starting LLM validation for {len(groups)} entity groups")

        validated_groups = []
        merge_decisions = []

        # Process groups in batches to manage API costs
        llm_batch_size = 100  # Process 100 groups at a time for LLM

        for batch_start in tqdm(range(0, len(groups), llm_batch_size), desc="LLM validation batches"):
            batch_end = min(batch_start + llm_batch_size, len(groups))
            group_batch = groups[batch_start:batch_end]

            # Get entity names for this batch
            all_entity_ids = []
            for group in group_batch:
                all_entity_ids.extend(group.entity_ids)

            entity_names = self.grouper._get_entity_names_for_batch(all_entity_ids)

            # Process each group with LLM
            for i, group in enumerate(group_batch):
                try:
                    # Get entity names for this group
                    group_entity_names = [
                        entity_names.get(entity_id, f"Entity_{entity_id}")
                        for entity_id in group.entity_ids
                    ]

                    # Skip if we don't have names
                    if not any(name for name in group_entity_names if not name.startswith("Entity_")):
                        validated_groups.append(group)
                        continue

                    # Get LLM decision
                    resolution_result = self._get_llm_resolution_decision(group_entity_names)

                    if resolution_result.should_merge:
                        # Create resolved group with LLM-determined name
                        resolved_group = EntityGroup(
                            entity_ids=group.entity_ids,
                            representative_id=group.representative_id,
                            confidence=group.confidence
                        )
                        validated_groups.append(resolved_group)

                        # Store merge decision for database update
                        merge_decisions.append({
                            'entity_ids': group.entity_ids,
                            'resolved_name': resolution_result.resolved_name,
                            'confidence': resolution_result.confidence,
                            'reasoning': resolution_result.reasoning
                        })

                        logger.info(f"LLM approved merge: {group_entity_names} → '{resolution_result.resolved_name}'")
                    else:
                        # LLM rejected merge - split back into individual entities
                        logger.info(f"LLM rejected merge: {group_entity_names}")
                        for entity_id in group.entity_ids:
                            individual_group = EntityGroup(
                                entity_ids=[entity_id],
                                representative_id=entity_id,
                                confidence=0.0
                            )
                            validated_groups.append(individual_group)

                except Exception as e:
                    logger.error(f"Error in LLM validation for group {i+1}: {e}")
                    # On error, keep original group
                    validated_groups.append(group)

        logger.info(f"LLM validation complete: {len(groups)} → {len(validated_groups)} groups")
        logger.info(f"LLM approved {len(merge_decisions)} merges")

        # Store merge decisions for database update
        self.merge_decisions = merge_decisions

        return validated_groups

    def _get_llm_resolution_decision(self, entity_names: List[str]) -> EntityResolutionResult:
        """
        Get LLM decision on whether entities should be merged and the resolved name

        Args:
            entity_names: List of entity names to evaluate

        Returns:
            EntityResolutionResult with merge decision and resolved name
        """
        # Create system prompt for entity resolution
        system_prompt = """You are an expert in entity resolution and data deduplication. Your task is to determine if a group of entity names should be merged into a single entity, and if so, what the canonical name should be.
                    Guidelines:
                    1. Merge entities that refer to the same real-world entity (person, organization, location, etc.)
                    2. Consider variations in formatting, abbreviations, and minor spelling differences
                    3. Do NOT merge entities that are clearly different (e.g., different people with similar names)
                    4. Choose the most complete and formal name as the resolved name
                    5. Be conservative - when in doubt, do not merge
                    
                    Response format:
                    Decision: [YES/NO]
                    Resolved Name: [The canonical name to use, or N/A if NO]
                    Confidence: [HIGH/MEDIUM/LOW]
                    Reasoning: [Brief explanation of your decision]"""

        # Create user prompt with entity names
        entity_list = "\n".join([f"- {name}" for name in entity_names])
        user_prompt = f"""Analyze these entity names and determine if they should be merged:
        {entity_list}
        
        Should these entities be merged into a single entity? If yes, what should be the canonical name?"""

        try:
            # Send request to LLM
            llm_client = LLMClient(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.llm_temperature,
                provider=self.llm_provider
            )

            response = llm_client.send_message()

            if not response:
                logger.warning("Empty LLM response, defaulting to no merge")
                return EntityResolutionResult(
                    should_merge=False,
                    resolved_name="",
                    confidence="LOW",
                    reasoning="LLM response was empty"
                )

            # Parse LLM response
            return self._parse_llm_response(response)

        except Exception as e:
            logger.error(f"Error getting LLM decision: {e}")
            return EntityResolutionResult(
                should_merge=False,
                resolved_name="",
                confidence="LOW",
                reasoning=f"Error: {str(e)}"
            )

    def _parse_llm_response(self, response: str) -> EntityResolutionResult:
        """
        Parse LLM response into structured result

        Args:
            response: Raw LLM response text

        Returns:
            EntityResolutionResult object
        """
        lines = response.strip().split('\n')

        decision = False
        resolved_name = ""
        confidence = "LOW"
        reasoning = ""

        for line in lines:
            line = line.strip()
            if line.startswith("Decision:"):
                decision_text = line.split(":", 1)[1].strip().upper()
                decision = decision_text.startswith("YES")
            elif line.startswith("Resolved Name:"):
                resolved_name = line.split(":", 1)[1].strip()
                if resolved_name.upper() == "N/A":
                    resolved_name = ""
            elif line.startswith("Confidence:"):
                confidence = line.split(":", 1)[1].strip().upper()
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        return EntityResolutionResult(
            should_merge=decision,
            resolved_name=resolved_name,
            confidence=confidence,
            reasoning=reasoning
        )

    def _update_database_with_resolutions(self, final_groups: List[EntityGroup]):
        """
        Update database with final entity resolutions

        Args:
            final_groups: List of final resolved entity groups
        """
        if not hasattr(self, 'merge_decisions'):
            logger.info("No merge decisions to update in database")
            return

        logger.info(f"Updating database with {len(self.merge_decisions)} entity resolutions")

        try:
            with ResolutionQueries() as db:
                for decision in self.merge_decisions:
                    # Update entity resolution in database
                    # This would depend on your specific database schema
                    # Example implementation:
                    db.update_entity_resolution(
                        entity_ids=decision['entity_ids'],
                        resolved_name=decision['resolved_name'],
                        confidence=decision['confidence'],
                        reasoning=decision['reasoning']
                    )

            logger.info("Database updates completed successfully")

        except Exception as e:
            logger.error(f"Error updating database with resolutions: {e}")

    def get_resolution_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the resolution process

        Returns:
            Dictionary with resolution statistics
        """
        stats = {}

        if hasattr(self, 'merge_decisions'):
            stats['total_merges'] = len(self.merge_decisions)
            stats['high_confidence_merges'] = len([d for d in self.merge_decisions if d['confidence'] == 'HIGH'])
            stats['medium_confidence_merges'] = len([d for d in self.merge_decisions if d['confidence'] == 'MEDIUM'])
            stats['low_confidence_merges'] = len([d for d in self.merge_decisions if d['confidence'] == 'LOW'])

        return stats

    def print_resolution_summary(self):
        """Print a summary of the resolution process"""
        stats = self.get_resolution_statistics()

        print("\n" + "=" * 60)
        print("ENTITY RESOLUTION SUMMARY")
        print("=" * 60)

        if stats:
            print(f"Total merges approved by LLM: {stats.get('total_merges', 0)}")
            print(f"High confidence merges: {stats.get('high_confidence_merges', 0)}")
            print(f"Medium confidence merges: {stats.get('medium_confidence_merges', 0)}")
            print(f"Low confidence merges: {stats.get('low_confidence_merges', 0)}")
        else:
            print("No merge statistics available")

        print("=" * 60)
