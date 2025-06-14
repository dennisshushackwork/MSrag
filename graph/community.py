"""
A class designed to perform community detection on a graph using hierarchical leiden algorithm.
1. Loads the graph
2. Performs community detection using Leiden algorithm.
3. Summarises the community detection results (summarises the relationships within each community).
4. Includes fallback mechanism for failed LLM calls and cleanup of failed communities.
"""

# External imports:
import os
import time
import logging
import networkx as nx
from dotenv import load_dotenv
from graspologic.partition import hierarchical_leiden
from typing import List, Dict, Tuple, Set
import asyncio

# Internal imports:
from postgres.community import CommunityQueries
from llm.prompts.community.summary import CommunitySummary  # Assuming this is your class
from emb.communities import CommunityGroupEmbedder

# Setting up logging & environmental variables:
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class CommunityDetection:
    """
    Performs community detection using the Hierarchical Leiden algorithm and
    creates the community summaries asynchronously with fallback mechanism.
    """

    def __init__(self, batch_size: int = 1_000_000):
        # Defines the batch size for loading entities and relationships from the database.
        self.batch_size: int = batch_size
        # Initializes an empty NetworkX graph to represent entities and their relationships.
        self.G = nx.Graph()
        # Tracks the number of nodes (entities) in the graph.
        self.node_count: int = 0
        # Tracks the number of edges (relationships) in the graph.
        self.edge_count: int = 0
        # Stores the results of the Leiden algorithm's community partitions.
        self.partitions = None
        # Maximum allowed size for a cluster (community). Loaded from environment variables. (default=10 microsoft)
        self.max_cluster_size: int = int(os.getenv("MAX_CLUSTER_SIZE", 10))
        # Maximum hierarchical level for community detection. Loaded from environment variables.
        self.max_level: int = int(os.getenv("MAX_CLUSTER_LEVEL", 4))
        # Batch size for processing communities during summarization.
        self.summarization_community_process_batch_size: int = 30
        # Token limit for the LLM context when generating summaries.
        self.llm_context_limit: int = 8000
        # Name of the LLM model to use for summarization. Defaults to "openai".
        self.llm_model_name: str = os.getenv("LLM_MODEL_NAME", "openai")
        # Maps Leiden cluster IDs to their corresponding database community IDs.
        self.leiden_id_to_db_id_map: Dict = {}
        # Track failed communities for fallback and potential deletion
        self.failed_communities: Set[int] = set()
        self.too_long_communities: Set[int] = set()

    def build_graph(self):
        """
        Builds a NetworkX graph by loading entities and relationships from the database
        in batches.
        """
        logger.info("Building the networkx graph from the database.")
        entity_offset = 0
        total_entities = 0
        self.G.clear()

        # Loop to load entities in batches until all are processed.
        while True:
            with CommunityQueries() as db:
                entities_dict = db.load_entities_in_batches(batch_size=self.batch_size, offset=entity_offset)
                if not entities_dict:
                    logger.info("All entities have been loaded from the database.")
                    break
                # logger.info(f"Processing entity batch with offset {entity_offset}") # Can be verbose
                for row in entities_dict:
                    # Adds each entity as a node to the graph.
                    self.G.add_node(row["entity_id"])
                    total_entities += 1
                entity_offset += self.batch_size
        logger.info(f"All entities have been loaded. Total entities loaded: {total_entities}")

        logger.info("Loading relationships and adding them as edges...")
        rel_offset = 0
        total_relationships = 0
        # Loop to load relationships in batches until all are processed.
        while True:
            with CommunityQueries() as db:
                rel_dict = db.load_relationships_in_batches(self.batch_size, rel_offset)
                if not rel_dict:
                    logger.info("All relationships have been loaded.")
                    break
                # logger.info(f"Processing relationships batch with offset {rel_offset}") # Can be verbose
                for row in rel_dict:
                    from_entity = row['from_entity']
                    to_entity = row['to_entity']
                    # Adds an edge to the graph if both entities exist as nodes.
                    if self.G.has_node(from_entity) and self.G.has_node(to_entity):
                        self.G.add_edge(
                            from_entity,
                            to_entity,
                            rel_id=row.get('rel_id'),
                            weight=row.get('rel_weight', 1)
                        )
                        total_relationships += 1
                rel_offset += self.batch_size
        logger.info(f"Total relationships added as edges: {total_relationships}")

        # Identifies and removes isolated nodes (entities without any relationships).
        isolates = list(nx.isolates(self.G))
        if isolates:
            logger.info(f"Removing {len(isolates)} non-connected (isolated) entities...")
            self.G.remove_nodes_from(isolates)

        # Updates the node and edge counts after graph construction.
        self.node_count = self.G.number_of_nodes()
        self.edge_count = self.G.number_of_edges()
        logger.info(f"Graph has been constructed. Edge count: {self.edge_count}, Node count: {self.node_count}")

    def create_communities(self):  # Stays synchronous
        """
        Applies the Hierarchical Leiden algorithm to detect communities within the graph.
        Handles cases where the graph has no nodes or no edges.
        """
        if self.G.number_of_nodes() == 0:
            logger.warning("Graph has no nodes. Skipping community detection.")
            self.partitions = []
            return
        if self.G.number_of_edges() == 0:
            logger.warning("Graph has no edges. Using fallback for community detection (each node is a community).")
            self.partitions = []
            # For graphs with no edges, each node is considered its own community.
            for i, node_id in enumerate(self.G.nodes()):
                # Mimics the structure of a graspologic PartitionTuple for consistency.
                from collections import namedtuple
                PartitionTuple = namedtuple("PartitionTuple",
                                            ["node", "cluster", "level", "parent_cluster", "modularity",
                                             "normalized_modularity"])
                self.partitions.append(PartitionTuple(node_id, i, 0, None, 0.0, 0.0))
            logger.info(f"Created {len(self.partitions)} individual communities for edgeless graph.")
            return

        logger.info("Applying hierarchical Leiden clustering...")
        try:
            # Executes the Hierarchical Leiden algorithm.
            self.partitions = hierarchical_leiden(self.G, max_cluster_size=self.max_cluster_size)
            logger.info("Hierarchical Leiden clustering completed successfully.")
        except Exception as e:
            logger.error(f"Error during hierarchical Leiden clustering: {e}")
            self.partitions = []  # Ensures partitions is empty on error

    def aggregate_communities(self):
        """
        Aggregates the detected communities, maps Leiden IDs to database IDs,
        and stores community and entity-community associations in the database.
        """
        if not self.partitions:
            logger.warning("No partitions found from community detection. Skipping aggregation.")
            return

        clusters_by_leiden_id = {}
        # Organizes partitions by their Leiden cluster ID.
        for part in self.partitions:
            if part.level > self.max_level:
                continue
            leiden_cluster_id = part.cluster
            if leiden_cluster_id not in clusters_by_leiden_id:
                clusters_by_leiden_id[leiden_cluster_id] = {
                    'nodes': [],
                    'parent_cluster': part.parent_cluster,  # This is a Leiden ID
                    'level': part.level
                }
            clusters_by_leiden_id[leiden_cluster_id]['nodes'].append(part.node)

        communities_to_insert_temp = []  # Store with leiden parent id first
        leiden_cluster_ids_ordered = []

        # Sort communities by level to ensure parents are created before children
        sorted_clusters = sorted(clusters_by_leiden_id.items(), key=lambda x: x[1]['level'])

        # Prepares community data for insertion, including their Leiden IDs and levels.
        for leiden_cluster_id, cluster_data in sorted_clusters:
            cluster_data['num_nodes'] = len(list(set(cluster_data['nodes'])))
            # Store raw leiden parent_cluster id for now
            communities_to_insert_temp.append({
                'leiden_id': leiden_cluster_id,
                'level': cluster_data['level'],
                'leiden_parent_id': cluster_data['parent_cluster'],
                'num_nodes': cluster_data['num_nodes'],
                'community_embed': True
            })
            leiden_cluster_ids_ordered.append(leiden_cluster_id)  # Keep order for ID mapping

        if not communities_to_insert_temp:
            logger.info("No communities to insert after aggregation.")
            return

        # Process communities level by level to handle parent-child relationships properly
        communities_by_level = {}
        for community_data in communities_to_insert_temp:
            level = community_data['level']
            if level not in communities_by_level:
                communities_by_level[level] = []
            communities_by_level[level].append(community_data)

        self.leiden_id_to_db_id_map.clear()

        # Process each level in order (0, 1, 2, ...)
        for level in sorted(communities_by_level.keys()):
            level_communities = communities_by_level[level]
            logger.info(f"Processing {len(level_communities)} communities at level {level}")

            # Prepare insertion data for this level
            community_group_values = []
            for data in level_communities:
                # For level 0, parent is always None
                # For level > 0, look up parent DB ID from mapping
                db_parent_id = None
                if data['leiden_parent_id'] is not None:
                    db_parent_id = self.leiden_id_to_db_id_map.get(data['leiden_parent_id'])
                    if db_parent_id is None:
                        logger.warning(f"Could not find DB ID for parent Leiden ID {data['leiden_parent_id']} at level {level}")

                community_group_values.append((
                    data['level'],
                    db_parent_id,
                    data['num_nodes'],
                    data['community_embed']
                ))

            # Insert communities for this level
            with CommunityQueries() as db:
                new_db_community_ids = db.insert_community_groups(community_group_values)

            logger.info(f"Level {level}: Generated DB Community IDs: {new_db_community_ids}")

            # Update the mapping for this level
            if len(level_communities) == len(new_db_community_ids):
                for i, community_data in enumerate(level_communities):
                    leiden_id = community_data['leiden_id']
                    self.leiden_id_to_db_id_map[leiden_id] = new_db_community_ids[i]
            else:
                logger.error(f"Mismatch between number of communities and DB IDs at level {level}")
                return

        logger.info(f"Complete Leiden ID to DB ID mapping: {self.leiden_id_to_db_id_map}")

        # Prepares entity-community associations for insertion into the database.
        entity_communities_to_insert = []
        for leiden_cluster_id, cluster_data in clusters_by_leiden_id.items():
            db_community_id = self.leiden_id_to_db_id_map.get(leiden_cluster_id)
            if db_community_id is None:
                logger.warning(
                    f"No DB ID found for Leiden cluster ID {leiden_cluster_id} when preparing CommunityNode. Skipping its nodes.")
                continue
            for node_id in cluster_data['nodes']:
                entity_communities_to_insert.append((db_community_id, node_id))

        if entity_communities_to_insert:
            with CommunityQueries() as db:
                db.insert_community_nodes(entity_communities_to_insert)
            logger.info(f"Community-entity connections saved: {len(entity_communities_to_insert)}")

        # Creates associations between communities and documents/chunks in the database.
        all_db_community_ids = list(self.leiden_id_to_db_id_map.values())
        if all_db_community_ids:
            with CommunityQueries() as db:
                for db_comm_id in all_db_community_ids:
                    db.create_community_document_associations(db_comm_id)
                logger.info("Finished creating all community-document associations.")
                for db_comm_id in all_db_community_ids:
                    db.create_community_chunk_associations(db_comm_id)
                logger.info("Finished creating all community-chunk associations.")

        # Clears graph and partition data from memory after aggregation.
        self.partitions = None
        self.G.clear()

    def _blocking_generate_summary(self, relationship_text_for_prompt: str):
        """
        Generates a summary using a synchronous LLM client. This method is designed
        to be run in a separate thread when called from an async context.
        """
        try:
            # Initializes the CommunitySummary class with relationship text and LLM model.
            community_summary_instance = CommunitySummary(relationships=relationship_text_for_prompt,
                                                          model=self.llm_model_name)
            # Generates the summary text using the LLM.
            summary_text = community_summary_instance.generate_summary()
            return summary_text
        except Exception as e:
            logger.error(f"Error in _blocking_generate_summary: {e}")
            return None

    async def _process_community_batch_with_fallback(self, community_ids_batch: List[int],
                                                   community_relationships_data: Dict[int, List[Dict]],
                                                   attempt_number: int = 1) -> List[Tuple[str, int]]:
        """
        Process a batch of communities with fallback mechanism for failed LLM calls.

        Args:
            community_ids_batch: List of community IDs to process
            community_relationships_data: Mapping of community ID to relationships
            attempt_number: Current attempt number (1 or 2)

        Returns:
            List of tuples (summary_text, community_id) for successful summaries
        """
        tasks = []
        batch_failed_communities = set()

        # Prepare summarization tasks for each community.
        for comm_id in community_ids_batch:
            relationships = community_relationships_data.get(comm_id, [])

            if not relationships:
                logger.info(f"Community {comm_id} has no internal relationships. Scheduling placeholder summary.")
                # Creates a completed future for communities with no relationships.
                future = asyncio.Future()
                future.set_result(("No internal relationships found.", comm_id))
                tasks.append(future)
                continue

            all_rel_descriptions = [rel['description'] for rel in relationships]
            total_tokens = sum(rel.get('tokens', 0) for rel in relationships)

            # Check if the total token count exceeds the LLM context limit.
            if total_tokens > self.llm_context_limit:
                logger.warning(
                    f"Community {comm_id} relationships' total tokens ({total_tokens}) exceed LLM context limit ({self.llm_context_limit}). Marking for deletion.")
                self.too_long_communities.add(comm_id)
                continue

            relationship_text_for_prompt = "\n- ".join(all_rel_descriptions)
            if all_rel_descriptions:
                relationship_text_for_prompt = "- " + relationship_text_for_prompt

            # Creates an asyncio task to run the blocking LLM call in a separate thread
            task = asyncio.to_thread(
                lambda p, c_id: (self._blocking_generate_summary(p), c_id),
                relationship_text_for_prompt,
                comm_id
            )
            tasks.append(task)

        # Run all summarization tasks for the current batch concurrently.
        successful_summaries = []
        if tasks:
            logger.info(f"Gathering {len(tasks)} summarization tasks for concurrent execution (attempt {attempt_number})...")
            results_with_comm_id = await asyncio.gather(*tasks, return_exceptions=True)

            for result_item in results_with_comm_id:
                if isinstance(result_item, Exception):
                    logger.error(f"A summarization task failed on attempt {attempt_number}: {result_item}")
                    continue

                # Unpack the result
                summary_text_or_none, current_comm_id = result_item

                if summary_text_or_none is not None:
                    logger.info(f"Summary received for community {current_comm_id} on attempt {attempt_number}: {str(summary_text_or_none)[:100]}...")
                    successful_summaries.append((str(summary_text_or_none), current_comm_id))
                else:
                    logger.warning(f"LLM returned empty or failed summary for community {current_comm_id} on attempt {attempt_number}.")
                    batch_failed_communities.add(current_comm_id)

        # Track failed communities for potential retry
        if attempt_number == 1:
            self.failed_communities.update(batch_failed_communities)

        return successful_summaries

    async def generate_community_summaries(self):
        """
        Generates summaries for communities by fetching their internal relationships,
        prompting an LLM concurrently with fallback mechanism, and storing the summaries in the database.
        This method operates asynchronously.
        """
        logger.info("Starting asynchronous community summarization process with fallback mechanism...")
        processed_community_ids_for_summaries = []
        all_successful_summaries = []

        offset = 0
        # First pass: Process all communities
        while True:
            # Fetch community IDs for the current batch
            with CommunityQueries() as db:
                community_ids_batch = db.get_all_community_ids_for_summarization(
                    batch_size=self.summarization_community_process_batch_size,
                    offset=offset
                )

            if not community_ids_batch:
                logger.info("All communities have been processed for summarization or no communities found.")
                break

            logger.info(f"Processing batch of {len(community_ids_batch)} communities for summarization. Offset: {offset}")

            # Fetch internal relationships for the current batch of communities
            with CommunityQueries() as db:
                community_relationships_data = db.get_internal_relationships_for_communities(community_ids_batch)

            # Process batch with attempt 1
            successful_summaries = await self._process_community_batch_with_fallback(
                community_ids_batch, community_relationships_data, attempt_number=1
            )

            all_successful_summaries.extend(successful_summaries)
            offset += self.summarization_community_process_batch_size

        # Second pass: Retry failed communities (fallback mechanism)
        if self.failed_communities:
            logger.info(f"Retrying {len(self.failed_communities)} failed communities with fallback mechanism...")

            failed_community_ids = list(self.failed_communities)
            # Clear the set to track new failures in retry
            self.failed_communities.clear()

            # Fetch relationships for failed communities
            with CommunityQueries() as db:
                retry_relationships_data = db.get_internal_relationships_for_communities(failed_community_ids)

            # Process failed communities with attempt 2
            retry_successful_summaries = await self._process_community_batch_with_fallback(
                failed_community_ids, retry_relationships_data, attempt_number=2
            )

            all_successful_summaries.extend(retry_successful_summaries)
            logger.info(f"Retry completed. Successfully recovered {len(retry_successful_summaries)} communities.")

        # Update summaries in the database for all successful summaries
        if all_successful_summaries:
            # Process in batches to avoid overwhelming the database
            batch_size = 100
            for i in range(0, len(all_successful_summaries), batch_size):
                batch = all_successful_summaries[i:i + batch_size]
                with CommunityQueries() as db:
                    db.batch_update_community_summaries(batch)
                processed_community_ids_for_summaries.extend([s[1] for s in batch])

        # Delete communities that failed after all attempts or are too long
        communities_to_delete = self.failed_communities.union(self.too_long_communities)
        if communities_to_delete:
            logger.warning(f"Deleting {len(communities_to_delete)} communities that failed after all attempts or exceeded token limits...")
            await self._delete_failed_communities(list(communities_to_delete))

        # Run the embedder_mlr_test only for the communities that have successfully received a summary
        # and were not marked for deletion.
        logger.info("Triggering CommunityGroupEmbedder after community summarization and cleanup.")
        community_group_embedder = CommunityGroupEmbedder()
        community_group_embedder.process_community_group_emb_batches()

        logger.info(f"Community summarization process completed. Successfully processed {len(processed_community_ids_for_summaries)} summaries. "
                   f"Deleted {len(communities_to_delete)} failed communities.")

    async def _delete_failed_communities(self, community_ids_to_delete: List[int]):
        """
        Delete communities that failed summarization from the database.
        This includes cascading deletes for related records.
        """
        if not community_ids_to_delete:
            return

        logger.info(f"Deleting {len(community_ids_to_delete)} failed communities from database...")
        try:
            with CommunityQueries() as db:
                db.delete_failed_communities(community_ids_to_delete)
        except Exception as e:
            logger.error(f"Error deleting failed communities: {e}")

    async def build_and_save_communities(self):
        """
        Executes the full pipeline for community detection, aggregation, and
        asynchronous summarization with fallback mechanism.
        """
        start_time = time.time()
        logger.info("Starting full community detection and aggregation pipeline with fallback mechanism...")

        # Synchronous parts of the pipeline: graph building, community creation, and aggregation.
        self.build_graph()
        self.create_communities()
        self.aggregate_communities()

        logger.info("Proceeding to generate community summaries asynchronously with fallback...")
        # Await the asynchronous generation of community summaries with fallback
        await self.generate_community_summaries()

        end_time = time.time()
        logger.info(f"Full community pipeline (including async summarization with fallback) took {end_time - start_time:.2f} seconds.")

        logger.info("Clearing graph and partition data from memory.")
        # Clear graph and partition data to free up memory.
        if self.partitions is not None:
            del self.partitions
            self.partitions = None
        self.G.clear()


if __name__ == '__main__':
    logger.info("--- Starting Community Detection Script with Fallback Mechanism ---")

    # Ensure essential environment variables, like LLM_MODEL_NAME, are set.
    if not os.getenv("LLM_MODEL_NAME"):
        # It's better to ensure this is set in your .env or environment
        logger.warning("LLM_MODEL_NAME not set in environment. Using placeholder 'openai'.")
        # os.environ["LLM_MODEL_NAME"] = "openai" # Avoid setting env var here if possible

    community_detector = CommunityDetection(batch_size=100_000)

    # Run the main asynchronous method using asyncio.run().
    asyncio.run(community_detector.build_and_save_communities())

    logger.info("--- Community Detection Script Finished ---")