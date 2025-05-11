"""
Graph Database, which does BFS.
"""
# External imports:
import time
import asyncio
import logging

# Internal imports:
from graphdb.kuzugraph import KuzuGraphStore

# Load environmental variables & Set the logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MyGraphService(KuzuGraphStore):
    """
    This class implements the BFS algorithm.
    """
    def __init__(self, recreate_db: bool = False):
        super().__init__(recreate_db)
        logger.info("MyGraphService initialized and KuzuGraphStore is set up.")

    def get_entity_count(self) -> int:
        """Returns the number of entities in the database"""
        query = "MATCH (e:Entity) RETURN count(e) as count"
        result = self.connection.execute(query)
        count = result.get_next()[0] if result.has_next() else 0
        return count

    def get_relationship_count(self) -> int:
        """Returns the number of relationships in the database"""
        query = "MATCH ()-[r:Relationship]->() RETURN count(r)"
        result = self.connection.execute(query)
        count = result.get_next()[0] if result.has_next() else 0
        return count

    @staticmethod
    def bi_directional_bfs(start_entity_id: int, depth: int):
        """Creates the BFS query for the entities"""

        # Forward traversal (outgoing relationships)
        forward_query = \
            f"""
           MATCH path = (start:Entity {{id: {start_entity_id}}})-[:Relationship*1..{depth}]->(connected:Entity)
           RETURN connected.id AS connected_id, length(path) AS distance, 'outgoing' AS direction
           """
        # Backward traversal (incoming relationships)
        backward_query = f"""
          MATCH path = (start:Entity {{id: {start_entity_id}}})<-[:Relationship*1..{depth}]-(connected:Entity)
          RETURN connected.id AS connected_id, length(path) AS distance, 'incoming' AS direction
          """

        # Combine results with UNION
        combined_query = f"""
              {forward_query}
              UNION
              {backward_query}
              ORDER BY distance
              """
        return combined_query

    async def perform_bidirectional_bfs_async(self, start_entity_id, max_depth=2):
        """
        Performs a bidirectional BFS asynchronously.
        """
        query = self.bi_directional_bfs(start_entity_id, max_depth)
        result = await self.async_connection.execute(query)
        return result, start_entity_id


    async def run_multiple_bfs_searches(self, entity_ids, max_depth=2):
        """
        Runs multiple bidirectional BFS searches in parallel.
        Args:
            entity_ids: List of entity IDs to start BFS from
            max_depth: Maximum depth for BFS
        Returns:
            Dictionary mapping entity IDs to search results
        """
        print(f"Running {len(entity_ids)} parallel BFS searches...")
        start_time = time.time()

        # Create tasks for all BFS operations
        tasks = [self.perform_bidirectional_bfs_async(entity_id, max_depth) for entity_id in entity_ids]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Process results
        processed_results = {}
        for result, entity_id in results:
            nodes = []
            while hasattr(result, 'has_next') and result.has_next():
                row = result.get_next()
                nodes.append({
                    'connected_id': row[0],
                    'distance': row[1],
                    'direction': row[2]
                })
            processed_results[entity_id] = nodes
            print(f"BFS from entity {entity_id} found {len(nodes)} connected nodes")

        end_time = time.time()
        print(f"All BFS searches completed in {end_time - start_time:.2f} seconds")
        print(processed_results)
        return processed_results


if __name__ == "__main__":
    # This is the main entry point of the script.
    # It defines and runs an asynchronous function to execute the test scenario.

    async def execute_main_test_scenario():
        """
        Defines and executes the main test scenario for MyGraphService.
        """
        logger.info("Starting main test scenario execution...")
        # Initialize the service. recreate_db=True is often used for clean test runs.
        graph_service_instance = MyGraphService(recreate_db=True)

        # Using 'with' ensures resources are managed correctly (e.g., connections closed),
        # assuming KuzuGraphStore implements __enter__ and __exit__.
        with graph_service_instance as db:
            logger.info("Graph service context entered for testing.")

            try:
                # 1. Get initial database statistics

                # 2. Define parameters for the BFS test
                test_entity_ids = [401]  # Sample entity IDs
                max_depth_for_test = 3  # Max depth for BFS

                logger.info(
                    f"Starting BFS test for entity IDs: {test_entity_ids} with max depth: {max_depth_for_test}.")

                # Perform the multiple BFS searches
                bfs_results = await db.run_multiple_bfs_searches(
                    entity_ids=test_entity_ids,
                    max_depth=max_depth_for_test
                )

                # 3. Log the BFS results
                logger.info("BFS Test Results:")
                if bfs_results:
                    for entity_id, connected_nodes in bfs_results.items():
                        logger.info(f"  Results for starting entity ID {entity_id}:")
                        if connected_nodes:
                            for node_info in connected_nodes:  # Renamed 'node' to 'node_info' for clarity
                                logger.info(
                                    f"    - Connected ID: {node_info['connected_id']}, Distance: {node_info['distance']}, Direction: {node_info['direction']}")
                        else:
                            logger.info(
                                "    - No connected nodes found (or an error occurred for this specific entity).")
                else:
                    logger.info(
                        "  No BFS results returned. This could be due to an empty database, non-existent IDs, or all tasks failing.")

            except Exception as e:
                logger.error(f"An error occurred during the test scenario: {e}", exc_info=True)
            finally:
                logger.info("Exiting graph service context. Test scenario finished.")


    # Run the main asynchronous test scenario
    try:
        asyncio.run(execute_main_test_scenario())
    except KeyboardInterrupt:
        logger.info("Script execution interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"A critical error occurred preventing the asyncio event loop from running: {e}", exc_info=True)