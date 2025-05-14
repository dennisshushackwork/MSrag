"""
Graph Database with optimized undirected DFS for external calling using Kuzu's async API.
"""

# External imports:
import time
import asyncio
import logging
from typing import List

# Internal imports:
from graphdatabase.kuzudb import KuzuDB

# Setting up the logger:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DFS(KuzuDB):
    """
    This class implements an undirected DFS graph traversal.
    """
    def __init__(self, create):
        super().__init__(create=create)
        logger.info("Initializing DFS")

    def get_entity_count(self) -> int:
        """Returns the number of entities in the graph."""
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
    def undirected_dfs(start_entity_id: int, depth: int) -> str:
        """
        Creates an undirected BFS query for relationship IDs.
        Args:
            start_entity_id: The ID of the starting entity
            depth: Maximum BFS depth to traverse
        """
        query = f"""
                    MATCH path = (start:Entity {{id: {start_entity_id}}})-[:Relationship*1..{depth}]-(:Entity)
                    UNWIND relationships(path) AS rel
                    RETURN DISTINCT rel.id AS relationship_id
                    """
        return query

    async def perform_undirected_bfs_async(self, start_entity_id: int, max_depth: int = 2):
        """
        Performs an undirected BFS asynchronously.
        """
        query = self.undirected_dfs(start_entity_id, max_depth)
        result = await self.async_connection.execute(query)
        return result, start_entity_id

    async def run_multiple_bfs_searches(self, entity_ids: List[int], max_depth: int = 2):
        """
        Runs multiple undirected BFS searches in parallel.
        """
        start_time = time.time()
        logger.info(f"Running {len(entity_ids)} parallel undirected BFS searches...")
        tasks = [self.perform_undirected_bfs_async(entity_id, max_depth) for entity_id in entity_ids]
        results = await asyncio.gather(*tasks)

        # Process results
        relationship_ids = []
        for result, entity_id in results:
            while hasattr(result, 'has_next') and result.has_next():
                row = result.get_next()
                relationship_ids.append(row[0])
            # Store as a list in the results
            logger.debug(f"Undirected BFS from entity {entity_id} found {len(relationship_ids)} unique relationships")

        # Get total unique relationships across all entity searches
        all_relationships = set(relationship_ids)
        # Turn it back to a list:
        all_relationships = list(all_relationships)
        end_time = time.time()
        logger.info(f"All BFS searches completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Found {len(all_relationships)} unique relationships across all entity searches")
        return all_relationships

    @staticmethod
    async def get_relationships_async(entity_ids: List[int], max_depth: int = 2):
        """
        Async function for FastAPI to get relationships for a list of entities.
        """
        graph_service = DFS(create=False)
        return await graph_service.run_multiple_bfs_searches(entity_ids, max_depth)



















