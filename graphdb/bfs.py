"""
Graph Database, which does BFS.
"""
# External imports:
import time
import asyncio
import logging
from typing import List

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

    def perform_bfs(self, start_entities: List[str], depth: int):
        """Performs Breadth First Search (BFS) on the starting entities up to depth d"""
        pass


if __name__ == "__main__":
    graph = MyGraphService(recreate_db=True)
    with graph as db:
        count = db.get_entity_count()
        count_1 = db.get_relationship_count()
        print(count)
        print(count_1)

