"""
Entity Resolution System for GraphRAG
This module handles the complete entity resolution workflow including:
1. Similarity calculation between entities
2. Graph-based grouping using connected components
3. Levenshtein distance filtering
4. LLM-based merge decisions
5. Database updates with proper constraint handling
"""

# External imports
import logging
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import networkx as nx
from Levenshtein import distance as levenshtein_distance
from itertools import combinations
import json
from tqdm import tqdm
from dotenv import load_dotenv
from psycopg2.extras import execute_values, DictCursor
import psycopg2

# Internal imports
from postgres.base import Postgres
from postgres.queries import CommunityQueries

# Load environmental variables & logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EntityResolutionQueries(CommunityQueries):
    """Extended queries for entity resolution operations"""

    def __init__(self):
        super().__init__()
        self.min_similarity = 0.85  # Store min_similarity here for queries
    """Extended queries for entity resolution operations"""

    def __init__(self):
        super().__init__()

    def get_entity_id_batch(self, offset: int, limit: int) -> List[int]:
        """Get a batch of entity IDs that have embeddings"""
        query = """
        SELECT entity_id
        FROM Entity
        WHERE entity_emb IS NOT NULL
        ORDER BY entity_id
        LIMIT %s OFFSET %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (limit, offset))
            return [row[0] for row in cur.fetchall()]

    def insert_entity_similarities_batch(self, similarities: List[Dict]) -> None:
        """Insert batch of entity similarities from database query results"""
        if not similarities:
            return

        # Convert to tuples and add both directions
        similarity_tuples = []
        for sim in similarities:
            similarity_tuples.append((
                sim['entity_id_1'],
                sim['entity_id_2'],
                sim['cosine_similarity']
            ))
            similarity_tuples.append((
                sim['entity_id_2'],
                sim['entity_id_1'],
                sim['cosine_similarity']
            ))

        query = """
        INSERT INTO EntitySimilarity (from_entity, to_entity, similarity)
        VALUES %s
        ON CONFLICT (from_entity, to_entity) 
        DO UPDATE SET similarity = EXCLUDED.similarity
        """
        with self.conn.cursor() as cur:
            execute_values(cur, query, similarity_tuples)
        self.conn.commit()

    def get_similar_entities(self, min_similarity: float = 0.85) -> List[Dict]:
        """Get all entity pairs with similarity above threshold"""
        query = """
        SELECT es.from_entity, es.to_entity, es.similarity,
               e1.entity_name as from_name, e2.entity_name as to_name
        FROM EntitySimilarity es
        JOIN Entity e1 ON es.from_entity = e1.entity_id
        JOIN Entity e2 ON es.to_entity = e2.entity_id
        WHERE es.similarity >= %s
        ORDER BY es.similarity DESC
        """
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (min_similarity,))
            return cur.fetchall()

    def get_entity_relationships(self, entity_id: int) -> List[Dict]:
        """Get all relationships for an entity"""
        query = """
        SELECT rel_id, from_entity, to_entity, rel_description, rel_chunk_id
        FROM Relationship
        WHERE from_entity = %s OR to_entity = %s
        """
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (entity_id, entity_id))
            return cur.fetchall()

    def get_entity_associations(self, entity_id: int) -> Dict[str, List[int]]:
        """Get all associations (chunks, documents) for an entity"""
        associations = {}

        # Get chunks
        query = "SELECT chunk_id FROM EntityChunk WHERE entity_id = %s"
        with self.conn.cursor() as cur:
            cur.execute(query, (entity_id,))
            associations['chunks'] = [row[0] for row in cur.fetchall()]

        # Get documents
        query = "SELECT document_id FROM EntityDocument WHERE entity_id = %s"
        with self.conn.cursor() as cur:
            cur.execute(query, (entity_id,))
            associations['documents'] = [row[0] for row in cur.fetchall()]

        return associations

    def merge_entities(self, keep_entity_id: int, merge_entity_ids: List[int],
                      new_name: Optional[str] = None) -> None:
        """
        Merge multiple entities into one, handling all constraints
        """
        try:
            with self.conn.cursor() as cur:
                # Start transaction
                cur.execute("BEGIN")

                # Update entity name if provided
                if new_name:
                    cur.execute(
                        "UPDATE Entity SET entity_name = %s WHERE entity_id = %s",
                        (new_name, keep_entity_id)
                    )

                # Update relationships
                for merge_id in merge_entity_ids:
                    # Update from_entity references
                    cur.execute("""
                        UPDATE Relationship 
                        SET from_entity = %s 
                        WHERE from_entity = %s AND to_entity != %s
                    """, (keep_entity_id, merge_id, keep_entity_id))

                    # Update to_entity references
                    cur.execute("""
                        UPDATE Relationship 
                        SET to_entity = %s 
                        WHERE to_entity = %s AND from_entity != %s
                    """, (keep_entity_id, merge_id, keep_entity_id))

                    # Delete self-referential relationships that might have been created
                    cur.execute("""
                        DELETE FROM Relationship 
                        WHERE from_entity = %s AND to_entity = %s
                    """, (keep_entity_id, keep_entity_id))

                # Merge EntityChunk associations
                for merge_id in merge_entity_ids:
                    cur.execute("""
                        INSERT INTO EntityChunk (entity_id, chunk_id)
                        SELECT %s, chunk_id 
                        FROM EntityChunk 
                        WHERE entity_id = %s
                        ON CONFLICT (entity_id, chunk_id) DO NOTHING
                    """, (keep_entity_id, merge_id))

                # Merge EntityDocument associations
                for merge_id in merge_entity_ids:
                    cur.execute("""
                        INSERT INTO EntityDocument (entity_id, document_id)
                        SELECT %s, document_id 
                        FROM EntityDocument 
                        WHERE entity_id = %s
                        ON CONFLICT (entity_id, document_id) DO NOTHING
                    """, (keep_entity_id, merge_id))

                # Delete associations for merged entities
                cur.execute(
                    "DELETE FROM EntityChunk WHERE entity_id = ANY(%s)",
                    (merge_entity_ids,)
                )
                cur.execute(
                    "DELETE FROM EntityDocument WHERE entity_id = ANY(%s)",
                    (merge_entity_ids,)
                )

                # Delete from CommunityNode if exists
                cur.execute(
                    "DELETE FROM CommunityNode WHERE entity_id = ANY(%s)",
                    (merge_entity_ids,)
                )

                # Delete from EntitySimilarity
                cur.execute(
                    "DELETE FROM EntitySimilarity WHERE from_entity = ANY(%s) OR to_entity = ANY(%s)",
                    (merge_entity_ids, merge_entity_ids)
                )

                # Finally, delete the merged entities
                cur.execute(
                    "DELETE FROM Entity WHERE entity_id = ANY(%s)",
                    (merge_entity_ids,)
                )

                # Commit transaction
                cur.execute("COMMIT")

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error merging entities: {e}")
            raise

    def clear_entity_similarities(self) -> None:
        """Clear all entity similarities for fresh calculation"""
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE EntitySimilarity")
        self.conn.commit()


class EntityResolution:
    """
    Main class for entity resolution workflow
    """

    def __init__(self, llm_client=None, batch_size: int = 20000):
        self.db = EntityResolutionQueries()
        self.llm_client = llm_client  # LLM client for merge decisions
        self.batch_size = batch_size
        self.min_similarity = 0.85
        self.max_group_size = 100
        self.levenshtein_threshold = 0.7  # Normalized threshold

        # Pass min_similarity to db queries
        self.db.min_similarity = self.min_similarity

    def calculate_all_similarities(self) -> None:
        """
        Calculate cosine similarities between all entities in batches using database's DiskANN index
        """
        logger.info("Starting entity similarity calculation using DiskANN index...")

        # Clear existing similarities
        self.db.clear_entity_similarities()

        # Get total entity count
        total_entities = self.db.count_entities()
        logger.info(f"Total entities to process: {total_entities}")

        # Process in batches
        for offset in tqdm(range(0, total_entities, self.batch_size), desc="Processing batches"):
            # Get batch of entity IDs
            entity_ids = self.db.get_entity_id_batch(offset, self.batch_size)

            if not entity_ids:
                continue

            # Calculate similarities using database query
            similarities = self.db.calculate_similarities_for_batch(entity_ids)

            # Insert similarities (both directions)
            if similarities:
                self.db.insert_entity_similarities_batch(similarities)

            logger.info(f"Processed {min(offset + self.batch_size, total_entities)}/{total_entities} entities")

    def find_entity_groups(self) -> List[Set[int]]:
        """
        Use graph-based approach to find connected components of similar entities
        """
        logger.info("Finding entity groups using connected components...")

        # Get all similar entity pairs
        similar_pairs = self.db.get_similar_entities(self.min_similarity)

        # Build graph
        G = nx.Graph()
        for pair in similar_pairs:
            G.add_edge(pair['from_entity'], pair['to_entity'],
                      weight=pair['similarity'],
                      from_name=pair['from_name'],
                      to_name=pair['to_name'])

        # Find connected components
        components = list(nx.connected_components(G))
        logger.info(f"Found {len(components)} connected components")

        # Filter by size
        valid_components = [comp for comp in components if 2 <= len(comp) <= self.max_group_size]
        logger.info(f"Valid components (2-{self.max_group_size} entities): {len(valid_components)}")

        return valid_components

    def filter_by_levenshtein(self, entity_group: Set[int]) -> List[Set[int]]:
        """
        Filter entity groups by Levenshtein distance, potentially splitting groups
        """
        # Get entity names
        entity_names = {}
        for entity_id in entity_group:
            with self.db.conn.cursor() as cur:
                cur.execute("SELECT entity_name FROM Entity WHERE entity_id = %s", (entity_id,))
                result = cur.fetchone()
                if result:
                    entity_names[entity_id] = result[0]

        # Build similarity graph based on Levenshtein distance
        G = nx.Graph()
        for id1, id2 in combinations(entity_group, 2):
            name1 = entity_names.get(id1, "")
            name2 = entity_names.get(id2, "")

            # Normalized Levenshtein distance
            max_len = max(len(name1), len(name2))
            if max_len > 0:
                normalized_distance = levenshtein_distance(name1, name2) / max_len
                similarity = 1 - normalized_distance

                if similarity >= self.levenshtein_threshold:
                    G.add_edge(id1, id2, weight=similarity)

        # Find connected components in the filtered graph
        return list(nx.connected_components(G))

    def decide_merge_with_llm(self, entities: List[Dict]) -> Dict:
        """
        Use LLM to decide if entities should be merged
        """
        if not self.llm_client:
            # Fallback to rule-based decision
            return self._rule_based_merge_decision(entities)

        # Prepare prompt
        entity_info = "\n".join([f"- {e['entity_name']} (ID: {e['entity_id']})" for e in entities])

        prompt = f"""
        Please analyze if the following entities refer to the same real-world entity and should be merged:
        
        {entity_info}
        
        Consider variations in naming, abbreviations, and context.
        
        Respond with JSON:
        {{
            "should_merge": true/false,
            "reason": "explanation",
            "preferred_name": "best name if merging"
        }}
        """

        # This is a placeholder - implement based on your LLM client
        # response = self.llm_client.complete(prompt)
        # return json.loads(response)

        return self._rule_based_merge_decision(entities)

    def _rule_based_merge_decision(self, entities: List[Dict]) -> Dict:
        """
        Fallback rule-based merge decision
        """
        names = [e['entity_name'] for e in entities]

        # Simple heuristic: merge if names are very similar
        should_merge = True
        for name1, name2 in combinations(names, 2):
            similarity = 1 - (levenshtein_distance(name1, name2) / max(len(name1), len(name2)))
            if similarity < 0.8:
                should_merge = False
                break

        # Choose the longest name as preferred
        preferred_name = max(names, key=len) if should_merge else None

        return {
            "should_merge": should_merge,
            "reason": "Based on name similarity",
            "preferred_name": preferred_name
        }

    def execute_resolution(self) -> Dict[str, int]:
        """
        Execute the complete entity resolution workflow
        """
        stats = {
            "groups_found": 0,
            "groups_processed": 0,
            "entities_merged": 0,
            "merges_completed": 0
        }

        try:
            # Step 1: Calculate similarities
            self.calculate_all_similarities()

            # Step 2: Find entity groups
            entity_groups = self.find_entity_groups()
            stats["groups_found"] = len(entity_groups)

            # Step 3-5: Process each group
            for group in tqdm(entity_groups, desc="Processing entity groups"):
                # Step 3: LeFevenshtein filtering
                filtered_groups = self.filter_by_levenshtein(group)

                for filtered_group in filtered_groups:
                    if len(filtered_group) < 2:
                        continue

                    stats["groups_processed"] += 1

                    # Get entity details
                    entities = []
                    for entity_id in filtered_group:
                        with self.db.conn.cursor(cursor_factory=DictCursor) as cur:
                            cur.execute(
                                "SELECT entity_id, entity_name FROM Entity WHERE entity_id = %s",
                                (entity_id,)
                            )
                            entity = cur.fetchone()
                            if entity:
                                entities.append(dict(entity))

                    # Step 4: LLM decision
                    merge_decision = self.decide_merge_with_llm(entities)

                    if merge_decision["should_merge"]:
                        # Step 5: Execute merge
                        entity_ids = [e['entity_id'] for e in entities]
                        keep_id = entity_ids[0]  # Keep the first entity
                        merge_ids = entity_ids[1:]  # Merge others into it

                        self.db.merge_entities(
                            keep_entity_id=keep_id,
                            merge_entity_ids=merge_ids,
                            new_name=merge_decision.get("preferred_name")
                        )

                        stats["entities_merged"] += len(merge_ids)
                        stats["merges_completed"] += 1

                        logger.info(f"Merged {len(merge_ids)} entities into entity {keep_id}")

            logger.info(f"Entity resolution completed. Stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error in entity resolution: {e}")
            raise

    def validate_resolution(self) -> Dict[str, Any]:
        """
        Validate the entity resolution results
        """
        validation_results = {}

        with self.db.conn.cursor() as cur:
            # Check for orphaned relationships
            cur.execute("""
                SELECT COUNT(*) FROM Relationship r
                WHERE NOT EXISTS (SELECT 1 FROM Entity e WHERE e.entity_id = r.from_entity)
                   OR NOT EXISTS (SELECT 1 FROM Entity e WHERE e.entity_id = r.to_entity)
            """)
            validation_results["orphaned_relationships"] = cur.fetchone()[0]

            # Check for self-referential relationships
            cur.execute("""
                SELECT COUNT(*) FROM Relationship
                WHERE from_entity = to_entity
            """)
            validation_results["self_referential_relationships"] = cur.fetchone()[0]

            # Check entity counts
            cur.execute("SELECT COUNT(*) FROM Entity")
            validation_results["total_entities"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM EntitySimilarity")
            validation_results["similarity_pairs"] = cur.fetchone()[0]

        return validation_results


# Example usage
if __name__ == "__main__":
    # Initialize the entity resolution system
    resolver = EntityResolution(llm_client=None, batch_size=20000)

    # Execute resolution
    stats = resolver.execute_resolution()
    print(f"Resolution completed: {stats}")

    # Validate results
    validation = resolver.validate_resolution()
    print(f"Validation results: {validation}")