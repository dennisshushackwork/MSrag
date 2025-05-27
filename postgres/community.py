"""
This class is specifically designed for the community detection algorithm.
"""
# External imports
import logging
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from psycopg2.extras import execute_values, DictCursor

# Internal imports:
from postgres.base import Postgres

# Load environmental variables & logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommunityQueries(Postgres):
    def __init__(self):
        super().__init__()

    # --------------------- Loading the Graph ----------------------------- #
    def load_entities_in_batches(self, batch_size: int, offset: int) -> List[Dict]:
        """Returns all the entities in batches"""
        query = """
                SELECT
                    entity_id
                FROM 
                    ENTITY
                ORDER BY entity_id
                    LIMIT %s OFFSET %s
                """
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (batch_size, offset))
            entities = cur.fetchall()
            # Reduced verbosity for this log
            # logger.info(f"Loaded {len(entities)} entities (batch size: {batch_size}, offset: {offset})")
            return entities

    def load_relationships_in_batches(self, batch_size: int, offset: int) -> List[Dict]:
        """Load relationships in batches for graph construction"""
        query = """
                SELECT
                    from_entity,
                    to_entity,
                    rel_id,
                    rel_weight
                FROM
                    Relationship
                ORDER BY rel_id
                LIMIT %s OFFSET %s;
                """
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (batch_size, offset))
            relationships = cur.fetchall()
            # Reduced verbosity for this log
            # logger.info(f"Loaded {len(relationships)} relationships (batch size: {batch_size}, offset: {offset})")
            return relationships

    # --------------------- Inserting the data after Community Detection ----------------------------- #
    def insert_community_groups(self, values):
        """
        Inserts the detected communities within the postgres database
        and returns the newly generated community_ids.
        """
        query = """
                    INSERT INTO CommunityGroup(
                        community_level,
                        community_parent,
                        community_num_nodes,
                        community_embed
                    )
                    VALUES %s
                    RETURNING community_id; 
                """
        generated_ids = []
        with self.conn.cursor() as cur:
            execute_values(cur, query, values, template=None, page_size=100)
            generated_ids = [row[0] for row in cur.fetchall()]
            self.conn.commit()
        logger.info(f"Communities saved: {len(values)}. New DB IDs generated: {generated_ids}")
        return generated_ids

    def batch_update_community_parents(self, parent_updates: List[Tuple[Optional[int], int]]):
        """
        Batch updates the community_parent for given community_ids.
        """
        if not parent_updates:
            logger.info("No community parent updates to perform.")
            return

        valid_updates = [item for item in parent_updates if item[0] is not None]

        if not valid_updates:
            logger.info("No valid parent IDs (non-NULL) to update.")
            return

        query = """
                UPDATE CommunityGroup
                SET community_parent = %s
                WHERE community_id = %s;
                """
        with self.conn.cursor() as cur:
            try:
                # Use executemany for batch updates
                cur.executemany(query, valid_updates)
                self.conn.commit()
                logger.info(f"Successfully updated parent_community_id for {len(valid_updates)} communities.")
            except Exception as e:
                # Log the specific error from psycopg2 if possible
                logger.error(f"Error during batch update of community parents: {e}")
                self.conn.rollback()

    def insert_community_nodes(self, values):
        """
        Inserts community-entity associations.
        """
        query = """
        INSERT INTO CommunityNode (community_id, entity_id)
        VALUES %s;
        """
        with self.conn.cursor() as cur:
            execute_values(cur, query, values, page_size=1000)
            self.conn.commit()

    def create_community_document_associations(self, community_id: int):
        """
        Creates the document-community summary associations for given community_id.
        """
        query = """
                INSERT INTO CommunityDocument (community_id, document_id)
                SELECT DISTINCT %s, ED.document_id
                FROM CommunityNode CN
                JOIN EntityDocument ED ON CN.entity_id = ED.entity_id
                WHERE CN.community_id = %s
                ON CONFLICT (community_id, document_id) DO NOTHING;
                """
        with self.conn.cursor() as cur:
            cur.execute(query, (community_id, community_id))
            rows_affected = cur.rowcount
            self.conn.commit()
            # logger.info(f"Created/verified {rows_affected} community-document associations for community {community_id}")

    def create_community_chunk_associations(self, community_id: int):
        """
        Creates an association between the communitygroup and the chunk.
        """
        query = """
            INSERT INTO CommunityChunk (community_id, chunk_id)
            SELECT DISTINCT %s, EC.chunk_id
            FROM CommunityNode CN
            JOIN EntityChunk EC ON CN.entity_id = EC.entity_id
            WHERE CN.community_id = %s
            ON CONFLICT (community_id, chunk_id) DO NOTHING;
            """
        with self.conn.cursor() as cur:
            cur.execute(query, (community_id, community_id))
            rows_affected = cur.rowcount
            self.conn.commit()
            # logger.info(f"Created/verified {rows_affected} community-chunk associations for community {community_id}")

    # ----------------- SQL Queries for community summarisation -------------#
    def get_all_community_ids_for_summarization(self, batch_size: int, offset: int) -> List[int]:
        """
        Gets the community ids in batches to create community Summaries.
        """
        query = """
                SELECT community_id
                FROM CommunityGroup
                ORDER BY community_id
                LIMIT %s OFFSET %s;
                """
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (batch_size, offset))
            community_ids = [row['community_id'] for row in cur.fetchall()]
        # logger.info(f"Fetched {len(community_ids)} community_ids for summarization (batch: {batch_size}, offset: {offset})")
        return community_ids

    def get_internal_relationships_for_communities(self, community_ids: List[int]) -> Dict[int, List[Dict]]:
        """
        Gets the relationships within the communities for summarisation
        """
        if not community_ids:
            return {}
        query = """
            SELECT
                cn1.community_id,
                r.rel_description,
                r.rel_tokens,
                e_from.entity_name as from_entity_name,
                e_to.entity_name as to_entity_name
            FROM
                Relationship r
            JOIN
                CommunityNode cn1 ON r.from_entity = cn1.entity_id
            JOIN
                CommunityNode cn2 ON r.to_entity = cn2.entity_id
            JOIN
                Entity e_from ON r.from_entity = e_from.entity_id
            JOIN
                Entity e_to ON r.to_entity = e_to.entity_id
            WHERE
                cn1.community_id = cn2.community_id AND
                cn1.community_id = ANY(%s);
        """
        rels_by_community: Dict[int, List[Dict]] = {cid: [] for cid in community_ids}
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (community_ids,))
            rows = cur.fetchall()
            for row in rows:
                original_rel_description = row['rel_description'] if row['rel_description'] is not None else "No description"
                rel_tokens = row['rel_tokens'] if row['rel_tokens'] is not None else 0
                formatted_description = f"{row['from_entity_name']} -> {row['to_entity_name']}: {original_rel_description}"
                rels_by_community[row['community_id']].append({
                    'description': formatted_description,
                    'tokens': rel_tokens
                })
        # logger.info(f"Fetched internal relationships for {len(community_ids)} communities. Found relationships for {len([k for k, v in rels_by_community.items() if v])} of them.")
        return rels_by_community

    def batch_update_community_summaries(self, summaries_data: List[Tuple[str, int]]):
        """
        Batch updates the community_summary for given community_ids using executemany.
        summaries_data is a list of tuples: (community_summary_text, community_id)
        """
        if not summaries_data:
            logger.info("No summaries to update.")
            return

        query = """
                UPDATE CommunityGroup
                SET community_summary = %s
                WHERE community_id = %s;
                """
        with self.conn.cursor() as cur:
            try:
                # Use executemany for batch updates
                cur.executemany(query, summaries_data)
                self.conn.commit()
                logger.info(f"Successfully updated summaries for {len(summaries_data)} communities.")
            except Exception as e:
                 # Log the specific error from psycopg2 if possible
                logger.error(f"Error during batch update of community summaries: {e}")
                self.conn.rollback()

    """
    This class is specifically designed for the community detection algorithm.
    """

    # External imports
    import logging
    from typing import List, Dict, Tuple, Optional
    from dotenv import load_dotenv
    from psycopg2.extras import execute_values, DictCursor

    # Internal imports:
    from postgres.base import Postgres

    # Load environmental variables & logging
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    class CommunityQueries(Postgres):
        def __init__(self):
            super().__init__()

        # --------------------- Loading the Graph ----------------------------- #
        def load_entities_in_batches(self, batch_size: int, offset: int) -> List[Dict]:
            """Returns all the entities in batches"""
            query = """
                    SELECT
                        entity_id
                    FROM 
                        ENTITY
                    ORDER BY entity_id
                        LIMIT %s OFFSET %s
                    """
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, (batch_size, offset))
                entities = cur.fetchall()
                # Reduced verbosity for this log
                # logger.info(f"Loaded {len(entities)} entities (batch size: {batch_size}, offset: {offset})")
                return entities

        def load_relationships_in_batches(self, batch_size: int, offset: int) -> List[Dict]:
            """Load relationships in batches for graph construction"""
            query = """
                    SELECT
                        from_entity,
                        to_entity,
                        rel_id,
                        rel_weight
                    FROM
                        Relationship
                    ORDER BY rel_id
                    LIMIT %s OFFSET %s;
                    """
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, (batch_size, offset))
                relationships = cur.fetchall()
                # Reduced verbosity for this log
                # logger.info(f"Loaded {len(relationships)} relationships (batch size: {batch_size}, offset: {offset})")
                return relationships

        # --------------------- Inserting the data after Community Detection ----------------------------- #
        def insert_community_groups(self, values):
            """
            Inserts the detected communities within the postgres database
            and returns the newly generated community_ids.
            """
            query = """
                        INSERT INTO CommunityGroup(
                            community_level,
                            community_parent,
                            community_num_nodes,
                            community_embed
                        )
                        VALUES %s
                        RETURNING community_id; 
                    """
            generated_ids = []
            with self.conn.cursor() as cur:
                execute_values(cur, query, values, template=None, page_size=100)
                generated_ids = [row[0] for row in cur.fetchall()]
                self.conn.commit()
            logger.info(f"Communities saved: {len(values)}. New DB IDs generated: {generated_ids}")
            return generated_ids

        def batch_update_community_parents(self, parent_updates: List[Tuple[Optional[int], int]]):
            """
            Batch updates the community_parent for given community_ids.
            """
            if not parent_updates:
                logger.info("No community parent updates to perform.")
                return

            valid_updates = [item for item in parent_updates if item[0] is not None]

            if not valid_updates:
                logger.info("No valid parent IDs (non-NULL) to update.")
                return

            query = """
                    UPDATE CommunityGroup
                    SET community_parent = %s
                    WHERE community_id = %s;
                    """
            with self.conn.cursor() as cur:
                try:
                    # Use executemany for batch updates
                    cur.executemany(query, valid_updates)
                    self.conn.commit()
                    logger.info(f"Successfully updated parent_community_id for {len(valid_updates)} communities.")
                except Exception as e:
                    # Log the specific error from psycopg2 if possible
                    logger.error(f"Error during batch update of community parents: {e}")
                    self.conn.rollback()

        def insert_community_nodes(self, values):
            """
            Inserts community-entity associations.
            """
            query = """
            INSERT INTO CommunityNode (community_id, entity_id)
            VALUES %s;
            """
            with self.conn.cursor() as cur:
                execute_values(cur, query, values, page_size=1000)
                self.conn.commit()

        def create_community_document_associations(self, community_id: int):
            """
            Creates the document-community summary associations for given community_id.
            """
            query = """
                    INSERT INTO CommunityDocument (community_id, document_id)
                    SELECT DISTINCT %s, ED.document_id
                    FROM CommunityNode CN
                    JOIN EntityDocument ED ON CN.entity_id = ED.entity_id
                    WHERE CN.community_id = %s
                    ON CONFLICT (community_id, document_id) DO NOTHING;
                    """
            with self.conn.cursor() as cur:
                cur.execute(query, (community_id, community_id))
                rows_affected = cur.rowcount
                self.conn.commit()
                # logger.info(f"Created/verified {rows_affected} community-document associations for community {community_id}")

        def create_community_chunk_associations(self, community_id: int):
            """
            Creates an association between the communitygroup and the chunk.
            """
            query = """
                INSERT INTO CommunityChunk (community_id, chunk_id)
                SELECT DISTINCT %s, EC.chunk_id
                FROM CommunityNode CN
                JOIN EntityChunk EC ON CN.entity_id = EC.entity_id
                WHERE CN.community_id = %s
                ON CONFLICT (community_id, chunk_id) DO NOTHING;
                """
            with self.conn.cursor() as cur:
                cur.execute(query, (community_id, community_id))
                rows_affected = cur.rowcount
                self.conn.commit()
                # logger.info(f"Created/verified {rows_affected} community-chunk associations for community {community_id}")

        # ----------------- SQL Queries for community summarisation -------------#
        def get_all_community_ids_for_summarization(self, batch_size: int, offset: int) -> List[int]:
            """
            Gets the community ids in batches to create community Summaries.
            """
            query = """
                    SELECT community_id
                    FROM CommunityGroup
                    ORDER BY community_id
                    LIMIT %s OFFSET %s;
                    """
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, (batch_size, offset))
                community_ids = [row['community_id'] for row in cur.fetchall()]
            # logger.info(f"Fetched {len(community_ids)} community_ids for summarization (batch: {batch_size}, offset: {offset})")
            return community_ids

        def get_internal_relationships_for_communities(self, community_ids: List[int]) -> Dict[int, List[Dict]]:
            """
            Gets the relationships within the communities for summarisation
            """
            if not community_ids:
                return {}
            query = """
                SELECT
                    cn1.community_id,
                    r.rel_description,
                    r.rel_tokens,
                    e_from.entity_name as from_entity_name,
                    e_to.entity_name as to_entity_name
                FROM
                    Relationship r
                JOIN
                    CommunityNode cn1 ON r.from_entity = cn1.entity_id
                JOIN
                    CommunityNode cn2 ON r.to_entity = cn2.entity_id
                JOIN
                    Entity e_from ON r.from_entity = e_from.entity_id
                JOIN
                    Entity e_to ON r.to_entity = e_to.entity_id
                WHERE
                    cn1.community_id = cn2.community_id AND
                    cn1.community_id = ANY(%s);
            """
            rels_by_community: Dict[int, List[Dict]] = {cid: [] for cid in community_ids}
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, (community_ids,))
                rows = cur.fetchall()
                for row in rows:
                    original_rel_description = row['rel_description'] if row[
                                                                             'rel_description'] is not None else "No description"
                    rel_tokens = row['rel_tokens'] if row['rel_tokens'] is not None else 0
                    formatted_description = f"{row['from_entity_name']} -> {row['to_entity_name']}: {original_rel_description}"
                    rels_by_community[row['community_id']].append({
                        'description': formatted_description,
                        'tokens': rel_tokens
                    })
            # logger.info(f"Fetched internal relationships for {len(community_ids)} communities. Found relationships for {len([k for k, v in rels_by_community.items() if v])} of them.")
            return rels_by_community

        def batch_update_community_summaries(self, summaries_data: List[Tuple[str, int]]):
            """
            Batch updates the community_summary for given community_ids using executemany.
            summaries_data is a list of tuples: (community_summary_text, community_id)
            """
            if not summaries_data:
                logger.info("No summaries to update.")
                return

            query = """
                    UPDATE CommunityGroup
                    SET community_summary = %s
                    WHERE community_id = %s;
                    """
            with self.conn.cursor() as cur:
                try:
                    # Use executemany for batch updates
                    cur.executemany(query, summaries_data)
                    self.conn.commit()
                    logger.info(f"Successfully updated summaries for {len(summaries_data)} communities.")
                except Exception as e:
                    # Log the specific error from psycopg2 if possible
                    logger.error(f"Error during batch update of community summaries: {e}")
                    self.conn.rollback()

        def delete_failed_communities(self, community_ids_to_delete: List[int]):
            """
            Deletes communities that failed summarization from the database.
            This includes cascading deletes for related records.
            """
            if not community_ids_to_delete:
                logger.info("No failed communities to delete.")
                return

            # Delete community nodes
            delete_nodes_query = """
                           DELETE FROM CommunityNode 
                           WHERE community_id = ANY(%s);
                       """

            # Delete community documents
            delete_docs_query = """
                           DELETE FROM CommunityDocument 
                           WHERE community_id = ANY(%s);
                       """

            # Delete community chunks
            delete_chunks_query = """
                           DELETE FROM CommunityChunk 
                           WHERE community_id = ANY(%s);
                       """

            # Delete community groups (main table)
            delete_groups_query = """
                           DELETE FROM CommunityGroup 
                           WHERE community_id = ANY(%s);
                       """

            with self.conn.cursor() as cur:
                try:
                    # Execute deletions in order
                    cur.execute(delete_nodes_query, (community_ids_to_delete,))
                    nodes_deleted = cur.rowcount
                    logger.info(f"Deleted {nodes_deleted} community node associations")

                    cur.execute(delete_docs_query, (community_ids_to_delete,))
                    docs_deleted = cur.rowcount
                    logger.info(f"Deleted {docs_deleted} community document associations")

                    cur.execute(delete_chunks_query, (community_ids_to_delete,))
                    chunks_deleted = cur.rowcount
                    logger.info(f"Deleted {chunks_deleted} community chunk associations")

                    cur.execute(delete_groups_query, (community_ids_to_delete,))
                    groups_deleted = cur.rowcount
                    logger.info(f"Deleted {groups_deleted} community groups")

                    self.conn.commit()
                    logger.info(
                        f"Successfully deleted all records for {len(community_ids_to_delete)} failed communities")

                except Exception as e:
                    logger.error(f"Error deleting failed communities: {e}")
                    self.conn.rollback()
                    raise


