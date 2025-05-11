"""
Initializes the Kuzu Graph Database for BFS.
"""
# External Imports:
import os
import kuzu
import shutil
import logging
from dotenv import load_dotenv

# Internal imports:
from postgres.base import Postgres

# Load environmental variables & Set the logger:
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KuzuGraphStore(Postgres):
    """
    Implements the Kuzu-Graphstore for optimizing Breadth-First Queries on the database.
    Uses Cypher for the queries. Loads only the entity and relationship IDs into the KuzuDB.
    Inherits from Postgres, to connect to the PostgresDB and access its connection details.
    """
    def __init__(self, recreate_db: bool = False):
        """
        Initializes the KuzuGraphStore.
        """
        super().__init__() # Initializes Postgres connection details from the parent class
        self.kuzu_path = os.getenv("KUZU_PATH")
        if not self.kuzu_path:
            raise ValueError("KUZU_PATH environment variable not set. Please set it to the desired Kuzu database directory.")

        self.database = None
        self.connection = None
        self.async_connection = None
        self.recreate_db = recreate_db # Control whether to recreate the DB on connect

        # Create and connect the database:
        self._connect_kuzu()
        self._init_postgres_extension()
        self._init_schema_if_needed()
        if recreate_db:
            self.load_graph_data()

    def _connect_kuzu(self):
        """Establishes KuzuDB connections (both sync and async)."""
        # If recreation is requested, remove the existing database directory
        if self.recreate_db and os.path.exists(self.kuzu_path):
            logger.info(f"Removing existing Kùzu database at {self.kuzu_path} due to recreate_db flag...")
            try:
                shutil.rmtree(self.kuzu_path)
            except OSError as e:
                logger.critical(f"Error removing existing Kùzu database at {self.kuzu_path}: {e}")
                raise

        # Create a connection to the database (Kùzu creates if not exists, connects otherwise)
        try:
            self.database = kuzu.Database(self.kuzu_path)
            self.connection = kuzu.Connection(self.database)

            # Creates async connections:
            self.async_connection = kuzu.AsyncConnection(
                self.database,
                max_concurrent_queries=10,
                max_threads_per_query=4
            )
            logger.info(f"Connected to Kùzu database at {self.kuzu_path}.")
        except Exception as e:
            logger.critical(f"Failed to connect to Kùzu database: {e}")
            self.cleanup()  # Attempt cleanup on connection failure
            raise

    def _init_postgres_extension(self):
        """
        Initializes the PostgreSQL extension within KuzuDB and attaches the PostgreSQL database.
        This allows Kuzu to directly query and copy data from PostgreSQL tables.
        """
        try:
            if self.connection is None:
                raise ConnectionError("Kuzu connection not established.")

            # Install and load PostgreSQL extension inside Kuzu
            self.connection.execute("INSTALL postgres;")
            self.connection.execute("LOAD postgres;")

            # Construct PostgreSQL connection string using inherited details from `Postgres` base class
            conn_string = f"dbname={self.name} user={self.user} host={self.host} password={self.pw} port={self.port}"

            # Attach PostgreSQL database as a foreign data source in Kuzu
            self.connection.execute(f"ATTACH '{conn_string}' AS pg_graph (dbtype postgres);")

            # Set the attached PostgreSQL database as the default for subsequent queries
            self.connection.execute("USE pg_graph;")

            logger.info("PostgreSQL database attached to KuzuDB.")
        except Exception as e:
            logger.critical(f"Error attaching PostgreSQL database to KuzuDB: {e}")
            self.cleanup()
            raise

    def _init_schema_if_needed(self):
        """
        Initializes the Kuzu database schema for 'Entity' nodes and 'Relationship' edges if needed.
        These tables mirror the `Entity` and `Relationship` tables in PostgreSQL based on IDs.
        """
        try:
            if self.connection is None:
                raise ConnectionError("Kuzu connection not established.")

            self.connection.execute(
                """CREATE NODE TABLE IF NOT EXISTS Entity(
                    id INT PRIMARY KEY
                )"""
            )
            self.connection.execute(
                """CREATE REL TABLE IF NOT EXISTS Relationship(
                    FROM Entity TO Entity
                )"""
            )
            logger.info("Kuzu schema initialized successfully.")
        except Exception as e:
            logger.critical(f"Error initializing Kuzu schema: {e}")
            self.cleanup()
            raise

    def load_graph_data(self):
        """
        Loads graph data (Entity IDs and Relationship links) from PostgreSQL tables
        into the KuzuDB using Kuzu's efficient `COPY FROM` statement with PostgreSQL foreign tables.
        """
        logger.info("Loading entities from PostgreSQL to KuzuDB...")
        try:
            if self.connection is None:
                raise ConnectionError("Kuzu connection not established.")

            # Copy entities from the PostgreSQL 'Entity' table into Kuzu's 'Entity' node table.
            # 'RETURN entity_id AS id' maps PostgreSQL's 'entity_id' column to Kuzu's 'id' primary key.
            self.connection.execute("""
                COPY Entity FROM (
                    LOAD FROM Entity
                    RETURN entity_id AS id
                )
            """)
            logger.info("Entities copied successfully.")

            logger.info("Loading relationships from PostgreSQL to KuzuDB...")
            # Copy relationships from the PostgreSQL 'Relationship' table into Kuzu's 'Relationship' edge table.
            # 'RETURN from_entity, to_entity' maps PostgreSQL's foreign keys to Kuzu's relationship endpoints.
            self.connection.execute("""
                COPY Relationship FROM (
                    LOAD FROM Relationship
                    RETURN from_entity, to_entity
                )
            """)
            logger.info("Relationships copied successfully.")
            logger.info("Graph data loading complete!")

        except Exception as e:
            logger.error(f"Error loading graph data into KuzuDB: {e}")
            raise

        # In KuzuGraphStore class:

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        The Kuzu connections are already established in __init__.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context related to this object.
        Ensures database resources are cleaned up.
        """
        self.cleanup()
        # If an exception occurred, you might want to log it or handle it.
        # Returning False (or nothing, which defaults to False) will re-raise the exception
        # if one occurred within the 'with' block.
        # Return True to suppress the exception.
        return False  # Or handle as needed

    def cleanup(self):
        """
        Clean up Kuzu database resources: close connections and dereference the database object.
        This helps in properly releasing database locks.
        """
        logger.info("Cleaning up Kuzu database resources...")
        try:
            if self.async_connection:
                self.async_connection.close()
                self.async_connection = None
            if self.connection:
                self.connection.close()
                self.connection = None
            if self.database:
                # Dereference the database object to allow Kuzu to release file locks
                del self.database
                self.database = None
            logger.info("Kuzu database resources cleaned up.")
        except Exception as e:
            logger.error(f"Error during Kuzu database cleanup: {e}")