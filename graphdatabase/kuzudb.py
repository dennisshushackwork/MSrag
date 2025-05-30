"""
Initializes the Kuzu Database.
"""

# External imports:
import os
import kuzu
import shutil
import logging
from dotenv import load_dotenv

# Internal imports:
from postgres.base import Postgres

# Load the environmental variables & initializes the logger:
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KuzuDB(Postgres):

    def __init__(self, create: bool = False):
        super().__init__()
        self.kuzu_path = os.getenv("KUZU_PATH")
        if not self.kuzu_path:
            raise ValueError(
                "KUZU_PATH environment variable not set. Please set it to the desired Kuzu database directory.")

        self.create = create
        self.database = None
        self.connection = None
        self.async_connection = None

        # Create and connect the database:
        self._connect_kuzu()
        if self.create:
            self._init_postgres_extension()
            self._init_schema_if_needed()
            self.load_graph_data()

    def _connect_kuzu(self):
        """Establishes KuzuDB connections (both sync and async)."""
        # If recreation is requested, remove the existing database directory
        if self.create and os.path.exists(self.kuzu_path):
            logger.info(f"Removing existing Kùzu database at {self.kuzu_path} due to create flag...")
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
                    id INT64 PRIMARY KEY

                )"""
            )
            self.connection.execute(
                """CREATE REL TABLE IF NOT EXISTS Relationship(
                    FROM Entity TO Entity,
                    id INT64 PRIMARY KEY
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

            self.connection.execute("""
                COPY Entity FROM (
                    LOAD FROM Entity
                    RETURN entity_id AS id
                )
            """)
            logger.info("Entities copied successfully.")
            logger.info("Loading relationships from PostgreSQL to KuzuDB...")
            self.connection.execute("""
                COPY Relationship FROM (
                    LOAD FROM Relationship
                    RETURN from_entity, to_entity, rel_id AS id
                )
            """)
            logger.info("Relationships copied successfully.")
            logger.info("Graph data loading complete!")
        except Exception as e:
            logger.error(f"Error loading graph data into KuzuDB: {e}")
            raise



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
        return False

    def cleanup(self):
        """
        Clean up Kuzu database resources: close connections and dereference the database object.
        This helps in properly releasing database locks.j
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

    def close_connection(self):
        """Closes the synchronous and asynchronous connections to the Kuzu database."""
        logger.info("Closing Kuzu database resources...")
        try:
            if self.async_connection:
                self.async_connection.close()
                self.async_connection = None
            if self.connection:
                self.connection.close()
                self.connection = None
            logger.info("Kuzu database resources closed.")
        except Exception as e:
            logger.error(f"Error during Kuzu database connection closing: {e}")

















