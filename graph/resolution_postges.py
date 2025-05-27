import numpy as np
import psycopg2
import os
import io
import logging
import sys
import struct  # For interpreting header bytes
from dotenv import load_dotenv

# Assuming your Postgres class is in a directory 'postgres' and file 'base.py'
# If 'postgres' is a top-level directory in your project structure:
from postgres.base import Postgres

# If 'base.py' (containing Postgres class) is in the same directory as this script:
# from base import Postgres

# --- Basic Logging Setup ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
load_dotenv()
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_FILE_PATH = "/Users/dennis/Documents/GitHub/MSrag/wiki_all_1M_data/base.1M.fbin"  # User's specific path
EMBEDDING_DIMENSION = 768
COPY_BATCH_SIZE = 10000


# --- Helper function to read .fbin ---
def read_fbin_vectors(filename, expected_dim, dtype=np.float32, header_bytes=0):
    logger.debug(f"Attempting to read .fbin file: {filename} with header_bytes={header_bytes}")
    try:
        if not os.path.exists(filename):
            logger.error(f"File not found at path: {filename}")
            return None

        total_file_bytes = os.path.getsize(filename)
        logger.debug(f"Total file size: {total_file_bytes} bytes.")

        if header_bytes > 0 and total_file_bytes >= header_bytes:
            with open(filename, 'rb') as f:
                header_content = f.read(header_bytes)
                logger.info(f"Read header ({header_bytes} bytes): {header_content.hex()}")
                if header_bytes == 8:
                    try:
                        num_vectors_from_header, dim_from_header = struct.unpack('<ii', header_content)
                        logger.info(f"Interpreted header as two little-endian int32s: "
                                    f"num_vectors_hdr={num_vectors_from_header}, dim_hdr={dim_from_header}")
                        if dim_from_header != expected_dim:
                            logger.warning(
                                f"Header dimension ({dim_from_header}) does not match expected dimension ({expected_dim})!")
                    except struct.error as se:
                        logger.warning(f"Could not interpret header as two int32s: {se}")
        elif header_bytes > 0 and total_file_bytes < header_bytes:
            logger.error(
                f"File size ({total_file_bytes}) is smaller than header_bytes ({header_bytes}). Cannot read header.")
            return None

        data_bytes = total_file_bytes - header_bytes
        logger.debug(f"Effective data bytes (after considering header): {data_bytes} bytes.")

        bytes_per_element = np.dtype(dtype).itemsize
        bytes_per_vector = expected_dim * bytes_per_element

        if data_bytes == 0:
            logger.warning(
                f"No vector data found after accounting for header (or file was empty/just header). Data bytes: {data_bytes}")
            return np.array([]).reshape(0, expected_dim)

        if bytes_per_vector == 0:
            logger.error("bytes_per_vector is zero (expected_dim might be zero). Cannot proceed.")
            raise ValueError("Calculated bytes_per_vector is zero.")

        if data_bytes % bytes_per_vector != 0:
            num_elements_in_data = data_bytes // bytes_per_element if bytes_per_element > 0 else 0
            logger.error(
                f"Data size ({data_bytes} bytes) after header is not a multiple of vector data size ({bytes_per_vector} bytes). "
                f"Total elements in data part if flat: {num_elements_in_data}."
            )
            raise ValueError("Data size (after header) indicates potential data corruption or incorrect dimensions.")

        num_vectors = data_bytes // bytes_per_vector
        logger.debug(f"Expected number of vectors from data part: {num_vectors}")

        elements_to_read = num_vectors * expected_dim
        vectors_flat = np.fromfile(filename, dtype=dtype, count=elements_to_read, offset=header_bytes)

        if vectors_flat.size != elements_to_read:
            logger.error(f"Read {vectors_flat.size} elements, but expected {elements_to_read} elements "
                         f"from data part. File might be truncated or offset logic issue.")
            raise ValueError("Mismatch in expected data size and elements read after offset.")

        reshaped_vectors = vectors_flat.reshape((num_vectors, expected_dim))
        logger.debug(f"Successfully read and reshaped vectors. Shape: {reshaped_vectors.shape}")
        return reshaped_vectors
    except Exception as e:
        logger.error(f"Exception in read_fbin_vectors for {filename}: {e}", exc_info=True)
        raise


class WikiDataIngester(Postgres):
    def __init__(self, data_file_path, embedding_dim, copy_batch_size):
        logger.info("WikiDataIngester __init__ called.")
        try:
            super().__init__()
            if self.conn:
                logger.info("Database connection successful (self.conn is set by Postgres base class).")
            else:
                logger.error(
                    "self.conn is None after super().__init__(). This indicates an issue in Postgres base class connection.")
        except Exception as e:
            logger.error(f"Error during Postgres super().__init__(): {e}", exc_info=True)
            raise

        self.data_file_path = data_file_path
        self.embedding_dim = embedding_dim
        self.copy_batch_size = copy_batch_size
        self.total_rows_ingested = 0
        logger.info(f"WikiDataIngester initialized with data file: {self.data_file_path}")

    def _ingest_batch_with_copy(self, batch_data):
        if not batch_data:
            logger.debug("Empty batch passed to _ingest_batch_with_copy. Skipping.")
            return 0

        logger.debug(f"Attempting to ingest batch of {len(batch_data)} records.")
        if not self.conn:
            logger.error("No database connection available in _ingest_batch_with_copy.")
            return -1

        original_autocommit = self.conn.autocommit
        self.conn.autocommit = False

        cursor = self.conn.cursor()
        buffer = io.StringIO()
        for name, emb_str, embed_bool in batch_data:
            clean_name = str(name).replace('\t', ' ').replace('\n', ' ')
            buffer.write(f"{clean_name}\t{emb_str}\t{embed_bool}\n")
        buffer.seek(0)

        try:
            cursor.copy_expert(
                sql="COPY Entity (entity_name, entity_emb, entity_embed) FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t')",
                file=buffer
            )
            self.conn.commit()
            logger.debug(f"Successfully committed batch of {len(batch_data)} records.")
            return len(batch_data)
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error during COPY: {e}", exc_info=True)
            if batch_data:
                logger.error(f"  Sample name from failed batch (potentially): {batch_data[0][0][:100]}...")
                logger.error(
                    f"  Sample vector string from failed batch: {batch_data[0][1][:200]}...")  # Log part of vector string
            return -1
        finally:
            cursor.close()
            self.conn.autocommit = original_autocommit
            logger.debug("Restored original autocommit state.")

    def process_and_ingest(self):
        logger.info(f"--- Starting Wiki-all 1M Dataset Ingestion ---")

        if not os.path.exists(self.data_file_path):
            logger.error(f"❌ Data file not found: {self.data_file_path}. Aborting ingestion.")
            return

        logger.info(f"Processing data file: {self.data_file_path}")
        try:
            vectors = read_fbin_vectors(self.data_file_path,
                                        expected_dim=self.embedding_dim,
                                        header_bytes=8)

            if vectors is None:
                logger.error("Vector data could not be loaded. Aborting.")
                return

            file_size_for_data = os.path.getsize(self.data_file_path) - 8 if os.path.exists(self.data_file_path) else -1
            if vectors.size == 0 and file_size_for_data > 0:
                logger.warning(
                    "read_fbin_vectors returned an empty array, but data bytes existed after header. Check read_fbin_vectors logic or file content.")

            logger.info(f"Loaded vectors. Shape: {vectors.shape}")
            if vectors.shape[0] > 0 and vectors.shape[1] != self.embedding_dim:
                logger.error(f"❌ Dimension mismatch! Expected {self.embedding_dim}, got {vectors.shape[1]}. Aborting.")
                return

        except Exception as e:
            logger.error(f"Critical error loading or processing vectors: {e}", exc_info=True)
            return

        current_batch_data = []
        logger.info(f"Starting loop to process {vectors.shape[0]} vectors...")
        for i in range(vectors.shape[0]):
            if i < 5 or i % (self.copy_batch_size * 5) == 0:
                logger.debug(f"Processing vector index {i}")

            entity_name = f"wiki_entity_{i}"
            embedding_vector = vectors[i]
            # CORRECTED VECTOR STRING FORMAT HERE:
            emb_str = "[" + ",".join(map(str, embedding_vector)) + "]"  # <<<<<<<<<<<<<<< CORRECTED
            entity_embed_flag = True

            current_batch_data.append((entity_name, emb_str, entity_embed_flag))

            if len(current_batch_data) >= self.copy_batch_size:
                logger.debug(f"Batch of size {len(current_batch_data)} ready for ingestion.")
                ingested_count = self._ingest_batch_with_copy(current_batch_data)
                if ingested_count > 0:
                    self.total_rows_ingested += ingested_count
                elif ingested_count == -1:
                    logger.error(
                        f"Aborting due to COPY error after {self.total_rows_ingested:,} successful insertions.")
                    return
                current_batch_data = []
                logger.info(f"  Ingested {self.total_rows_ingested:,} rows so far...")

        if current_batch_data:
            logger.debug(f"Final batch of size {len(current_batch_data)} ready for ingestion.")
            ingested_count = self._ingest_batch_with_copy(current_batch_data)
            if ingested_count > 0:
                self.total_rows_ingested += ingested_count
            elif ingested_count == -1:
                logger.error(
                    f"Aborting due to COPY error during final batch after {self.total_rows_ingested:,} successful insertions.")
                return

        logger.info(f"\n✅ Finished processing loop. Total entities ingested: {self.total_rows_ingested:,}")
        logger.info("\n--- Ingestion Complete ---")


def run_ingestion():
    logger.info("run_ingestion called.")
    try:
        logger.info("Creating WikiDataIngester instance...")
        ingester = WikiDataIngester(
            data_file_path=DATA_FILE_PATH,
            embedding_dim=EMBEDDING_DIMENSION,
            copy_batch_size=COPY_BATCH_SIZE
        )
        logger.info("WikiDataIngester instance created. Calling process_and_ingest()...")
        ingester.process_and_ingest()

    except psycopg2.Error as db_conn_error:
        logger.error(f"Database connection failed during Ingester initialization: {db_conn_error}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in run_ingestion: {e}", exc_info=True)
    finally:
        logger.info("Script finished.")


if __name__ == "__main__":
    logger.info("Script execution started (__main__).")
    run_ingestion()