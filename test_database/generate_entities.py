"""
Entity Data Generator for Performance Testing
Inserts 1 million entities with random embeddings into the database.
"""
# External imports
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from psycopg2.extras import execute_values, DictCursor
import time

# Internal imports:
from base import Postgres

# Load environmental variables & logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EntityDataGenerator(Postgres):
    def __init__(self):
        super().__init__()

    def count_entities(self) -> int:
        """Counts the number of entities"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(entity_id) as count FROM Entity")
            row = cur.fetchone()
        return row[0] if row else 0

    def generate_random_embedding(self, dimension: int = 256) -> List[float]:
        """Generate a random normalized embedding vector"""
        # Generate random vector
        return None

    def insert_entities_batch(self, start_id: int, batch_size: int = 1000) -> bool:
        """Insert a batch of entities with random embeddings"""
        try:
            # Generate batch data
            entities_data = []
            for i in range(batch_size):
                entity_name = f"entity_{start_id + i}"
                embedding = self.generate_random_embedding()
                entities_data.append((entity_name, embedding, True))  # Set embed=True

            # Batch insert using execute_values for optimal performance
            insert_query = """
                INSERT INTO Entity (entity_name, entity_emb, entity_embed)
                VALUES %s
                ON CONFLICT (entity_name) DO NOTHING
            """

            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    insert_query,
                    entities_data,
                    template=None,
                    page_size=batch_size
                )
                self.conn.commit()

            logger.info(f"Successfully inserted batch starting from entity_{start_id}")
            return True

        except Exception as e:
            logger.error(f"Error inserting batch starting from entity_{start_id}: {e}")
            self.conn.rollback()
            return False

    def insert_million_entities(self, total_entities: int = 1_000_000, batch_size: int = 1000) -> bool:
        """
        Insert 1 million entities with random embeddings
        Args:
            total_entities: Total number of entities to insert (default 1M)
            batch_size: Number of entities per batch (default 1000)
        """
        logger.info(f"Starting insertion of {total_entities:,} entities in batches of {batch_size}")

        # Check current count
        current_count = self.count_entities()
        logger.info(f"Current entity count: {current_count:,}")

        start_time = time.time()
        successful_batches = 0
        failed_batches = 0

        # Calculate number of batches needed
        num_batches = (total_entities + batch_size - 1) // batch_size

        try:
            for batch_num in range(num_batches):
                start_id = current_count + (batch_num * batch_size) + 1

                # Adjust batch size for last batch if needed
                current_batch_size = min(batch_size, total_entities - (batch_num * batch_size))

                if self.insert_entities_batch(start_id, current_batch_size):
                    successful_batches += 1
                else:
                    failed_batches += 1

                # Progress reporting every 100 batches
                if (batch_num + 1) % 100 == 0:
                    elapsed_time = time.time() - start_time
                    entities_inserted = successful_batches * batch_size
                    rate = entities_inserted / elapsed_time if elapsed_time > 0 else 0

                    logger.info(
                        f"Progress: {batch_num + 1}/{num_batches} batches "
                        f"({entities_inserted:,} entities) "
                        f"- Rate: {rate:.0f} entities/sec"
                    )

        except KeyboardInterrupt:
            logger.warning("Insertion interrupted by user")
            return False

        # Final statistics
        total_time = time.time() - start_time
        total_inserted = successful_batches * batch_size
        final_count = self.count_entities()

        logger.info(f"Insertion completed!")
        logger.info(f"Successful batches: {successful_batches}")
        logger.info(f"Failed batches: {failed_batches}")
        logger.info(f"Total entities inserted: {total_inserted:,}")
        logger.info(f"Final entity count: {final_count:,}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average rate: {total_inserted / total_time:.0f} entities/sec")

        return failed_batches == 0

    def create_test_entities_with_similar_names(self, count: int = 1000) -> bool:
        """
        Create entities with similar names for testing trigram similarity
        Useful for testing the GiST trigram index performance
        """
        logger.info(f"Creating {count} entities with similar names for testing")

        base_names = [
            "apple", "application", "apply", "approach", "appropriate",
            "google", "googol", "gogle", "goggle", "goooogle",
            "microsoft", "microsooft", "microsft", "micorsoft", "mikrosoft",
            "amazon", "amazn", "amazone", "amzon", "amazonn",
            "facebook", "facbook", "facebbok", "faceboook", "facebok"
        ]

        entities_data = []
        current_count = self.count_entities()

        for i in range(count):
            base_name = base_names[i % len(base_names)]
            # Add variations
            if i % 5 == 0:
                entity_name = f"{base_name}_{i}"
            elif i % 5 == 1:
                entity_name = f"{base_name}_company_{i}"
            elif i % 5 == 2:
                entity_name = f"{base_name}_corp_{i}"
            elif i % 5 == 3:
                entity_name = f"{base_name}_inc_{i}"
            else:
                entity_name = f"{base_name}_ltd_{i}"

            embedding = self.generate_random_embedding()
            entities_data.append((entity_name, embedding, True))

        try:
            insert_query = """
                INSERT INTO Entity (entity_name, entity_emb, entity_embed)
                VALUES %s
                ON CONFLICT (entity_name) DO NOTHING
            """

            with self.conn.cursor() as cur:
                execute_values(cur, insert_query, entities_data, page_size=1000)
                self.conn.commit()

            logger.info(f"Successfully created {count} test entities with similar names")
            return True

        except Exception as e:
            logger.error(f"Error creating test entities: {e}")
            self.conn.rollback()
            return False


def main():
    """Main function to run the entity data generation"""
    generator = EntityDataGenerator()

    print("Entity Data Generator")
    print("=" * 50)
    print("1. Insert 1 million random entities")
    print("2. Insert custom number of entities")
    print("3. Create test entities with similar names")
    print("4. Show current entity count")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        print("\nInserting 1 million entities...")
        success = generator.insert_million_entities()
        if success:
            print("✅ Successfully inserted 1 million entities!")
        else:
            print("❌ Some errors occurred during insertion")

    elif choice == "2":
        try:
            count = int(input("Enter number of entities to insert: "))
            batch_size = int(input("Enter batch size (default 1000): ") or "1000")
            print(f"\nInserting {count:,} entities...")
            success = generator.insert_million_entities(count, batch_size)
            if success:
                print(f"✅ Successfully inserted {count:,} entities!")
            else:
                print("❌ Some errors occurred during insertion")
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers.")

    elif choice == "3":
        try:
            count = int(input("Enter number of similar test entities (default 1000): ") or "1000")
            success = generator.create_test_entities_with_similar_names(count)
            if success:
                print(f"✅ Successfully created {count} test entities!")
            else:
                print("❌ Error creating test entities")
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

    elif choice == "4":
        count = generator.count_entities()
        print(f"\nCurrent entity count: {count:,}")

    else:
        print("❌ Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()