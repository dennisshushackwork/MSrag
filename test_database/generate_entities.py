"""
Entity Data Generator for Performance Testing
Inserts entities with realistic names (no embeddings) into the database.
Guarantees exactly the specified number of unique entities. => Please be aware that the names may now be
"""
# External imports
import logging
import random
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

        # Lists for generating realistic entity names
        self.person_first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
            "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Nancy", "Daniel", "Betty",
            "Matthew", "Helen", "Anthony", "Sandra", "Mark", "Donna", "Donald", "Carol",
            "Steven", "Ruth", "Paul", "Sharon", "Andrew", "Michelle", "Kenneth", "Laura",
            "Joshua", "Kimberly", "Kevin", "Deborah", "Brian", "Dorothy", "George", "Lisa",
            "Edward", "Nancy", "Ronald", "Karen", "Timothy", "Betty", "Jason", "Helen",
            "Jeffrey", "Sandra", "Ryan", "Donna", "Jacob", "Carol", "Gary", "Ruth",
            "Nicholas", "Sharon", "Eric", "Michelle", "Jonathan", "Laura", "Stephen", "Emily",
            "Larry", "Emma", "Justin", "Olivia", "Scott", "Sophia", "Brandon", "Ava",
            "Benjamin", "Isabella", "Samuel", "Mia", "Gregory", "Charlotte", "Alexander", "Amelia"
        ]

        self.person_last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
            "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
            "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
            "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
            "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
            "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
            "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
            "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
            "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson"
        ]

        self.company_prefixes = [
            "Global", "International", "Advanced", "Premier", "Elite", "Strategic", "Dynamic",
            "Innovative", "Progressive", "Superior", "Optimal", "Excellence", "Prime", "Apex",
            "Quantum", "Digital", "Smart", "Future", "Next", "Ultra", "Mega", "Super",
            "Pro", "Max", "Plus", "Tech", "Cyber", "Virtual", "Cloud", "Data",
            "United", "Allied", "Associated", "Consolidated", "Integrated", "Unified", "Combined"
        ]

        self.company_types = [
            "Systems", "Solutions", "Technologies", "Services", "Consulting", "Corporation",
            "Enterprises", "Industries", "Group", "Holdings", "Partners", "Associates",
            "Dynamics", "Innovations", "Ventures", "Capital", "Resources", "Networks",
            "Analytics", "Logistics", "Manufacturing", "Development", "Research", "Labs",
            "Studios", "Works", "Company", "Firm", "Agency", "Bureau", "Institute"
        ]

        self.locations = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
            "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
            "Fort Worth", "Columbus", "Charlotte", "San Francisco", "Indianapolis", "Seattle",
            "Denver", "Washington", "Boston", "El Paso", "Nashville", "Detroit", "Oklahoma City",
            "Portland", "Las Vegas", "Memphis", "Louisville", "Baltimore", "Milwaukee",
            "Albuquerque", "Tucson", "Fresno", "Sacramento", "Kansas City", "Mesa",
            "Atlanta", "Omaha", "Colorado Springs", "Raleigh", "Virginia Beach", "Long Beach",
            "Miami", "Oakland", "Minneapolis", "Tulsa", "Tampa", "Arlington", "New Orleans"
        ]

        self.organization_types = [
            "Foundation", "Institute", "Center", "Association", "Society", "Council",
            "League", "Union", "Alliance", "Coalition", "Consortium", "Federation",
            "Academy", "College", "University", "School", "Museum", "Library",
            "Hospital", "Clinic", "Laboratory", "Department", "Ministry", "Bureau"
        ]

    def count_entities(self) -> int:
        """Counts the total number of entities"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(entity_id) as count FROM Entity")
            row = cur.fetchone()
        return row[0] if row else 0

    def count_entities_with_and_without_embeddings(self) -> Tuple[int, int, int]:
        """Returns (total_entities, entities_with_embeddings, entities_without_embeddings)"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN entity_embed = true THEN 1 END) as with_embeddings,
                    COUNT(CASE WHEN entity_embed = false THEN 1 END) as without_embeddings
                FROM Entity
            """)
            row = cur.fetchone()
            return row[0], row[1], row[2] if row else (0, 0, 0)

    def generate_realistic_entity_name(self, entity_type: str = None) -> str:
        """Generate realistic entity names based on type"""
        if entity_type is None:
            entity_type = random.choice(["person", "company", "organization", "location"])

        if entity_type == "person":
            first_name = random.choice(self.person_first_names)
            last_name = random.choice(self.person_last_names)

            # Sometimes add middle initial or suffix
            rand_val = random.random()
            if rand_val < 0.3:
                middle_initial = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                return f"{first_name} {middle_initial}. {last_name}"
            elif rand_val < 0.4:
                suffix = random.choice(["Jr.", "Sr.", "III", "II"])
                return f"{first_name} {last_name} {suffix}"
            else:
                return f"{first_name} {last_name}"

        elif entity_type == "company":
            # Generate company names
            rand_val = random.random()
            if rand_val < 0.4:
                # Prefix + Type format
                prefix = random.choice(self.company_prefixes)
                company_type = random.choice(self.company_types)
                return f"{prefix} {company_type}"
            elif rand_val < 0.7:
                # Person's name + Company type
                last_name = random.choice(self.person_last_names)
                company_type = random.choice(["Corp", "Inc", "LLC", "Ltd", "Co", "Group"])
                return f"{last_name} {company_type}"
            else:
                # Two words + type
                word1 = random.choice(self.company_prefixes)
                word2 = random.choice(self.company_types)
                return f"{word1} {word2}"

        elif entity_type == "organization":
            # Generate organization names
            rand_val = random.random()
            if rand_val < 0.5:
                # Location + Organization type
                location = random.choice(self.locations)
                org_type = random.choice(self.organization_types)
                return f"{location} {org_type}"
            else:
                # Descriptive + Organization type
                descriptor = random.choice(["Medical", "Research", "Cultural", "Educational",
                                          "Environmental", "Social", "Technical", "Scientific"])
                org_type = random.choice(self.organization_types)
                return f"{descriptor} {org_type}"

        elif entity_type == "location":
            # Generate location names
            rand_val = random.random()
            if rand_val < 0.7:
                # Use existing location
                return random.choice(self.locations)
            else:
                # Generate fictional location
                prefix = random.choice(["North", "South", "East", "West", "Central", "Upper", "Lower"])
                base = random.choice(self.locations)
                return f"{prefix} {base}"

        # Fallback
        return f"Entity {random.randint(1000, 9999)}"

    def insert_entities_batch(self, batch_size: int = 1000, entity_distribution: dict = None,
                            global_counter: int = 0) -> Tuple[bool, int]:
        """Insert a batch of entities with realistic names ensuring uniqueness"""
        try:
            if entity_distribution is None:
                # Default distribution
                entity_distribution = {
                    "person": 0.4,      # 40% people
                    "company": 0.3,     # 30% companies
                    "organization": 0.2, # 20% organizations
                    "location": 0.1     # 10% locations
                }

            # Generate batch data with guaranteed unique names
            entities_data = []

            for i in range(batch_size):
                # Choose entity type based on distribution
                rand_val = random.random()
                cumulative = 0
                entity_type = "person"  # default

                for etype, prob in entity_distribution.items():
                    cumulative += prob
                    if rand_val <= cumulative:
                        entity_type = etype
                        break

                # Generate name with unique suffix to guarantee uniqueness
                base_name = self.generate_realistic_entity_name(entity_type)
                unique_id = global_counter + i + 1
                entity_name = f"{base_name} #{unique_id}"

                # No embedding - generated separately
                entities_data.append((entity_name, None, True))

            # Batch insert using execute_values for optimal performance
            insert_query = """
                INSERT INTO Entity (entity_name, entity_emb, entity_embed)
                VALUES %s
            """

            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    insert_query,
                    entities_data,
                    template=None,
                    page_size=batch_size
                )
                inserted_count = cur.rowcount
                self.conn.commit()

            logger.info(f"Successfully inserted {inserted_count} entities in this batch")
            return True, inserted_count

        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
            self.conn.rollback()
            return False, 0

    def insert_million_entities(self, total_entities: int = 1_000_000, batch_size: int = 1000) -> bool:
        """
        Insert exactly the specified number of entities with guaranteed unique names
        Args:
            total_entities: Total number of entities to insert (default 1M)
            batch_size: Number of entities per batch (default 1000)
        """
        logger.info(f"Starting insertion of exactly {total_entities:,} entities in batches of {batch_size}")

        # Check current count
        current_count = self.count_entities()
        logger.info(f"Current entity count: {current_count:,}")

        start_time = time.time()
        successful_batches = 0
        failed_batches = 0
        total_inserted = 0
        global_counter = current_count  # Use current count as starting point for unique IDs

        # Calculate number of batches needed
        num_batches = (total_entities + batch_size - 1) // batch_size

        try:
            for batch_num in range(num_batches):
                # Adjust batch size for last batch if needed
                current_batch_size = min(batch_size, total_entities - total_inserted)

                if current_batch_size <= 0:
                    break

                success, inserted_count = self.insert_entities_batch(
                    current_batch_size,
                    global_counter=global_counter + total_inserted
                )

                if success:
                    successful_batches += 1
                    total_inserted += inserted_count
                else:
                    failed_batches += 1

                # Progress reporting every 100 batches
                if (batch_num + 1) % 100 == 0 or total_inserted >= total_entities:
                    elapsed_time = time.time() - start_time
                    rate = total_inserted / elapsed_time if elapsed_time > 0 else 0

                    logger.info(
                        f"Progress: {batch_num + 1}/{num_batches} batches "
                        f"({total_inserted:,}/{total_entities:,} entities) "
                        f"- Rate: {rate:.0f} entities/sec"
                    )

                # Stop if we've inserted the target number
                if total_inserted >= total_entities:
                    logger.info(f"Target of {total_entities:,} entities reached!")
                    break

        except KeyboardInterrupt:
            logger.warning("Insertion interrupted by user")
            return False

        # Final statistics
        total_time = time.time() - start_time
        final_count = self.count_entities()

        logger.info(f"Insertion completed!")
        logger.info(f"Successful batches: {successful_batches}")
        logger.info(f"Failed batches: {failed_batches}")
        logger.info(f"Total entities inserted: {total_inserted:,}")
        logger.info(f"Target was: {total_entities:,}")
        logger.info(f"Final entity count: {final_count:,}")
        logger.info(f"Total time: {total_time:.2f} seconds")

        if total_inserted > 0:
            logger.info(f"Average rate: {total_inserted / total_time:.0f} entities/sec")

        # Check if we hit the target
        if total_inserted == total_entities:
            logger.info("✅ Successfully inserted exactly the target number of entities!")
            return True
        else:
            logger.warning(f"⚠️ Inserted {total_inserted:,} entities, target was {total_entities:,}")
            return False

    def create_test_entities_with_similar_names(self, count: int = 1000) -> bool:
        """
        Create entities with similar names for testing trigram similarity
        """
        logger.info(f"Creating {count} entities with similar names for testing")

        base_groups = [
            # Technology companies
            ["Apple Inc", "Apple Computer", "Apple Corp", "Apple Technologies", "Apple Systems"],
            ["Google LLC", "Google Inc", "Googol Corp", "Google Technologies", "Google Systems"],
            ["Microsoft Corp", "Microsoft Inc", "Microsooft Ltd", "Microsoft Technologies", "Microsoft Systems"],
            ["Amazon Inc", "Amazon Corp", "Amazn Ltd", "Amazon Technologies", "Amazon Services"],
            ["Facebook Inc", "Facebook Corp", "Meta Inc", "Facebook Technologies", "Meta Platforms"],

            # Similar person names
            ["John Smith", "Jon Smith", "Johnny Smith", "Jonathan Smith", "John Smyth"],
            ["Mary Johnson", "Marie Johnson", "Maria Johnson", "Mary Johnston", "Mary Jonson"],
            ["Michael Brown", "Mike Brown", "Mitchell Brown", "Michael Browne", "Mikael Brown"],
            ["Jennifer Davis", "Jenny Davis", "Jennifer Davies", "Jeniffer Davis", "Jennifer Daviss"],
            ["Robert Wilson", "Bob Wilson", "Roberto Wilson", "Robert Willson", "Robert Wilsons"],

            # Similar organizations
            ["Research Institute", "Research Center", "Research Foundation", "Research Laboratory", "Research Academy"],
            ["Medical Center", "Medical Hospital", "Medical Clinic", "Medical Institute", "Medical Foundation"],
            ["Technology University", "Tech University", "Technical University", "Technology College", "Tech Institute"],
            ["Cultural Center", "Culture Center", "Cultural Institute", "Cultural Foundation", "Culture Institute"],
            ["Environmental Agency", "Environment Agency", "Environmental Bureau", "Environmental Department", "Environment Bureau"]
        ]

        entities_data = []
        current_count = self.count_entities()

        for i in range(count):
            # Select a base group
            group = base_groups[i % len(base_groups)]
            # Select a name from the group
            base_name = group[i % len(group)]

            # Add unique ID to ensure uniqueness
            unique_id = current_count + i + 1
            entity_name = f"{base_name} #{unique_id}"

            # No embedding - generated separately
            entities_data.append((entity_name, None, False))

        try:
            insert_query = """
                INSERT INTO Entity (entity_name, entity_emb, entity_embed)
                VALUES %s
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

    def show_sample_entities(self, limit: int = 20) -> None:
        """Show a sample of generated entities"""
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT entity_id, entity_name, entity_embed
                    FROM Entity 
                    ORDER BY entity_id DESC 
                    LIMIT %s
                """, (limit,))

                results = cur.fetchall()

                if results:
                    print(f"\nSample of {len(results)} most recent entities:")
                    print("-" * 80)
                    for row in results:
                        embed_status = "✅" if row['entity_embed'] else "❌"
                        print(f"ID: {row['entity_id']:6} | Embedded: {embed_status} | Name: {row['entity_name']}")
                    print("-" * 80)
                else:
                    print("No entities found in database")

        except Exception as e:
            logger.error(f"Error showing sample entities: {e}")

    def clear_all_entities(self) -> bool:
        """Clear all entities from the database (use with caution!)"""
        try:
            response = input("⚠️ WARNING: This will delete ALL entities! Type 'DELETE ALL' to confirm: ")
            if response == "DELETE ALL":
                with self.conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE Entity RESTART IDENTITY CASCADE")
                    self.conn.commit()
                logger.info("All entities have been deleted from the database")
                return True
            else:
                print("Operation cancelled")
                return False
        except Exception as e:
            logger.error(f"Error clearing entities: {e}")
            self.conn.rollback()
            return False


def main():
    """Main function to run the entity data generation"""
    generator = EntityDataGenerator()

    print("Clean Entity Data Generator")
    print("=" * 50)
    print("1. Insert 1 million realistic entities")
    print("2. Insert custom number of entities")
    print("3. Create test entities with similar names")
    print("4. Show entity statistics")
    print("5. Show sample entities")
    print("6. Clear all entities (WARNING!)")

    choice = input("\nEnter your choice (1-6): ").strip()

    if choice == "1":
        print("\nInserting 1 million entities with realistic names...")
        success = generator.insert_million_entities()
        if success:
            print("✅ Successfully inserted 1 million entities!")
            generator.show_sample_entities()
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
                generator.show_sample_entities()
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
                generator.show_sample_entities()
            else:
                print("❌ Error creating test entities")
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

    elif choice == "4":
        total, with_emb, without_emb = generator.count_entities_with_and_without_embeddings()
        print(f"\nEntity Statistics:")
        print(f"Total entities: {total:,}")
        print(f"With embeddings: {with_emb:,}")
        print(f"Without embeddings: {without_emb:,}")
        if total > 0:
            print(f"Embedding completion: {(with_emb/total*100):.1f}%")
        else:
            print("No entities found")

    elif choice == "5":
        try:
            limit = int(input("Enter number of entities to show (default 20): ") or "20")
            generator.show_sample_entities(limit)
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

    elif choice == "6":
        success = generator.clear_all_entities()
        if success:
            print("✅ All entities cleared successfully")
        else:
            print("❌ Operation cancelled or failed")

    else:
        print("❌ Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()