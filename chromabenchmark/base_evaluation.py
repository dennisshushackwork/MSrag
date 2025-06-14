"""
Postgres-based evaluation system adapted from the ChromaDB Benchmark:
https://github.com/brandonstarxel/chunking_evaluation/tree/main
This system evaluates the different chunking strategies & entities-relationship extraction using PostgreSQL.
"""
# External imports:
import os
import json
import logging
import platform
import pandas as pd
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Optional, Literal

# Internal imports:
from chunkers.chunker import ChunkingService
from postgres.populate import PopulateQueries
from emb.chunks import ChunkEmbedder
from emb.embedder import Qwen3Embedder
from pipelines.retrieval import Retriever

# Configure logging:
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------- Helper Function from Original Evaluation -------------- #

def sum_of_ranges(ranges:List[Tuple[int, int]]) -> int:
    """Takes a list of ranges of the form (start,end).
    Calculates the total character length covered by a list of ranges.
    I.e. [(0, 10), (15, 20)] => (10-0) + (20-15) = 10 + 5 = 15"""
    return sum(end - start for start, end in ranges)

def union_ranges(ranges:List[Tuple[int, int]]):
    """
    Merges a list of possibly overlapping or contiguous ranges into a set of non-overlapping ranges.
    For example, [(0, 5), (3, 8), (10, 12)] becomes [(0, 8), (10, 12)]. This method is cruical for
    correctly identifying the total relevant span of text.
    """
    # Sort ranges based on the starting index
    sorted_ranges = sorted(ranges, key=lambda x: x[0])

    # Initialize with the first range
    merged_ranges = [sorted_ranges[0]]

    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged_ranges[-1]
        # Check if the current range overlaps or is contiguous with the last range in the merged list
        if current_start <= last_end:
            # Merge the two ranges
            merged_ranges[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap, add the current range as new
            merged_ranges.append((current_start, current_end))
    return merged_ranges

def intersect_two_ranges(range1, range2):
    """
    Finds the intersection (overlap) between two ranges.
    Returns the overlapping range as a tuple (start, end), or None if there's no overlap.
    For example, (0, 10) and (5, 15) would intersect at (5, 10). Important to find what parts of the retrieved
    chunk are too much.
    """
    start1, end1 = range1
    start2, end2 = range2

    # Calculate the maximum of the starting indices and the minimum of the ending indices
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)

    # Check if the intersection is valid (the start is less than or equal to the end)
    if intersect_start <= intersect_end:
        return (intersect_start, intersect_end)
    else:
        return None  # Return an None if there is no intersection


def difference(ranges, target):
    """
    Calculates the difference between a list of ranges and a single target range.
    It returns the parts of the original ranges that *do not* overlap with the target range.
    For example, if ranges = [(0, 10)] and target = (3, 7), it returns [(0, 3), (7, 10)].
    - ranges (list of tuples):
    - target: (tuple)
    """
    result = []
    target_start, target_end = target

    for start, end in ranges:
        if end < target_start or start > target_end:
            # No overlap
            result.append((start, end))
        elif start < target_start and end > target_end:
            # Target is a subset of this range, split it into two ranges
            result.append((start, target_start))
            result.append((target_end, end))
        elif start < target_start:
            # Overlap at the start
            result.append((start, target_start))
        elif end > target_end:
            # Overlap at the end
            result.append((target_end, end))
        # Else, this range is fully contained by the target, and is thus removed
    return result


class PostgresEvaluation:
    """PostgreSQL-based evaluation system for chunking strategies."""
    def __init__(self, questions_csv_path: str, corpora_id_paths: Optional[Dict[str, str]] = None):
        """
         Args:
            questions_csv_path: Path to CSV file containing evaluation questions and references (
            corpora_id_paths: Dictionary mapping corpus IDs to file paths {finance: /Users/dennis/Documents/GitHub/MSrag/chromabenchmark/dataset/corpora/finance.md}
        """
        self.questions_csv_path = questions_csv_path # Contains
        self.corpora_id_paths = corpora_id_paths
        self.questions_df = None
        self.corpus_list = [] # Holds the list of all the corpus filenames (ids)
        self.is_general = False
        self._load_questions_df() # Loads the questions for evaluation.
        # self._embed_questions() # Embeds the questions for further processing.

    def _load_questions_df(self):
        """Load the questions DataFrame from the given questions CSV file.
            - questions: The question in the CSV file
            - references: The references in the corpora (start and end index)
            - corpus_id: From which corpus the question was generated from.
        """
        if os.path.exists(self.questions_csv_path):
            self.questions_df = pd.read_csv(self.questions_csv_path)
            self.questions_df['references'] = self.questions_df['references'].apply(json.loads)
        else:
            self.questions_df = pd.DataFrame(columns=['question', 'references', 'corpus_id'])
        self.corpus_list = self.questions_df['corpus_id'].unique().tolist()
        logger.info(f"Loaded {len(self.questions_df)} questions for {len(self.corpus_list)} corpora")


    def _process_documents_to_postgres(self, chunk_type: Literal["token", "recursive", "cluster"],
                                       chunk_size: int = 400, chunk_overlap: int = 0) -> None:
        """This method generates the chunks using the provided splitter method and then finds the corresponding metadata"""

        # Delete all chunks by type & all documents:
        with PopulateQueries() as db:
            db.clear_chunks_by_type(chunk_type)
            db.clear_documents()

        # Iterating over all documents (corpora):
        for corpus_id in self.corpus_list:
            corpus_path = corpus_id
            logger.info("Processing corpus %s", corpus_path)
            if self.corpora_id_paths is not None:
                corpus_path = self.corpora_id_paths[corpus_id]

            # Read and parse document
            if platform.system() == 'Windows':
                with open(corpus_path, 'r', encoding='utf-8') as file:
                    corpus = file.read()
            else:
                with open(corpus_path, 'r') as file:
                    corpus = file.read()

            # Insert document into database and get document_id
            with PopulateQueries() as db:
                doc_id = db.set_document(corpus)

            # Create chunks using chunking service
            chunker = ChunkingService(
                doc=corpus,
                chunk_type=chunk_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_id=doc_id,
            )
            # Creating the chunks:
            chunker.create_chunk_from_document()

            # Embedd the chunks:
            embedder = ChunkEmbedder()
            embedder.process_chunk_emb_batches()

            # Create the entities and relationships: still open.
        return



    def _embed_questions(self):
        """Embedds the questions using the given embedding method."""
        questions = self.questions_df['question'].tolist()
        question_embeddings = []
        embedder = Qwen3Embedder()
        for question in questions:
            question_emb = embedder.embed_texts(question, are_queries=True)
            question_embeddings.append(question_emb)
        self.questions_df['embedding'] = question_embeddings
        logger.info(f"Generated embeddings for {len(questions)} questions")

    def _retrive_chunks(self, top_k: int, chunk_type: Literal["token", "recursive", "cluster"]):
        """Retrieves the chunks based on the given top_k chunk type."""
        pass



if __name__ == "__main__":
    questions_csv_path = "/home/dennis/Documents/MSrag/chromabenchmark/dataset/questions_df.csv"
    corpora_id_paths = {"finance": "/home/dennis/Documents/MSrag/chromabenchmark/dataset/corpora/finance.md",
                        "state_of_the_union": "/home/dennis/Documents/MSrag/chromabenchmark/dataset/corpora/state_of_the_union.md",
                        "wikitexts": "/home/dennis/Documents/MSrag/chromabenchmark/dataset/corpora/wikitexts.md",
                        "pubmed":"/home/dennis/Documents/MSrag/chromabenchmark/dataset/corpora/pubmed.md",
                        "chatlogs": "/home/dennis/Documents/MSrag/chromabenchmark/dataset/corpora/chatlogs.md"}
    # Initalise:
    evaluator = PostgresEvaluation(questions_csv_path, corpora_id_paths)
    print(evaluator.questions_df)
    evaluator._process_documents_to_postgres(chunk_type="token")



