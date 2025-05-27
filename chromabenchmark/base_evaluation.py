"""
This script performs the evaluation of the chromadb benchmark.
This script is heavily inspired by the ChromaDB benchmark presented by:
    - https://github.com/brandonstarxel/chunking_evaluation/tree/main
It had to be adjusted to meet the needs of the applications requirements (postgres/kuzu). Certain
packages had to be adjusted due to GPL Licencing.
"""

# External Imports:
import os
import json
import logging
import platform
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal

# Internal Imports:
from utils import rigorous_document_search
from emb.embedder import Embedder
from postgres.populate import PopulateQueries
from postgres.retrieval import RetrievalQueries
from chunkers.chunker import ChunkingService
from chunkers.recursivechunker import RecursiveChunker
from chunkers.sentencechunker import SentenceTokenChunker
from chunkers.tokenchunker import TokenChunker

# Configure logging:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------- Helper Function from Original Evaluation -------------- #
def sum_of_ranges(ranges):
    """Calculates the total character length covered by a list of ranges. I.e. [(0, 10), (15, 20)] (10-0) + (20-15) = 10 + 5 = 15"""
    return sum(end - start for start, end in ranges)

def union_ranges(ranges):
    """
    Takes a list of character ranges and merges any overlapping or adjacent ranges to produce a minimal set of unique, combined ranges
    union_ranges([(0, 10), (5, 15), (20, 25)]) would return [(0, 15), (20, 25)].
    """
    if not ranges:
        return []
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
    """Finds the overlapping character range between two individual ranges. For example, intersect_two_ranges((0, 10), (5, 15)) would return (5, 10)
    If the ranges have no overlap, return None"""
    # Unpack the ranges
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
    Takes a set of ranges and a target range, and returns the difference.
    Subtracts a target range from a list of ranges. It returns the parts of the input ranges that do not
    overlap with the target range. For example, if ranges = [(0, 20)] and target = (5, 15),
    the difference would be [(0, 5), (15, 20)]. This is used to track which parts of the ground truth
    references haven't been covered by retrieved chunks.
    Args:
    - ranges (list of tuples): A list of tuples representing ranges. Each tuple is (a, b) where a <= b.
    - target (tuple): A tuple representing a target range (c, d) where c <= d.

    Returns:
    - List of tuples representing ranges after removing the segments that overlap with the target range.
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
        Initialize the evaluation system.
        Args:
            questions_csv_path: Path to CSV file containing evaluation questions and references
            corpora_id_paths: Dictionary mapping corpus IDs to file paths
        """
        self.questions_csv_path = questions_csv_path
        self.corpora_id_paths = corpora_id_paths
        self.corpus_list = []
        self.embedder = Embedder()
        self.is_general = False

        # Loads the actual questions:
        self._load_questions_df()

    def _load_questions_df(self):
        """Load the questions DataFrame from CSV file."""
        if os.path.exists(self.questions_csv_path):
            self.questions_df = pd.read_csv(self.questions_csv_path)
            self.questions_df['references'] = self.questions_df['references'].apply(json.loads)
        else:
            self.questions_df = pd.DataFrame(columns=['question', 'references', 'corpus_id'])
        self.corpus_list = self.questions_df['corpus_id'].unique().tolist()
        logger.info(f"Loaded {len(self.questions_df)} questions for {len(self.corpus_list)} corpora")

    def _get_chunks_and_metadata_from_postgres(self, chunk_type: str) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve chunks and their metadata from PostgreSQL database.
        """
        documents = []
        metadatas = []

        with RetrievalQueries() as db:
            # Get all chunks of the specified type
            chunks = db.get_chunks_by_type(chunk_type)

            for chunk in chunks:
                chunk_id, chunk_text, chunk_document_id, chunk_tokens, start_idx, end_idx = chunk

                # Convert document_id to corpus_id for compatibility
                corpus_id = str(chunk_document_id)

                documents.append(chunk_text)
                metadatas.append({
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "corpus_id": corpus_id,
                    "chunk_id": chunk_id,
                    "tokens": chunk_tokens
                })

        logger.info(f"Retrieved {len(documents)} chunks from PostgreSQL")
        return documents, metadatas

    def _get_chunks_and_metadata_from_files(self, splitter) -> Tuple[List[str], List[Dict]]:
        """Generate chunks from files using the provided splitter (fallback method)."""
        documents = []
        metadatas = []

        for corpus_id in self.corpus_list:
            corpus_path = corpus_id
            if self.corpora_id_paths is not None:
                corpus_path = self.corpora_id_paths[corpus_id]

            # Read file with proper encoding
            if platform.system() == 'Windows':
                with open(corpus_path, 'r', encoding='utf-8') as file:
                    corpus = file.read()
            else:
                with open(corpus_path, 'r') as file:
                    corpus = file.read()

            current_documents = splitter.split_text(corpus)
            current_metadatas = []
            for document in current_documents:
                try:
                    _, start_index, end_index = rigorous_document_search(corpus, document)
                except Exception as e:
                    logger.error(f"Error finding {document[:50]}... in {corpus_id}: {e}")
                    raise Exception(f"Error finding document chunk in {corpus_id}")

                current_metadatas.append({
                    "start_index": start_index,
                    "end_index": end_index,
                    "corpus_id": corpus_id
                })

            documents.extend(current_documents)
            metadatas.extend(current_metadatas)

        return documents, metadatas

    def _process_documents_to_postgres(self, chunk_type: Literal["token", "recursive", "sentence"],
                                       chunk_size: int = 400, chunk_overlap: int = 100) -> None:
        """
        Process documents and store chunks in PostgreSQL database.
        Args:
            chunk_type: Type of chunking to use
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        # Clear existing chunks of this type
        with PopulateQueries() as db:
            db.clear_chunks_by_type(chunk_type)

        for corpus_id in self.corpus_list:
            corpus_path = corpus_id
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

            # Create chunks using your chunking service
            chunker = ChunkingService(
                doc=corpus,
                chunk_type=chunk_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_id=doc_id
            )
            chunker.create_chunk_from_document()
            logger.info(f"Processed corpus {corpus_id} with {chunk_type} chunking")

    def _full_precision_score(self, chunk_metadatas: List[Dict]) -> Tuple[List[float], List[int]]:
        """
        Calculate precision_omega scores for all chunks that overlap with references.

        Args:
            chunk_metadatas: List of chunk metadata dictionaries

        Returns:
            Tuple of (ioc_scores, highlighted_chunks_count)
        """
        ioc_scores = []
        highlighted_chunks_count = []

        for index, row in self.questions_df.iterrows():
            question = row['question']
            references = row['references']
            corpus_id = row['corpus_id']

            ioc_score = 0
            numerator_sets = []
            denominator_chunks_sets = []
            unused_highlights = [(x['start_index'], x['end_index']) for x in references]
            highlighted_chunk_count = 0

            for metadata in chunk_metadatas:
                chunk_start = metadata['start_index']
                chunk_end = metadata['end_index']
                chunk_corpus_id = str(metadata['corpus_id'])

                if chunk_corpus_id != corpus_id:
                    continue

                contains_highlight = False

                for ref_obj in references:
                    ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
                    intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))

                    if intersection is not None:
                        contains_highlight = True
                        unused_highlights = difference(unused_highlights, intersection)
                        numerator_sets = union_ranges([intersection] + numerator_sets)
                        denominator_chunks_sets = union_ranges([(chunk_start, chunk_end)] + denominator_chunks_sets)

                if contains_highlight:
                    highlighted_chunk_count += 1

            highlighted_chunks_count.append(highlighted_chunk_count)

            # Combine unused highlights and chunks for final denominator
            denominator_sets = union_ranges(denominator_chunks_sets + unused_highlights)

            # Calculate ioc_score if there are numerator sets
            if numerator_sets:
                ioc_score = sum_of_ranges(numerator_sets) / sum_of_ranges(denominator_sets)

            ioc_scores.append(ioc_score)

        return ioc_scores, highlighted_chunks_count

    def _scores_from_retrievals(self, retrievals: List[List[Dict]], highlighted_chunks_count: List[int]) -> Tuple[
        List[float], List[float], List[float]]:
        """
        Calculate IoU, Precision, and Recall scores from retrieved chunks.

        Args:
            retrievals: List of retrieved chunk metadata for each question
            highlighted_chunks_count: Number of chunks to consider for each question

        Returns:
            Tuple of (iou_scores, recall_scores, precision_scores)
        """
        iou_scores = []
        recall_scores = []
        precision_scores = []

        for (index, row), highlighted_chunk_count, metadatas in zip(
                self.questions_df.iterrows(), highlighted_chunks_count, retrievals
        ):
            question = row['question']
            references = row['references']
            corpus_id = row['corpus_id']

            numerator_sets = []
            denominator_chunks_sets = []
            unused_highlights = [(x['start_index'], x['end_index']) for x in references]

            for metadata in metadatas[:highlighted_chunk_count]:
                chunk_start = metadata['start_index']
                chunk_end = metadata['end_index']
                chunk_corpus_id = str(metadata['corpus_id'])

                if chunk_corpus_id != corpus_id:
                    continue

                for ref_obj in references:
                    ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
                    intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))

                    if intersection is not None:
                        unused_highlights = difference(unused_highlights, intersection)
                        numerator_sets = union_ranges([intersection] + numerator_sets)
                        denominator_chunks_sets = union_ranges([(chunk_start, chunk_end)] + denominator_chunks_sets)

            if numerator_sets:
                numerator_value = sum_of_ranges(numerator_sets)
            else:
                numerator_value = 0

            recall_denominator = sum_of_ranges([(x['start_index'], x['end_index']) for x in references])
            precision_denominator = sum_of_ranges(
                [(x['start_index'], x['end_index']) for x in metadatas[:highlighted_chunk_count]])
            iou_denominator = precision_denominator + sum_of_ranges(unused_highlights)

            recall_score = numerator_value / recall_denominator if recall_denominator > 0 else 0
            precision_score = numerator_value / precision_denominator if precision_denominator > 0 else 0
            iou_score = numerator_value / iou_denominator if iou_denominator > 0 else 0

            recall_scores.append(recall_score)
            precision_scores.append(precision_score)
            iou_scores.append(iou_score)

        return iou_scores, recall_scores, precision_scores

    def _retrieve_chunks_for_questions(self, chunk_type: str, retrieve: int = 5) -> List[List[Dict]]:
        """
        Retrieve chunks for each question using PostgreSQL similarity search.
        Args:
            chunk_type: Type of chunking used
            retrieve: Number of chunks to retrieve per question
        Returns:
            List of retrieved chunk metadata for each question
        """
        retrievals = []

        with RetrievalQueries() as db:
            for index, row in self.questions_df.iterrows():
                question = row['question']

                # Generate embedding for the question
                embedding = self.embedder.embed_texts([question], are_queries=True)[0]

                # Perform hybrid search using your existing method
                chunks = db.hybrid_search(question, str(embedding), chunk_type)

                # Convert to metadata format
                chunk_metadatas = []
                for chunk in chunks[:retrieve]:
                    # Assuming chunk format: (chunk_id, chunk_text, similarity, tokens, start_idx, end_idx, doc_id)
                    if len(chunk) >= 7:
                        chunk_id, chunk_text, similarity, tokens, start_idx, end_idx, doc_id = chunk[:7]
                        chunk_metadatas.append({
                            "start_index": start_idx,
                            "end_index": end_idx,
                            "corpus_id": str(doc_id),
                            "chunk_id": chunk_id,
                            "similarity": similarity,
                            "tokens": tokens
                        })
                retrievals.append(chunk_metadatas)

        return retrievals

    def run(self, chunk_type: Literal["token", "recursive", "sentence"],
            chunk_size: int = 400, chunk_overlap: int = 100, retrieve: int = 5,
            use_existing_chunks: bool = True) -> Dict:
        """
        Run the evaluation for a specific chunking strategy.

        Args:
            chunk_type: Type of chunking to evaluate
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            retrieve: Number of chunks to retrieve per question
            use_existing_chunks: Whether to use existing chunks in DB or create new ones

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Running evaluation for {chunk_type} chunking (size={chunk_size}, overlap={chunk_overlap})")

        # Process documents and create chunks if needed
        if not use_existing_chunks:
            self._process_documents_to_postgres(chunk_type, chunk_size, chunk_overlap)

        # Get chunks and metadata from PostgreSQL
        try:
            docs, metadatas = self._get_chunks_and_metadata_from_postgres(chunk_type)
        except Exception as e:
            logger.warning(f"Failed to get chunks from PostgreSQL: {e}. Using file-based approach.")
            return {"error": "Could not retrieve chunks from PostgreSQL"}

        # Calculate precision_omega scores (all chunks that overlap with references)
        brute_iou_scores, highlighted_chunks_count = self._full_precision_score(metadatas)

        # Adjust retrieve count if needed
        if retrieve == -1:
            maximum_n = min(20, max(highlighted_chunks_count) if highlighted_chunks_count else 5)
        else:
            highlighted_chunks_count = [retrieve] * len(highlighted_chunks_count)
            maximum_n = retrieve

        # Retrieve chunks for each question using similarity search
        retrievals = self._retrieve_chunks_for_questions(chunk_type, maximum_n)

        # Calculate final scores
        iou_scores, recall_scores, precision_scores = self._scores_from_retrievals(retrievals, highlighted_chunks_count)

        # Organize results by corpus
        corpora_scores = {}
        for index, row in self.questions_df.iterrows():
            corpus_id = row['corpus_id']
            if corpus_id not in corpora_scores:
                corpora_scores[corpus_id] = {
                    "precision_omega_scores": [],
                    "iou_scores": [],
                    "recall_scores": [],
                    "precision_scores": []
                }

            corpora_scores[corpus_id]['precision_omega_scores'].append(brute_iou_scores[index])
            corpora_scores[corpus_id]['iou_scores'].append(iou_scores[index])
            corpora_scores[corpus_id]['recall_scores'].append(recall_scores[index])
            corpora_scores[corpus_id]['precision_scores'].append(precision_scores[index])

        # Calculate overall statistics
        results = {
            "corpora_scores": corpora_scores,
            "iou_mean": np.mean(iou_scores),
            "iou_std": np.std(iou_scores),
            "recall_mean": np.mean(recall_scores),
            "recall_std": np.std(recall_scores),
            "precision_omega_mean": np.mean(brute_iou_scores),
            "precision_omega_std": np.std(brute_iou_scores),
            "precision_mean": np.mean(precision_scores),
            "precision_std": np.std(precision_scores),
            "chunk_type": chunk_type,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieve_count": retrieve
        }

        logger.info(f"Evaluation completed - IoU: {results['iou_mean']:.3f}, "
                    f"Precision: {results['precision_mean']:.3f}, "
                    f"Recall: {results['recall_mean']:.3f}")

        return results

    def run_comparative_evaluation(self, chunk_configs: List[Dict], retrieve: int = 5) -> pd.DataFrame:
        """
        Run evaluation for multiple chunking configurations and return comparison results.

        Args:
            chunk_configs: List of dictionaries with chunking configuration
            retrieve: Number of chunks to retrieve per question

        Returns:
            DataFrame with comparison results
        """
        results = []

        for config in chunk_configs:
            chunk_type = config['chunk_type']
            chunk_size = config.get('chunk_size', 400)
            chunk_overlap = config.get('chunk_overlap', 100)

            try:
                result = self.run(chunk_type, chunk_size, chunk_overlap, retrieve)
                result['config'] = f"{chunk_type}_{chunk_size}_{chunk_overlap}"
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate config {config}: {e}")

        # Convert to DataFrame for easy comparison
        df_results = pd.DataFrame([
            {
                'config': r['config'],
                'chunk_type': r['chunk_type'],
                'chunk_size': r['chunk_size'],
                'chunk_overlap': r['chunk_overlap'],
                'iou_mean': r['iou_mean'],
                'iou_std': r['iou_std'],
                'recall_mean': r['recall_mean'],
                'recall_std': r['recall_std'],
                'precision_mean': r['precision_mean'],
                'precision_std': r['precision_std'],
                'precision_omega_mean': r['precision_omega_mean'],
                'precision_omega_std': r['precision_omega_std']
            }
            for r in results
        ])

        return df_results.sort_values('iou_mean', ascending=False)
