#!/usr/bin/env python3
"""
RAGBench evaluation script for Qwen3 Embedder with MRL dimension testing.
Tests 128D, 256D, and 1024D embeddings on training documents with training questions.
As an evaluation training dataset we use:
"""
import logging
import numpy as np
import pandas as pd
import sys
import os
import time
import random
from typing import List, Dict
from datasets import load_dataset
import torch
import json
from collections import defaultdict

# Import your embedder (adjust path as needed)
from emb.embedder import Qwen3Embedder

# Configure logger for immediate stdout output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',)
logger = logging.getLogger(__name__)

def log_and_flush(message: str, level: str = "info"):
    """This method logs message to stdout and flush it to disk."""
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    sys.stdout.flush()
    sys.stderr.flush()


class RAGBenchEvaluator:
    """Evaluates Qwen3 embedder on RAGBench with different MRL dimensions."""
    def __init__(self, embedder_class, max_docs_per_dataset: int = 20, cache_dir: str = "./embedding_cache"):
        """
        Initialize the evaluator.

        Args:
            embedder_class: The Qwen3Embedder class
            max_docs_per_dataset: Maximum documents to sample from each dataset (reduced for memory)
            cache_dir: Directory to save embeddings to disk
        """
        self.embedder_class = embedder_class
        self.max_docs_per_dataset = max_docs_per_dataset # Maximum allowed documents per dataset
        self.cache_dir = cache_dir
        self.embedder = None # The embedder instance
        self.corpus = []
        self.corpus_embeddings_files = {}  # Will store file paths for each dimension
        self.questions = []
        self.question_to_relevant_docs = {}  # Maps question_id to list of RELEVANT DOC IDS

        # Dimensions to test
        self.dimensions = [128, 256, 1024]

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_ragbench_data(self):
        """Load and process RAGBench datasets from training split."""
        log_and_flush("Loading RAGBench datasets...")

        ragbench_datasets = [
            'covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa',
            'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa',
            'tatqa', 'techqa'
        ]

        all_training_examples = []

        # Step 1: Load all training examples from all datasets
        for dataset_name in ragbench_datasets:
            try:
                log_and_flush(f"Loading {dataset_name}...")
                dataset = load_dataset("rungalileo/ragbench", dataset_name)

                if 'train' in dataset:
                    train_data = dataset['train']
                    log_and_flush(f"{dataset_name} train split: {len(train_data)} examples")

                    n_examples = min(self.max_docs_per_dataset, len(train_data))
                    if len(train_data) > n_examples:
                        indices = random.sample(range(len(train_data)), n_examples)
                        train_data = train_data.select(indices)
                        log_and_flush(f"Sampled {n_examples} examples from {dataset_name}")

                    for i, example in enumerate(train_data):
                        all_training_examples.append({
                            'dataset': dataset_name,
                            'original_idx': i,
                            'example': example
                        })
            except Exception as e:
                log_and_flush(f"Error loading {dataset_name}: {e}", "error")
                continue

        log_and_flush(f"Loaded {len(all_training_examples)} total training examples")

        # Step 2: Build corpus from ALL documents in all examples
        doc_id_counter = 0
        for example_meta in all_training_examples:
            example = example_meta['example']
            dataset_name = example_meta['dataset']
            source_example_idx = f"{dataset_name}_{example_meta['original_idx']}"

            doc_texts = self._extract_all_documents_from_example(example)
            for doc_text in doc_texts:
                self.corpus.append({
                    'id': f"doc_{doc_id_counter}", # Stable ID
                    'text': doc_text,
                    'dataset': dataset_name,
                    'source_example_idx': source_example_idx
                })
                doc_id_counter += 1

        # Build relevance mapping using STABLE DOCUMENT IDs before any shuffling occurs.
        source_key_to_doc_ids = defaultdict(list)
        for doc in self.corpus:
            source_key_to_doc_ids[doc['source_example_idx']].append(doc['id'])

        # Step 3: Build questions and map to relevant document IDs
        question_id_counter = 0
        for example_meta in all_training_examples:
            example = example_meta['example']
            dataset_name = example_meta['dataset']
            source_example_idx = f"{dataset_name}_{example_meta['original_idx']}"

            question_text = self._extract_question(example)
            if question_text:
                question_id = f"q_{question_id_counter}"
                self.questions.append({
                    'id': question_id,
                    'question': question_text,
                    'dataset': dataset_name,
                    'source_example_idx': source_example_idx
                })
                # Use the pre-built mapping for relevance
                self.question_to_relevant_docs[question_id] = source_key_to_doc_ids[source_example_idx]
                question_id_counter += 1

        log_and_flush(f"Built corpus: {len(self.corpus)} documents")
        log_and_flush(f"Built questions: {len(self.questions)} questions")

        # Shuffling is now safe because the relevance map uses stable IDs, not indices.
        random.shuffle(self.corpus)
        random.shuffle(self.questions)

        # Limit to 100 questions
        if len(self.questions) > 100:
            kept_questions = self.questions[:100]
            kept_question_ids = {q['id'] for q in kept_questions}
            self.question_to_relevant_docs = {
                qid: docs for qid, docs in self.question_to_relevant_docs.items() if qid in kept_question_ids
            }
            self.questions = kept_questions
            log_and_flush(f"Limited to 100 evaluation questions")

        num_questions_with_relevant = sum(1 for docs in self.question_to_relevant_docs.values() if docs)
        avg_relevant_per_question = np.mean([len(docs) for docs in self.question_to_relevant_docs.values()])
        log_and_flush(f"Questions with relevant docs: {num_questions_with_relevant}/{len(self.questions)}")
        log_and_flush(f"Average relevant docs per question: {avg_relevant_per_question:.2f}")

    def _extract_all_documents_from_example(self, example: Dict) -> List[str]:
        """Extract ALL document texts from a RAGBench training example."""
        documents = []
        possible_fields = ['documents', 'retrieved_contexts', 'context', 'passages', 'content']

        for field in possible_fields:
            if field in example and example[field]:
                content = example[field]
                items_to_process = content if isinstance(content, list) else [content]

                for item in items_to_process:
                    if isinstance(item, str):
                        item = item.strip()
                        if len(item) > 50:
                            # Limit document length to avoid memory issues
                            documents.append(item[:2000] + "..." if len(item) > 2000 else item)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(documents))

    def _extract_question(self, example: Dict) -> str:
        """Extract question from RAGBench training example."""
        possible_fields = ['question', 'query', 'input', 'prompt']
        for field in possible_fields:
            if field in example and example[field]:
                return str(example[field]).strip()
        return None

    def initialize_embedder(self):
        """Initialize the Qwen3 embedder."""
        log_and_flush("Initializing Qwen3 embedder...")
        self.embedder = self.embedder_class()
        if hasattr(self.embedder, 'verify_gpu_usage'):
            self.embedder.verify_gpu_usage()

    def generate_corpus_embeddings(self):
        """Generate embeddings for the corpus in all dimensions and save to disk."""
        if not self.embedder:
            self.initialize_embedder()

        log_and_flush(f"Generating corpus embeddings for {len(self.corpus)} documents...")
        corpus_texts = [doc['text'] for doc in self.corpus]

        for dim in self.dimensions:
            embedding_file = os.path.join(self.cache_dir, f"corpus_embeddings_{dim}d.npy")

            # --- START OF FIX for IndexError ---
            # Check if a valid cache file exists that matches the current corpus size
            cache_is_valid = False
            if os.path.exists(embedding_file):
                try:
                    # Load the cached embeddings to check their shape
                    cached_embeddings = np.load(embedding_file)
                    if cached_embeddings.shape[0] == len(self.corpus):
                        log_and_flush(f"Found valid {dim}D embeddings cache for {len(self.corpus)} documents.")
                        self.corpus_embeddings_files[dim] = embedding_file
                        cache_is_valid = True
                    else:
                        log_and_flush(f"Cache mismatch for {dim}D: Found {cached_embeddings.shape[0]} embeddings, but corpus has {len(self.corpus)} docs. Regenerating.", "warning")
                except Exception as e:
                    log_and_flush(f"Could not load or validate cache file {embedding_file}: {e}. Regenerating.", "warning")

            if cache_is_valid:
                continue

            log_and_flush(f"Generating {dim}D embeddings...")
            start_time = time.time()
            all_embeddings = []
            batch_size = 50

            for i in range(0, len(corpus_texts), batch_size):
                batch_texts = corpus_texts[i:i + batch_size]
                try:
                    # Simplified logic using the same method for all dimensions
                    batch_embeddings = self.embedder.embed_texts_custom_dim(
                        batch_texts, embedding_dim=dim, are_queries=False
                    )
                    all_embeddings.extend(batch_embeddings)
                except torch.cuda.OutOfMemoryError:
                    log_and_flush(f"OOM at batch {i//batch_size + 1}, retrying one by one...", "warning")
                    for single_text in batch_texts:
                        try:
                            single_embedding = self.embedder.embed_texts_custom_dim(
                                [single_text], embedding_dim=dim, are_queries=False
                            )
                            all_embeddings.extend(single_embedding)
                        except Exception as e:
                            log_and_flush(f"Failed to embed document {len(all_embeddings)}: {e}", "error")
                            all_embeddings.append([0.0] * dim) # Add a zero vector

                if (i // batch_size + 1) % 50 == 0:
                    log_and_flush(f"Processed {len(all_embeddings)}/{len(corpus_texts)} documents")
                    torch.cuda.empty_cache()

            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            np.save(embedding_file, embeddings_array)
            self.corpus_embeddings_files[dim] = embedding_file
            end_time = time.time()
            log_and_flush(f"{dim}D embeddings generated in {end_time - start_time:.2f}s, shape: {embeddings_array.shape}")

    def load_corpus_embeddings(self, dim: int) -> np.ndarray:
        """Load corpus embeddings from disk for a specific dimension."""
        embedding_file = self.corpus_embeddings_files[dim]
        log_and_flush(f"Loading {dim}D embeddings from {embedding_file}")
        return np.load(embedding_file)

    def evaluate_retrieval(self, k_values: List[int] = [1, 5, 10, 20]) -> Dict:
        """Evaluate retrieval performance for all dimensions."""
        log_and_flush(f"Evaluating retrieval on {len(self.questions)} questions...")
        results = {}

        for dim in self.dimensions:
            log_and_flush(f"--- Evaluating {dim}D embeddings ---")
            dim_results = defaultdict(list)
            corpus_embeddings = self.load_corpus_embeddings(dim)
            query_texts = [q['question'] for q in self.questions]

            log_and_flush(f"Generating {dim}D query embeddings...")
            query_embeddings = self.embedder.embed_texts_custom_dim(
                query_texts, embedding_dim=dim, are_queries=True
            )
            query_embeddings = np.array(query_embeddings, dtype=np.float32)

            for i, (query_emb, question) in enumerate(zip(query_embeddings, self.questions)):
                question_id = question['id']
                # Use a set of stable IDs for efficient and correct ground truth checking
                relevant_doc_ids = set(self.question_to_relevant_docs.get(question_id, []))

                if not relevant_doc_ids:
                    continue # Skip questions with no relevant documents in the corpus

                similarities = np.dot(corpus_embeddings, query_emb)
                top_indices = np.argsort(similarities)[::-1]

                for k in k_values:
                    # Get the IDs of the retrieved documents and compare sets
                    top_k_indices = top_indices[:k]
                    retrieved_doc_ids = {self.corpus[idx]['id'] for idx in top_k_indices}
                    retrieved_relevant_count = len(retrieved_doc_ids.intersection(relevant_doc_ids))

                    precision_k = retrieved_relevant_count / k
                    recall_k = retrieved_relevant_count / len(relevant_doc_ids)

                    dim_results[f'precision@{k}'].append(precision_k)
                    dim_results[f'recall@{k}'].append(recall_k)

            avg_results = {metric: np.mean(values) for metric, values in dim_results.items() if values}
            results[f'{dim}D'] = avg_results
            log_and_flush(f"{dim}D results: {avg_results}")

        return results

    def run_evaluation(self):
        """Run the complete evaluation pipeline."""
        self.load_ragbench_data()
        self.initialize_embedder()
        self.generate_corpus_embeddings()
        results = self.evaluate_retrieval()
        self.save_results(results)
        return results

    def save_results(self, results: Dict):
        """Save evaluation results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"qwen3_mrl_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        log_and_flush(f"Detailed results saved to {results_file}")

        df = pd.DataFrame.from_dict({(i,j): results[i][j]
                                    for i in results.keys()
                                    for j in results[i].keys()},
                                   orient='index')
        df.index = pd.MultiIndex.from_tuples(df.index, names=['dimension', 'metric'])
        df.columns = ['value']

        csv_file = f"qwen3_mrl_summary_{timestamp}.csv"
        df.to_csv(csv_file)
        log_and_flush(f"Summary saved to {csv_file}")

        log_and_flush("\n=== EVALUATION SUMMARY ===")
        print(df.unstack(level=0).round(4))
        log_and_flush("==========================")

def main():
    """Main execution function."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    evaluator = RAGBenchEvaluator(
        embedder_class=Qwen3Embedder,
        max_docs_per_dataset=100,
        cache_dir="./embedding_cache"
    )
    evaluator.run_evaluation()
    log_and_flush("Evaluation completed!")

if __name__ == "__main__":
    main()
