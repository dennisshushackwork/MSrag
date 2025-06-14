"""
RAGBench Retrieval-Only Evaluation Framework - ENHANCED WITH DEBUG OUTPUT
Evaluates 4 retrieval strategies with realistic multi-domain testing
Focuses only on retrieval quality, not response generation
https://arxiv.org/html/2407.11005v1

ENHANCEMENT: Prints document content and first 2 retrieved chunks for each question
"""

# External imports:
import platform
import json
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
import random

# Internal imports:
from emb.chunks import ChunkEmbedder
from postgres.populate import PopulateQueries
from chunkers.chunker import ChunkingService

@dataclass
class RetrievalConfig:
    """Configuration for retrieval evaluation"""
    retriever_name: str
    method: str  # 'semantic' or 'hybrid'
    chunk_type: str = "token"
    top_k: int = 10
    use_reranker: bool = False

@dataclass
class RetrievalResult:
    """Container for retrieval evaluation results"""
    config_name: str
    dataset_name: str
    question_id: str
    question_text: str
    retrieved_chunks: List[Dict]
    relevant_documents: List[str]
    metrics: Dict[str, float]
    response_time: float
    context_length: int
    success: bool = True
    error_message: str = ""

class ImprovedRAGBenchEvaluator:
    """
    IMPROVED evaluation framework for realistic retrieval testing
    - Multi-domain challenge (medical + legal + financial + technical)
    - Stricter relevance thresholds (60% instead of 30%)
    - Better evaluation metrics (Jaccard similarity + bidirectional overlap)
    - ENHANCED: Debug output showing documents and retrieved chunks
    """

    def __init__(self, retriever, output_dir: str = "./realistic_retrieval_evaluation", debug_output: bool = True):
        self.retriever = retriever
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.debug_output = debug_output  # NEW: Control debug printing

        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # CHALLENGING dataset selection - mix multiple domains
        self.challenging_datasets = [
            'covidqa',    # Medical/biomedical
            'cuad',       # Legal contracts
            'finqa',      # Financial reasoning
            'techqa',     # Technical support
            'msmarco'     # General knowledge
        ]

        # Define the 4 retrieval strategies to evaluate
        self.retrieval_configs = [
            RetrievalConfig(
                retriever_name="semantic_basic",
                method="semantic",
                chunk_type="token",
                top_k=10,
                use_reranker=False
            ),
            RetrievalConfig(
                retriever_name="semantic_reranked",
                method="semantic",
                chunk_type="token",
                top_k=10,
                use_reranker=True
            ),
            RetrievalConfig(
                retriever_name="hybrid_basic",
                method="hybrid",
                chunk_type="token",
                top_k=10,
                use_reranker=False
            ),
            RetrievalConfig(
                retriever_name="hybrid_reranked",
                method="hybrid",
                chunk_type="token",
                top_k=10,
                use_reranker=True
            )
        ]

    def print_debug_info(self, question: str, relevant_documents: List[str],
                        retrieved_chunks: List[Dict], config_name: str,
                        dataset_name: str, question_id: str):
        """
        NEW: Print debug information for each question evaluation
        Shows the question, related documents, and first 2 retrieved chunks
        """
        if not self.debug_output:
            return

        print(f"\n{'='*80}")
        print(f"🔍 EVALUATION DEBUG: {config_name}")
        print(f"Dataset: {dataset_name} | Question ID: {question_id}")
        print(f"{'='*80}")

        # Print the question
        print(f"\n📝 QUESTION:")
        print(f"   {question}")

        # Print relevant documents (ground truth)
        print(f"\n📚 RELEVANT DOCUMENTS ({len(relevant_documents)} total):")
        for i, doc in enumerate(relevant_documents):
            # Truncate very long documents for readability
            doc_preview = doc[:300] + "..." if len(doc) > 300 else doc
            print(f"   [{i+1}] {doc_preview}")

        # Print first 2 retrieved chunks
        if retrieved_chunks:
            print(f"\n🎯 RETRIEVED CHUNKS (showing first 2 of {len(retrieved_chunks)}):")
            for i, chunk in enumerate(retrieved_chunks[:2]):
                if isinstance(chunk, dict):
                    # Extract chunk content
                    content = (chunk.get('content') or
                             chunk.get('text') or
                             chunk.get('chunk_text') or
                             chunk.get('passage_text', ''))

                    # Extract metadata if available
                    score = chunk.get('score', chunk.get('similarity', 'N/A'))
                    doc_id = chunk.get('document_id', chunk.get('doc_id', 'N/A'))

                    # Truncate content for readability
                    content_preview = content[:400] + "..." if len(content) > 400 else content

                    print(f"   [{i+1}] Score: {score} | Doc ID: {doc_id}")
                    print(f"       Content: {content_preview}")
                else:
                    # Handle non-dict chunks
                    chunk_str = str(chunk)
                    chunk_preview = chunk_str[:400] + "..." if len(chunk_str) > 400 else chunk_str
                    print(f"   [{i+1}] {chunk_preview}")
        else:
            print(f"\n❌ NO CHUNKS RETRIEVED")

        print(f"\n{'='*80}")

    def load_ragbench_datasets(self, datasets_to_load: List[str] = None,
                              sample_size_per_dataset: int = 25) -> Dict[str, List[Dict]]:
        """
        Load RAGBench test datasets for CHALLENGING retrieval evaluation

        Args:
            datasets_to_load: List of dataset names (default: challenging mix)
            sample_size_per_dataset: Number of samples per dataset
        """
        if datasets_to_load is None:
            datasets_to_load = self.challenging_datasets

        loaded_datasets = {}

        for dataset_name in datasets_to_load:
            try:
                self.logger.info(f"Loading {dataset_name} test set...")

                # Load test split only
                dataset = load_dataset("rungalileo/ragbench", dataset_name, split="test")

                # Convert to list and sample
                dataset_list = []
                for i, item in enumerate(dataset):
                    dataset_list.append({
                        'id': item.get('id', str(i)),
                        'question': item['question'],
                        'documents': item['documents']  # Ground truth relevant documents
                    })

                # Sample for manageable evaluation
                if len(dataset_list) > sample_size_per_dataset:
                    dataset_list = random.sample(dataset_list, sample_size_per_dataset)

                loaded_datasets[dataset_name] = dataset_list
                self.logger.info(f"Loaded {len(dataset_list)} samples from {dataset_name}")

            except Exception as e:
                self.logger.error(f"Failed to load {dataset_name}: {str(e)}")
                continue

        total_samples = sum(len(data) for data in loaded_datasets.values())
        self.logger.info(f"Total evaluation samples: {total_samples} across {len(loaded_datasets)} domains")
        self.logger.info("🎯 MULTI-DOMAIN CHALLENGE: This tests cross-domain retrieval ability!")

        return loaded_datasets

    def process_documents_to_database(self, datasets: Dict[str, List[Dict]],
                                    chunk_type: Literal["token", "recursive", "cluster"] = "token",
                                    chunk_size: int = 400,
                                    chunk_overlap: int = 0) -> Dict[str, str]:
        """
        Process all documents from datasets into PostgreSQL database using your pipeline
        """

        self.logger.info("Processing RAGBench documents to database...")
        self.logger.info("🔥 CREATING CHALLENGING MULTI-DOMAIN DATABASE...")

        # Create temporary directory for documents
        docs_dir = self.output_dir / "temp_documents"
        docs_dir.mkdir(exist_ok=True)

        # Extract all unique documents across ALL domains
        unique_documents = set()
        document_mapping = {}
        domain_stats = defaultdict(int)

        for dataset_name, items in datasets.items():
            for item in items:
                for doc_idx, document in enumerate(item['documents']):
                    if document not in unique_documents:
                        unique_documents.add(document)
                        doc_id = f"{dataset_name}_{item['id']}_{doc_idx}"
                        document_mapping[document] = {
                            'doc_id': doc_id,
                            'dataset': dataset_name,
                            'item_id': item['id']
                        }
                        domain_stats[dataset_name] += 1

        self.logger.info(f"Found {len(unique_documents)} unique documents across domains:")
        for domain, count in domain_stats.items():
            self.logger.info(f"  {domain}: {count} documents")

        # Create document files
        corpus_list = []
        for i, document in enumerate(tqdm(unique_documents, desc="Creating document files")):
            file_path = docs_dir / f"doc_{i:06d}.txt"

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(document)

            corpus_list.append(str(file_path))
            document_mapping[document]['file_path'] = str(file_path)

        # Save document mapping
        mapping_file = self.output_dir / "document_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump({doc: info for doc, info in document_mapping.items()},
                     f, indent=2, ensure_ascii=False)

        # Process documents using your actual pipeline
        self.logger.info(f"Processing {len(corpus_list)} documents into PostgreSQL...")

        # Delete all chunks by type & all documents:
        with PopulateQueries() as db:
            db.clear_chunks_by_type(chunk_type)
            db.clear_documents()

        # Process each document using your pipeline
        for corpus_id in tqdm(corpus_list, desc="Processing documents to database"):
            corpus_path = corpus_id
            self.logger.info("Processing corpus %s", corpus_path)

            # Read and parse document
            if platform.system() == 'Windows':
                with open(corpus_path, 'r', encoding='utf-8') as file:
                    corpus = file.read()
            else:
                with open(corpus_path, 'r') as file:
                    corpus = file.read()

            # Insert document into database and get document_id
            with PopulateQueries() as db:
                db_doc_id = db.set_document(corpus)

            # Update document mapping with the actual database document ID
            for document_content, mapping_info in document_mapping.items():
                if mapping_info['file_path'] == corpus_path:
                    mapping_info['db_document_id'] = db_doc_id
                    break

            # Create chunks using chunking service
            chunker = ChunkingService(
                doc=corpus,
                chunk_type=chunk_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_id=db_doc_id,
            )

            # Creating the chunks:
            chunker.create_chunk_from_document()

            # Embed the chunks:
            embedder = ChunkEmbedder()
            embedder.process_chunk_emb_batches()

        # Save updated document mapping with database IDs
        mapping_file = self.output_dir / "document_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump({doc: info for doc, info in document_mapping.items()},
                     f, indent=2, ensure_ascii=False)

        self.logger.info("Document processing completed!")
        self.logger.info("✅ MULTI-DOMAIN DATABASE READY - Now retrieval must find RIGHT domain chunks!")

        # Log mapping summary
        db_ids = [info.get('db_document_id') for info in document_mapping.values() if 'db_document_id' in info]
        self.logger.info(f"Processed {len(db_ids)} documents with database IDs: {min(db_ids) if db_ids else 'N/A'} to {max(db_ids) if db_ids else 'N/A'}")

        return document_mapping

    def calculate_improved_retrieval_metrics(self, retrieved_chunks: List[Dict],
                                           relevant_documents: List[str],
                                           k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        IMPROVED retrieval metrics with stricter evaluation criteria

        Args:
            retrieved_chunks: Chunks returned by retrieval system
            relevant_documents: Ground truth relevant documents from RAGBench
            k_values: K values for precision@k, recall@k calculations
        """
        metrics = {}

        if not retrieved_chunks or not relevant_documents:
            # Return zero metrics if no results
            for k in k_values:
                metrics[f'precision@{k}'] = 0.0
                metrics[f'recall@{k}'] = 0.0
                metrics[f'f1@{k}'] = 0.0
            metrics.update({'mrr': 0.0, 'map': 0.0, 'hit_rate': 0.0})
            return metrics

        # Extract chunk text content
        retrieved_texts = []
        for chunk in retrieved_chunks:
            if isinstance(chunk, dict):
                content = (chunk.get('content') or
                          chunk.get('text') or
                          chunk.get('chunk_text') or
                          chunk.get('passage_text', ''))
                retrieved_texts.append(content.lower().strip() if content else '')
            else:
                retrieved_texts.append(str(chunk).lower().strip())

        # Prepare ground truth documents
        relevant_docs_lower = [doc.lower().strip() for doc in relevant_documents]

        # IMPROVED RELEVANCE SCORING with multiple methods
        relevance_scores = []
        for retrieved_text in retrieved_texts:
            if not retrieved_text:
                relevance_scores.append(0.0)
                continue

            max_relevance = 0.0
            retrieved_words = set(retrieved_text.split())

            for relevant_doc in relevant_docs_lower:
                if not relevant_doc:
                    continue

                relevant_words = set(relevant_doc.split())
                if len(relevant_words) == 0:
                    continue

                # Method 1: Jaccard Similarity (intersection over union)
                intersection = len(retrieved_words & relevant_words)
                union = len(retrieved_words | relevant_words)
                jaccard_sim = intersection / union if union > 0 else 0

                # Method 2: Bidirectional overlap (both directions matter)
                overlap_retrieved = intersection / len(retrieved_words) if len(retrieved_words) > 0 else 0
                overlap_relevant = intersection / len(relevant_words) if len(relevant_words) > 0 else 0
                bidirectional = (overlap_retrieved + overlap_relevant) / 2

                # Method 3: Exact substring matching (very strict)
                substring_match = (retrieved_text in relevant_doc or relevant_doc in retrieved_text)

                # Combine methods for final relevance score
                relevance = max(
                    jaccard_sim * 0.4,              # 40% weight to Jaccard
                    bidirectional * 0.4,            # 40% weight to bidirectional
                    1.0 if substring_match else 0   # 20% weight to exact match
                )

                max_relevance = max(max_relevance, relevance)

            # STRICTER THRESHOLD: Require 60% relevance instead of 30%
            relevance_scores.append(1.0 if max_relevance >= 0.6 else 0.0)

        # Calculate metrics for each k value
        for k in k_values:
            scores_k = relevance_scores[:k]
            relevant_retrieved = sum(scores_k)

            # Precision@k: fraction of retrieved chunks that are relevant
            precision_k = relevant_retrieved / k if k > 0 else 0
            metrics[f'precision@{k}'] = precision_k

            # Recall@k: fraction of relevant docs that have chunks retrieved
            matched_docs = 0
            for relevant_doc in relevant_docs_lower:
                for i in range(min(k, len(retrieved_texts))):
                    retrieved_text = retrieved_texts[i]
                    if not retrieved_text or not relevant_doc:
                        continue

                    retrieved_words = set(retrieved_text.split())
                    relevant_words = set(relevant_doc.split())

                    if len(relevant_words) > 0:
                        # Use Jaccard similarity for recall (slightly more lenient)
                        intersection = len(retrieved_words & relevant_words)
                        union = len(retrieved_words | relevant_words)
                        jaccard_sim = intersection / union if union > 0 else 0

                        if jaccard_sim >= 0.3:  # 30% threshold for recall
                            matched_docs += 1
                            break

            recall_k = matched_docs / len(relevant_documents) if len(relevant_documents) > 0 else 0
            metrics[f'recall@{k}'] = recall_k

            # F1@k
            if precision_k + recall_k > 0:
                f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                f1_k = 0
            metrics[f'f1@{k}'] = f1_k

        # Mean Reciprocal Rank (MRR)
        mrr = 0
        for i, score in enumerate(relevance_scores):
            if score > 0:
                mrr = 1 / (i + 1)
                break
        metrics['mrr'] = mrr

        # Mean Average Precision (MAP)
        if len(relevant_documents) > 0:
            ap = 0
            relevant_found = 0
            for i, score in enumerate(relevance_scores):
                if score > 0:
                    relevant_found += 1
                    precision_at_i = relevant_found / (i + 1)
                    ap += precision_at_i
            metrics['map'] = ap / len(relevant_documents) if relevant_found > 0 else 0
        else:
            metrics['map'] = 0

        # Hit Rate (Success Rate): whether any relevant chunk was found
        metrics['hit_rate'] = 1.0 if any(score > 0 for score in relevance_scores) else 0.0

        return metrics

    def evaluate_single_query(self, config: RetrievalConfig,
                             dataset_name: str, item: Dict) -> RetrievalResult:
        """Evaluate retrieval for a single query"""

        question = item['question']
        relevant_documents = item['documents']

        try:
            start_time = time.time()

            # Call appropriate retrieval method
            if config.method == "semantic":
                retrieved_chunks = self.retriever.semantic_retrieval(
                    query=question,
                    chunking_method=config.chunk_type,
                    top_k=config.top_k,
                    chroma=True,  # Return chunks for evaluation
                    re_ranker=config.use_reranker
                )
            elif config.method == "hybrid":
                retrieved_chunks = self.retriever.hybrid_search(
                    query=question,
                    chunking_method=config.chunk_type,
                    top_k=config.top_k,
                    chroma=True,  # Return chunks for evaluation
                    re_ranker=config.use_reranker
                )
            else:
                raise ValueError(f"Unknown retrieval method: {config.method}")

            response_time = time.time() - start_time

            # Handle different response formats
            if isinstance(retrieved_chunks, list) and len(retrieved_chunks) > 0:
                if (len(retrieved_chunks) == 3 and
                    isinstance(retrieved_chunks[1], (int, float)) and
                    isinstance(retrieved_chunks[2], (int, float))):
                    # This is the [response, time, context_length] format
                    chunks = retrieved_chunks[0] if isinstance(retrieved_chunks[0], list) else []
                else:
                    # This is direct chunks list
                    chunks = retrieved_chunks
            else:
                chunks = []

            # NEW: Print debug information for each evaluation
            self.print_debug_info(
                question=question,
                relevant_documents=relevant_documents,
                retrieved_chunks=chunks,
                config_name=config.retriever_name,
                dataset_name=dataset_name,
                question_id=item['id']
            )

            # Calculate context length
            context_length = 0
            if chunks:
                context_length = sum(chunk.get('tokens', 0) for chunk in chunks if isinstance(chunk, dict))

            # Calculate IMPROVED retrieval metrics
            metrics = self.calculate_improved_retrieval_metrics(chunks, relevant_documents)

            return RetrievalResult(
                config_name=config.retriever_name,
                dataset_name=dataset_name,
                question_id=item['id'],
                question_text=question,
                retrieved_chunks=chunks,
                relevant_documents=relevant_documents,
                metrics=metrics,
                response_time=response_time,
                context_length=context_length,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Error evaluating query {item['id']} with {config.retriever_name}: {str(e)}")
            return RetrievalResult(
                config_name=config.retriever_name,
                dataset_name=dataset_name,
                question_id=item['id'],
                question_text=question,
                retrieved_chunks=[],
                relevant_documents=relevant_documents,
                metrics={},
                response_time=0,
                context_length=0,
                success=False,
                error_message=str(e)
            )

    def run_full_evaluation(self, datasets: Dict[str, List[Dict]]) -> Dict[str, List[RetrievalResult]]:
        """Run retrieval evaluation for all configurations on all datasets"""

        all_results = {}

        for config in self.retrieval_configs:
            self.logger.info(f"🔍 Evaluating {config.retriever_name}...")

            config_results = []
            total_questions = sum(len(items) for items in datasets.values())

            with tqdm(total=total_questions, desc=f"Evaluating {config.retriever_name}") as pbar:
                for dataset_name, items in datasets.items():
                    for item in items:
                        result = self.evaluate_single_query(config, dataset_name, item)
                        config_results.append(result)

                        # Update progress bar
                        pbar.update(1)
                        if config_results:
                            success_count = sum(1 for r in config_results if r.success)
                            success_rate = success_count / len(config_results)
                            avg_mrr = np.mean([r.metrics.get('mrr', 0) for r in config_results if r.success])
                            pbar.set_postfix({
                                'dataset': dataset_name[:8],
                                'success': f'{success_rate:.1%}',
                                'MRR': f'{avg_mrr:.3f}'
                            })

            all_results[config.retriever_name] = config_results

            # Log summary
            successful_results = [r for r in config_results if r.success]
            if successful_results:
                avg_mrr = np.mean([r.metrics.get('mrr', 0) for r in successful_results])
                avg_precision_5 = np.mean([r.metrics.get('precision@5', 0) for r in successful_results])
                avg_hit_rate = np.mean([r.metrics.get('hit_rate', 0) for r in successful_results])
                avg_time = np.mean([r.response_time for r in successful_results])

                self.logger.info(f"{config.retriever_name} Results:")
                self.logger.info(f"  Success: {len(successful_results)}/{len(config_results)}")
                self.logger.info(f"  Avg MRR: {avg_mrr:.3f}")
                self.logger.info(f"  Avg P@5: {avg_precision_5:.3f}")
                self.logger.info(f"  Avg Hit Rate: {avg_hit_rate:.3f}")
                self.logger.info(f"  Avg Time: {avg_time:.3f}s")

        return all_results

    def create_summary_report(self, all_results: Dict[str, List[RetrievalResult]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create summary report of retrieval performance"""

        # Overall summary
        overall_summary = []
        dataset_summary = []

        for config_name, results in all_results.items():
            successful_results = [r for r in results if r.success]

            if not successful_results:
                continue

            # Calculate overall metrics
            overall_metrics = {}
            metric_names = successful_results[0].metrics.keys()

            for metric in metric_names:
                values = [r.metrics.get(metric, 0) for r in successful_results]
                overall_metrics[metric] = np.mean(values)

            response_times = [r.response_time for r in successful_results]
            context_lengths = [r.context_length for r in successful_results]

            overall_summary.append({
                'retriever': config_name,
                'total_queries': len(results),
                'successful_queries': len(successful_results),
                'success_rate': len(successful_results) / len(results),
                'avg_response_time': np.mean(response_times),
                'avg_context_length': np.mean(context_lengths),
                **overall_metrics
            })

            # Per-dataset breakdown
            results_by_dataset = defaultdict(list)
            for result in successful_results:
                results_by_dataset[result.dataset_name].append(result)

            for dataset_name, dataset_results in results_by_dataset.items():
                dataset_metrics = {}
                for metric in metric_names:
                    values = [r.metrics.get(metric, 0) for r in dataset_results]
                    dataset_metrics[metric] = np.mean(values)

                dataset_summary.append({
                    'retriever': config_name,
                    'dataset': dataset_name,
                    'num_queries': len(dataset_results),
                    **dataset_metrics
                })

        overall_df = pd.DataFrame(overall_summary)
        dataset_df = pd.DataFrame(dataset_summary)

        return overall_df, dataset_df

    def analyze_realistic_results(self, overall_df: pd.DataFrame, dataset_df: pd.DataFrame):
        """Analyze results to check if they are realistic"""

        print("\n" + "🔍 REALISTIC RESULTS ANALYSIS")
        print("="*60)
        print("What to expect with IMPROVED evaluation:")
        print("- MRR: 0.2-0.7 (not 0.99)")
        print("- Hit Rate: 0.5-0.9 (not 1.0)")
        print("- Clear differences between strategies")
        print("- Cross-domain performance variations")
        print()

        if not overall_df.empty:
            # Check if results are realistic
            avg_mrr = overall_df['mrr'].mean()
            avg_hit_rate = overall_df['hit_rate'].mean()
            mrr_std = overall_df['mrr'].std()

            print("🎯 REALISM CHECK:")
            print(f"Average MRR: {avg_mrr:.3f} {'✅ Realistic' if 0.15 <= avg_mrr <= 0.8 else '❌ Still unrealistic'}")
            print(f"Average Hit Rate: {avg_hit_rate:.3f} {'✅ Realistic' if 0.4 <= avg_hit_rate <= 0.95 else '❌ Still unrealistic'}")
            print(f"Strategy Differences: {mrr_std:.3f} {'✅ Good variation' if mrr_std > 0.02 else '❌ Too similar'}")

            # Show per-dataset performance
            if not dataset_df.empty:
                print(f"\n📊 CROSS-DOMAIN PERFORMANCE:")
                pivot_mrr = dataset_df.pivot(index='dataset', columns='retriever', values='mrr')
                print(pivot_mrr.round(3))

    def save_results_and_report(self, all_results: Dict[str, List[RetrievalResult]],
                               overall_df: pd.DataFrame, dataset_df: pd.DataFrame):
        """Save results and generate final report"""

        # Save CSV summaries
        overall_df.to_csv(self.output_dir / "realistic_retrieval_summary.csv", index=False)
        dataset_df.to_csv(self.output_dir / "realistic_dataset_breakdown.csv", index=False)

        # Save detailed JSON results
        detailed_results = {}
        for config_name, results in all_results.items():
            detailed_results[config_name] = [
                {
                    'dataset': r.dataset_name,
                    'question_id': r.question_id,
                    'success': r.success,
                    'metrics': r.metrics,
                    'response_time': r.response_time,
                    'num_chunks_retrieved': len(r.retrieved_chunks)
                }
                for r in results
            ]

        with open(self.output_dir / "realistic_detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # Print final report
        print("\n" + "="*80)
        print("🎯 REALISTIC RETRIEVAL EVALUATION RESULTS")
        print("="*80)

        if not overall_df.empty:
            print("\nOVERALL PERFORMANCE:")
            display_cols = ['retriever', 'success_rate', 'mrr', 'precision@5', 'hit_rate', 'avg_response_time']
            display_cols = [col for col in display_cols if col in overall_df.columns]
            print(overall_df[display_cols].round(4).to_string(index=False))

            print("\nBEST PERFORMERS:")
            key_metrics = ['mrr', 'precision@5', 'hit_rate']
            for metric in key_metrics:
                if metric in overall_df.columns:
                    best_idx = overall_df[metric].idxmax()
                    best_retriever = overall_df.loc[best_idx, 'retriever']
                    best_score = overall_df.loc[best_idx, metric]
                    print(f"  {metric:12}: {best_retriever:20} ({best_score:.4f})")

        self.logger.info(f"Results saved to {self.output_dir}")

    def debug_retrieval_matches(self, all_results: Dict[str, List[RetrievalResult]], num_examples: int = 5):
        """
        NEW: Debug function to analyze why retrieval matches are occurring
        """
        print(f"\n🔍 DEBUGGING RETRIEVAL MATCHES (Top {num_examples} examples)")
        print("="*80)

        for config_name, results in all_results.items():
            successful_results = [r for r in results if r.success and r.metrics.get('hit_rate', 0) > 0]

            if not successful_results:
                continue

            print(f"\n📊 {config_name.upper()} - Successful Retrievals:")

            # Sort by MRR to get best examples
            successful_results.sort(key=lambda x: x.metrics.get('mrr', 0), reverse=True)

            for i, result in enumerate(successful_results[:num_examples]):
                print(f"\n  Example {i+1}: MRR={result.metrics.get('mrr', 0):.3f}")
                print(f"  Question: {result.question_text[:100]}...")

                if result.retrieved_chunks:
                    chunk = result.retrieved_chunks[0]  # First chunk
                    if isinstance(chunk, dict):
                        content = (chunk.get('content') or
                                 chunk.get('text') or
                                 chunk.get('chunk_text') or
                                 chunk.get('passage_text', ''))
                        print(f"  Best Chunk: {content[:150]}...")

                print(f"  Ground Truth: {result.relevant_documents[0][:150] if result.relevant_documents else 'None'}...")
                print("  " + "-"*60)

    def analyze_why_scores_too_high(self, all_results: Dict[str, List[RetrievalResult]]):
        """
        NEW: Analyze if evaluation scores are unrealistically high
        """
        print(f"\n🎯 ANALYZING EVALUATION REALISM")
        print("="*80)

        all_successful = []
        for config_name, results in all_results.items():
            successful_results = [r for r in results if r.success]
            all_successful.extend(successful_results)

        if not all_successful:
            print("No successful results to analyze")
            return

        # Analyze MRR distribution
        mrr_scores = [r.metrics.get('mrr', 0) for r in all_successful]
        hit_rates = [r.metrics.get('hit_rate', 0) for r in all_successful]

        print(f"MRR Statistics:")
        print(f"  Mean: {np.mean(mrr_scores):.3f}")
        print(f"  Median: {np.median(mrr_scores):.3f}")
        print(f"  Std: {np.std(mrr_scores):.3f}")
        print(f"  Perfect scores (1.0): {sum(1 for s in mrr_scores if s >= 0.99)}/{len(mrr_scores)}")

        print(f"\nHit Rate Statistics:")
        print(f"  Mean: {np.mean(hit_rates):.3f}")
        print(f"  Perfect hit rate (1.0): {sum(1 for s in hit_rates if s >= 0.99)}/{len(hit_rates)}")

        # Check for unrealistic patterns
        perfect_mrr_count = sum(1 for s in mrr_scores if s >= 0.99)
        if perfect_mrr_count > len(mrr_scores) * 0.3:
            print(f"\n⚠️  WARNING: {perfect_mrr_count} perfect MRR scores suggest evaluation may be too lenient")

        perfect_hit_count = sum(1 for s in hit_rates if s >= 0.99)
        if perfect_hit_count > len(hit_rates) * 0.8:
            print(f"\n⚠️  WARNING: {perfect_hit_count} perfect hit rates suggest documents are too similar to chunks")


# Main execution function - ENHANCED VERSION
def run_realistic_retrieval_evaluation(retriever,
                                      datasets_to_test: List[str] = None,
                                      samples_per_dataset: int = 25,
                                      chunk_type: Literal["token", "recursive", "cluster"] = "token",
                                      chunk_size: int = 400,
                                      chunk_overlap: int = 0,
                                      debug_output: bool = True):
    """
    ENHANCED retrieval evaluation pipeline with realistic multi-domain testing

    NEW: Includes debug output showing documents and retrieved chunks for each question

    Args:
        retriever: Your retriever instance
        datasets_to_test: List of datasets (default: challenging mix)
        samples_per_dataset: Number of samples per dataset
        chunk_type: Type of chunking to use
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        debug_output: Whether to print debug info for each evaluation (NEW)
    """

    if datasets_to_test is None:
        # Default to challenging multi-domain mix
        datasets_to_test = ['covidqa', 'cuad', 'finqa', 'techqa', 'msmarco']

    print("🎯 ENHANCED RETRIEVAL EVALUATION WITH DEBUG OUTPUT")
    print("="*60)
    print("IMPROVEMENTS:")
    print("✅ Multi-domain challenge (medical + legal + financial + technical)")
    print("✅ Stricter relevance threshold (60% instead of 30%)")
    print("✅ Better metrics (Jaccard + bidirectional overlap)")
    print("✅ Realistic expected scores (MRR: 0.2-0.7, Hit Rate: 0.5-0.9)")
    print("✅ NEW: Debug output showing documents and chunks for each question")
    print()

    # Initialize ENHANCED evaluator with debug output
    evaluator = ImprovedRAGBenchEvaluator(retriever, debug_output=debug_output)

    # Load datasets
    datasets = evaluator.load_ragbench_datasets(
        datasets_to_load=datasets_to_test,
        sample_size_per_dataset=samples_per_dataset
    )

    # Process documents to database using your pipeline
    evaluator.process_documents_to_database(
        datasets,
        chunk_type=chunk_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Run evaluation
    all_results = evaluator.run_full_evaluation(datasets)

    # Create summary and save results
    overall_df, dataset_df = evaluator.create_summary_report(all_results)
    evaluator.save_results_and_report(all_results, overall_df, dataset_df)

    # Run debug analysis
    evaluator.debug_retrieval_matches(all_results, num_examples=3)
    evaluator.analyze_why_scores_too_high(all_results)

    return overall_df, dataset_df, all_results


if __name__ == "__main__":
    print("🎯 ENHANCED RAGBench Retrieval Evaluation Framework")
    print("="*60)
    print("USAGE EXAMPLES:")
    print()
    print("1. BASIC REALISTIC EVALUATION WITH DEBUG:")
    print("   from pipelines.retrieval import Retriever")
    print("   retriever = Retriever(model='openai')")
    print("   overall_df, dataset_df, results = run_realistic_retrieval_evaluation(retriever)")
    print()
    print("2. CUSTOM EVALUATION WITH DEBUG OUTPUT:")
    print("   overall_df, dataset_df, results = run_realistic_retrieval_evaluation(")
    print("       retriever=retriever,")
    print("       datasets_to_test=['covidqa', 'cuad', 'finqa'],  # 3 domains")
    print("       samples_per_dataset=30,  # 90 total questions")
    print("       debug_output=True  # Show documents and chunks for each question")
    print("   )")
    print()
    print("3. QUICK TEST WITHOUT DEBUG OUTPUT:")
    print("   overall_df, dataset_df, results = run_realistic_retrieval_evaluation(")
    print("       retriever=retriever,")
    print("       datasets_to_test=['covidqa', 'finqa'],  # 2 domains")
    print("       samples_per_dataset=10,  # 20 total questions")
    print("       debug_output=False  # No debug printing")
    print("   )")
    print()
    print("NEW DEBUG FEATURES:")
    print("• Shows relevant documents for each question")
    print("• Displays first 2 retrieved chunks with scores")
    print("• Analyzes why evaluation scores might be too high")
    print("• Provides examples of successful retrieval matches")
    print()
    print("KEY IMPROVEMENTS:")
    print("• 60% relevance threshold (instead of 30%)")
    print("• Jaccard similarity + bidirectional overlap")
    print("• Multi-domain testing (medical + legal + financial)")
    print("• Expected realistic scores: MRR 0.2-0.7, Hit Rate 0.5-0.9")
    print("• Clear performance differences between strategies")
    print()
    print("Required imports: ChunkEmbedder, PopulateQueries, ChunkingService")