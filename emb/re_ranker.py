"""
Qwen3 Reranker model: Qwen/Qwen3-Reranker-0.6B
Text reranking model for improved relevance scoring in RAG pipelines.
Apache-2.0 License: https://apache.org/licenses/LICENSE-2.0.
Model loaded from local directory to avoid HuggingFace rate limits.
"""
# External imports:
import torch
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialising the Logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Qwen3Reranker:
    """
    Qwen3 Reranker Service using the Singleton Design Pattern.
    Loads Qwen/Qwen3-Reranker-0.6B from local directory.
    Provides CrossEncoder-style interface for document reranking.
    """
    _instance = None  # Holds the instance of the reranker
    _is_initialized = False  # Flag for reranker initialization

    def __new__(cls, model_path: str = None):
        """Creates a new reranker instance."""
        if cls._instance is None:
            logger.info("Creating new Qwen3 reranker instance.")
            cls._instance = super(Qwen3Reranker, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None):
        """
        Initializes a new Qwen3 reranker instance.
        Args:
            model_path: Path to local model directory. If None, uses environment variable.
        """
        if Qwen3Reranker._is_initialized:
            return

        # Use provided path, environment variable, or fallback to default
        if model_path is None:
            model_path = os.getenv("QWEN3_RERANKER")

        # Convert relative path to absolute path
        if model_path and not os.path.isabs(model_path):
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Resolve the relative path from the script directory
            model_path = os.path.abspath(os.path.join(script_dir, model_path))

        # Verify the model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at: {model_path}")

        # Check for required files
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                logger.warning(f"Required file {file} not found in {model_path}")

        logger.info(f"Initializing Qwen3 reranker with local model from: {model_path}")

        self.model_path = model_path
        self.max_length = 8192  # Qwen3 reranker context length

        # 1) Pick the correct device:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA on GPU!")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        # 2) Load config from local directory:
        logger.info("Loading Qwen3 reranker configuration from local directory...")
        self.config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        # 3) Load tokenizer and model from local directory:
        logger.info("Loading Qwen3 reranker tokenizer from local directory...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            padding_side='left'
        )

        logger.info("Loading Qwen3 reranker model from local directory...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device).eval()

        # 4) Set up special tokens for yes/no classification
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        # 5) Set up prompt templates
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        Qwen3Reranker._is_initialized = True
        logger.info("Qwen3 Reranker initialization complete using local model.")

    def format_instruction(self, instruction: str, query: str, doc: str) -> str:
        """Format instruction for Qwen3 reranker as specified in documentation."""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """Process input pairs for the reranker model."""
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

        # Add prefix and suffix tokens
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens

        # Pad and convert to tensors
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)

        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        return inputs

    def compute_logits(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """Compute relevance scores from model logits."""
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        return scores

    def predict(self, sentence_pairs: List[List[str]], instruction: str = None) -> List[float]:
        """
        Predict relevance scores for query-document pairs.
        Compatible with CrossEncoder interface.

        Args:
            sentence_pairs: List of [query, document] pairs
            instruction: Optional task instruction for better performance

        Returns:
            List of relevance scores (0.0 to 1.0)
        """
        if not sentence_pairs:
            return []

        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'

        # Format all pairs with instructions
        formatted_pairs = []
        for query, doc in sentence_pairs:
            formatted_pair = self.format_instruction(instruction, query, doc)
            formatted_pairs.append(formatted_pair)

        # Process inputs and compute scores
        inputs = self.process_inputs(formatted_pairs)
        scores = self.compute_logits(inputs)

        return scores

    def rank(self, query: str, documents: List[str], instruction: str = None, return_documents: bool = True) -> List[Dict[str, Any]]:
        """
        Rank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts
            instruction: Optional task instruction
            return_documents: Whether to include document text in results

        Returns:
            List of dictionaries with 'corpus_id', 'score', and optionally 'text'
        """
        if not documents:
            return []

        # Create sentence pairs
        sentence_pairs = [[query, doc] for doc in documents]

        # Get relevance scores
        scores = self.predict(sentence_pairs, instruction)

        # Create results
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            result = {
                'corpus_id': i,
                'score': score
            }
            if return_documents:
                result['text'] = doc
            results.append(result)

        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        return results

    def rerank_with_model(self, query: str, documents: List[Dict[str, Any]],
                         top_k: int = 5, instruction: str = None,
                         content_key: str = 'content') -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to a query.
        Compatible with your existing interface.

        Args:
            query: Search query
            documents: List of document dictionaries
            top_k: Number of top documents to return
            instruction: Optional task instruction
            content_key: Key in document dict that contains the text content

        Returns:
            List of reranked documents with added 'rerank_score' field
        """
        if not documents:
            return []

        # Extract document contents for scoring
        doc_contents = []
        for doc in documents:
            if content_key in doc:
                doc_contents.append(doc[content_key])
            else:
                # Fallback: try common content keys or convert to string
                content = doc.get('text', doc.get('content', doc.get('page_content', str(doc))))
                doc_contents.append(content)

        # Create pairs of [query, document_content] for the model
        sentence_pairs = [[query, content] for content in doc_contents]

        # Predict the relevance scores
        scores = self.predict(sentence_pairs, instruction)

        # Add scores to each document (create copies to avoid modifying originals)
        reranked_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = score
            reranked_docs.append(doc_copy)

        # Sort documents by the new rerank score in descending order
        reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Return the top_k documents
        return reranked_docs[:top_k]

    def verify_gpu_usage(self):
        """Verify that the model and tensors are actually on GPU"""
        print(f"\n=== Qwen3 Reranker GPU Verification ===")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Current Device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

            # Check if model parameters are on GPU
            model_device = next(self.model.parameters()).device
            print(f"Model Parameters Device: {model_device}")

            # Test a small reranking to see GPU utilization
            print("Testing GPU utilization with small reranking...")

            # Clear GPU cache and get initial memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(0)

            # Run a test reranking
            test_query = "What is the capital of China?"
            test_doc = "Beijing is the capital city of China."
            test_pairs = [[test_query, test_doc]]

            with torch.no_grad():
                scores = self.predict(test_pairs)
                print(f"Test relevance score: {scores[0]:.4f}")

            # Check memory after inference
            final_memory = torch.cuda.memory_allocated(0)
            memory_used = (final_memory - initial_memory) / 1024**2
            print(f"Memory used for inference: {memory_used:.2f} MB")
        else:
            print("CUDA not available - running on CPU")
        print(f"======================================\n")

    def test_reranker_compatibility(self):
        """Test that reranker works as expected with tutorial examples"""
        print("\n=== Testing Qwen3 Reranker Compatibility ===")

        task = 'Given a web search query, retrieve relevant passages that answer the query'
        queries = [
            "What is the capital of China?",
            "Explain gravity"
        ]
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
        ]

        print("Testing query-document pairs...")

        # Test each query against each document
        for i, query in enumerate(queries):
            print(f"\nQuery {i+1}: '{query}'")
            for j, doc in enumerate(documents):
                sentence_pairs = [[query, doc]]
                scores = self.predict(sentence_pairs, task)
                print(f"  Document {j+1} relevance: {scores[0]:.4f}")

        # Test the rerank_with_model function
        print("\nTesting rerank_with_model function...")
        test_documents = [
            {'content': documents[0], 'id': 1, 'source': 'test1'},
            {'content': documents[1], 'id': 2, 'source': 'test2'}
        ]

        reranked = self.rerank_with_model(
            query=queries[0],
            documents=test_documents,
            top_k=2,
            instruction=task
        )

        print("Reranked results:")
        for i, doc in enumerate(reranked):
            print(f"  Rank {i+1}: Score {doc['rerank_score']:.4f} - {doc['content'][:50]}...")

        print("=============================================\n")


# Test the Qwen3 reranker
if __name__ == "__main__":
    try:
        reranker = Qwen3Reranker()

        # Verify GPU usage
        reranker.verify_gpu_usage()

        # Test compatibility
        reranker.test_reranker_compatibility()

        # Test your specific use case
        print("Testing with your interface...")

        # Sample documents in your format
        documents = [
            {'content': 'Colin Shushack is born in Hedingen', 'id': 1},
            {'content': 'Dennis Shushack works as an engineer', 'id': 2},
            {'content': 'Hedingen is a small town in Switzerland', 'id': 3},
            {'content': 'Switzerland is known for its mountains', 'id': 4}
        ]

        query = "Where was Colin born?"

        # Time the reranking
        import time
        start_time = time.time()

        reranked_results = reranker.rerank_with_model(
            query=query,
            documents=documents,
            top_k=3,
            instruction="Find information that directly answers the user's question"
        )

        end_time = time.time()

        print(f"Reranking took: {end_time - start_time:.3f} seconds")
        print("Reranked results:")
        for i, doc in enumerate(reranked_results):
            print(f"  {i+1}. Score: {doc['rerank_score']:.4f} - {doc['content']}")

        print("Qwen3 reranker local model loading successful!")

    except Exception as e:
        print(f"Error loading Qwen3 reranker: {e}")
        print("Make sure your QWEN3_RERANKER environment variable is set correctly in your .env file.")