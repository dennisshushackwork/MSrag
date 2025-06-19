"""
Qwen3 Embedding model: Qwen/Qwen3-Embedding-0.6B
Uses MRL to support custom dimensions from 32 to 1024. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
Apache-2.0 License: https://apache.org/licenses/LICENSE-2.0.
Model loaded from local directory to avoid HuggingFace rate limits.

OPTIMIZED VERSION:
- Memory efficient float32 precision throughout
- pgvector-compatible string output
- No float64 conversion waste
- Faster processing with proper tensor handling
"""
# External imports:
import torch
import logging
import numpy as np
from typing import List
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialising the Logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Qwen3Embedder:
    """
    Qwen3 Embedding Service using the Singleton Design Pattern.
    Loads Qwen/Qwen3-Embedding-0.6B from local directory.
    Supports custom embedding dimensions from 32 to 1024 via MRL.
    OPTIMIZED: Memory efficient with pgvector string output support.
    """
    _instance = None  # Holds the instance of the embedder
    _is_initialized = False  # Flag for embedding initialization
    TRUNCATION_DIM = 256  # Default truncation dimension
    MAX_EMBEDDING_DIM = 1024  # Qwen3's maximum embedding dimension

    def __new__(cls, model_path: str = None):
        """Creates a new embedder instance."""
        if cls._instance is None:
            logger.info("Creating new Qwen3 embedder instance.")
            cls._instance = super(Qwen3Embedder, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None):
        """
        Initializes a new Qwen3 embedder instance.
        Args:
            model_path: Path to local model directory. If None, uses environment variable.
        """
        if Qwen3Embedder._is_initialized:
            return

        # Use provided path, environment variable, or fallback to default
        if model_path is None:
            model_path = os.getenv("QWEN3_EMBEDDING")

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

        logger.info(f"Initializing optimized Qwen3 embedder with local model from: {model_path}")

        self.model_path = model_path
        self.max_tokens = 32000  # Qwen3 supports 32K context length

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
        logger.info("Loading Qwen3 model configuration from local directory...")
        self.config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        # 3) Load tokenizer and model from local directory:
        logger.info("Loading Qwen3 tokenizer from local directory...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            padding_side='left'  # Qwen3 recommends left padding
        )

        logger.info("Loading Qwen3 model from local directory...")
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Explicit float32 for efficiency
        ).to(self.device)

        # Set model to eval mode for inference optimization
        self.model.eval()

        Qwen3Embedder._is_initialized = True
        logger.info("Optimized Qwen3 Embedder initialization complete using local model.")

    def tokenize_to_ids(self, text: str) -> List[int]:
        """Returns a list of integers representing the token ids of a given `text`."""
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_tokens,
        )
        return tokens["input_ids"].squeeze(0).tolist()

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decodes the tokens back"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a text document"""
        return len(self.tokenize_to_ids(text))

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Qwen3 specific pooling function - uses last token pooling as recommended.
        Optimized for better performance.
        """
        left_padding = attention_mask[:, -1].all()  # More efficient check
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format instruction for Qwen3 as recommended in the documentation."""
        return f'Instruct: {task_description}\nQuery:{query}'

    def _get_raw_embeddings(self, texts: list[str], are_queries: bool = False, custom_instruction: str = None) -> torch.Tensor:
        """
        Embeds the input texts into 1024-dimensional embeddings in full float32 precision.
        Follows Qwen3's recommended approach with instruction formatting.
        OPTIMIZED: Better memory management and tensor handling.
        """
        processed_texts = []

        # Apply instruction formatting for queries
        if are_queries:
            if custom_instruction is None:
                # Default instruction as recommended by Qwen3
                default_task = 'Given a web search query, retrieve relevant passages that answer the query'
                processed_texts = [self.get_detailed_instruct(default_task, text) for text in texts]
            else:
                processed_texts = [self.get_detailed_instruct(custom_instruction, text) for text in texts]
        else:
            # No instruction needed for documents
            processed_texts = texts

        # 1) Tokenize with left padding as recommended:
        batch_dict = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(self.device)

        # 2) Get model outputs with memory optimization
        with torch.no_grad():
            outputs = self.model(**batch_dict)

        # 3) Extract embeddings using last token pooling (Qwen3 specific)
        text_embeddings_1024d = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        return text_embeddings_1024d

    def cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculates the cosine similarity between two embeddings."""
        query = torch.tensor(emb1, device=self.device, dtype=torch.float32)
        document = torch.tensor(emb2, device=self.device, dtype=torch.float32)
        return F.cosine_similarity(query, document, dim=0).item()

    def cosine_similarity_from_strings(self, emb_str1: str, emb_str2: str) -> float:
        """Calculates cosine similarity directly from pgvector string format."""
        # Parse pgvector strings: '[1.0,2.0,3.0]' -> [1.0, 2.0, 3.0]
        emb1 = [float(x) for x in emb_str1.strip('[]').split(',')]
        emb2 = [float(x) for x in emb_str2.strip('[]').split(',')]
        return self.cosine_similarity(emb1, emb2)

    @staticmethod
    def _normalize_embeddings_torch(embeddings_matrix: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings to unit norm along axis 1."""
        return F.normalize(embeddings_matrix, p=2, dim=1)

    # MAIN EMBEDDING METHODS - OPTIMIZED FOR PGVECTOR STRINGS

    def embed_texts(self, texts: list[str], are_queries: bool = False, custom_instruction: str = None) -> List[str]:
        """
        Generates full 1024D embeddings as pgvector-compatible strings.
        OPTIMIZED: Returns float32 precision strings directly compatible with PostgreSQL pgvector.

        Returns:
            List of strings in format: ['[1.234567,2.345678,...]', ...]
        """
        # 1. Get 1024D embeddings (as torch.Tensor on self.device)
        embeddings_1024d_torch = self._get_raw_embeddings(texts=texts, are_queries=are_queries, custom_instruction=custom_instruction)

        # 2. CRITICAL: Normalize embeddings for proper cosine similarity
        normalized_embeddings_torch = self._normalize_embeddings_torch(embeddings_1024d_torch)

        # 3. Convert to numpy float32 (memory efficient)
        embeddings_numpy = normalized_embeddings_torch.detach().cpu().numpy().astype(np.float32)

        # 3. Convert to pgvector string format
        pgvector_strings = []
        for embedding in embeddings_numpy:
            # Format as '[1.234567,2.345678,...]' with 6 decimal places
            values_str = ','.join(f'{x:.6f}' for x in embedding)
            pgvector_strings.append(f'[{values_str}]')

        return pgvector_strings

    def embed_texts_truncated(self, texts: list[str], are_queries: bool = False, custom_instruction: str = None, truncation_dim: int = None) -> List[str]:
        """
        Generates 1024D embeddings, truncates them to specified dimension, returns as pgvector strings.
        OPTIMIZED: Memory efficient float32 precision with pgvector compatibility.

        Args:
            truncation_dim: Dimension to truncate to (32-1024). If None, uses TRUNCATION_DIM.

        Returns:
            List of pgvector-compatible strings: ['[1.234567,2.345678,...]', ...]
        """
        if truncation_dim is None:
            truncation_dim = self.TRUNCATION_DIM

        # Validate truncation dimension
        if truncation_dim < 32 or truncation_dim > self.MAX_EMBEDDING_DIM:
            raise ValueError(f"Truncation dimension must be between 32 and {self.MAX_EMBEDDING_DIM}")

        # 1. Get 1024D embeddings (as torch.Tensor on self.device)
        embeddings_1024d_torch = self._get_raw_embeddings(texts=texts, are_queries=are_queries, custom_instruction=custom_instruction)

        # 2. CRITICAL: Normalize BEFORE truncation for proper cosine similarity
        normalized_embeddings_torch = self._normalize_embeddings_torch(embeddings_1024d_torch)

        # 3. Truncate to specified dimension:
        truncated_embeddings_torch = normalized_embeddings_torch[:, :truncation_dim]

        # 4. Convert to numpy float32 (OPTIMIZED: maintains precision, saves memory)
        embeddings_numpy = truncated_embeddings_torch.detach().cpu().numpy().astype(np.float32)

        # 4. Convert to pgvector string format
        pgvector_strings = []
        for embedding in embeddings_numpy:
            # Format as '[1.234567,2.345678,...]' with 6 decimal places
            values_str = ','.join(f'{x:.6f}' for x in embedding)
            pgvector_strings.append(f'[{values_str}]')

        return pgvector_strings

    def embed_texts_custom_dim(self, texts: list[str], embedding_dim: int, are_queries: bool = False, custom_instruction: str = None) -> List[str]:
        """
        Generates embeddings with custom dimension (32-1024) using MRL support.
        Returns pgvector-compatible strings.
        This leverages Qwen3's native MRL (Matryoshka Representation Learning) capability.

        Returns:
            List of pgvector-compatible strings: ['[1.234567,2.345678,...]', ...]
        """
        if embedding_dim < 32 or embedding_dim > self.MAX_EMBEDDING_DIM:
            raise ValueError(f"Embedding dimension must be between 32 and {self.MAX_EMBEDDING_DIM}")

        return self.embed_texts_truncated(texts, are_queries, custom_instruction, embedding_dim)


    def verify_gpu_usage(self):
        """Verify that the model and tensors are actually on GPU"""
        print(f"\n=== Qwen3 GPU Verification ===")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Current Device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

            # Check if model parameters are on GPU
            model_device = next(self.model.parameters()).device
            print(f"Model Parameters Device: {model_device}")

            # Test a small embedding to see GPU utilization
            print("Testing GPU utilization with small embedding...")

            # Clear GPU cache and get initial memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(0)

            # Run a test embedding
            test_text = ["This is a test sentence to verify GPU usage"]
            with torch.no_grad():
                tokens = self.tokenizer(
                    test_text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_tokens
                ).to(self.device)

                # Check if tokens are on GPU
                print(f"Input Tokens Device: {tokens['input_ids'].device}")

                # Run model inference
                outputs = self.model(**tokens)
                embeddings = self.last_token_pool(outputs.last_hidden_state, tokens['attention_mask'])
                print(f"Output Embeddings Device: {embeddings.device}")

            # Check memory after inference
            final_memory = torch.cuda.memory_allocated(0)
            memory_used = (final_memory - initial_memory) / 1024**2
            print(f"Memory used for inference: {memory_used:.2f} MB")
        else:
            print("CUDA not available - running on CPU")
        print(f"===============================\n")

    def test_embeddings_match_tutorial(self):
        """Test that embeddings match the Qwen3 tutorial example format"""
        print("\n=== Testing Qwen3 Embedding Compatibility ===")

        # Use the same test inputs as the Qwen3 tutorial
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        queries = [
            'What is the capital of China?',
            'Explain gravity'
        ]
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
        ]

        print("Generating query embeddings with instructions...")
        query_embeddings_str = self.embed_texts(queries, are_queries=True, custom_instruction=task)

        print("Generating document embeddings...")
        document_embeddings_str = self.embed_texts(documents, are_queries=False)

        # Convert back to tensors for similarity calculation
        query_embeddings = []
        for emb_str in query_embeddings_str:
            emb = [float(x) for x in emb_str.strip('[]').split(',')]
            query_embeddings.append(emb)

        document_embeddings = []
        for emb_str in document_embeddings_str:
            emb = [float(x) for x in emb_str.strip('[]').split(',')]
            document_embeddings.append(emb)

        # Calculate similarity scores like in the tutorial
        query_tensor = torch.tensor(query_embeddings, device=self.device)
        doc_tensor = torch.tensor(document_embeddings, device=self.device)

        scores = (query_tensor @ doc_tensor.T)
        print(f"Similarity scores: {scores.tolist()}")

        # Expected from Qwen3 tutorial: [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]
        print("Qwen3 tutorial expected: [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]")
        print("If your scores are close to these values, the implementation is correct!")
        print("=============================================\n")



# Test the optimized Qwen3 embedder
if __name__ == "__main__":
    try:
        embedder = Qwen3Embedder()

        # Verify GPU usage
        embedder.verify_gpu_usage()

        # Test compatibility with tutorial
        embedder.test_embeddings_match_tutorial()

        # Test your original use case with optimized output
        text1 = "Brack Obama"
        text2 = "Barack Hussein Obama"
        print("Generating optimized Qwen3 embeddings for your test case...")

        # Time the embedding generation
        import time
        start_time = time.time()
        embeddings = embedder.embed_texts_custom_dim(texts=[text1, text2], are_queries=False, embedding_dim=256)
        similarity = embedder.cosine_similarity_from_strings(embeddings[0], embeddings[1])
        print(embeddings[0])
        print(embeddings[1])
        print("YOOO")
        print(f"Cosine similarity: {similarity}")
        end_time = time.time()

        print(f"Embedding generation took: {end_time - start_time:.3f} seconds")
        print(f"First embedding (pgvector format): {embeddings[0]}")
        print(f"Second embedding (pgvector format): {embeddings[1]}")


        # Test different dimensions
        print("\nTesting different dimensions...")
        embeddings_512d = embedder.embed_texts_custom_dim([text1], embedding_dim=512, are_queries=False)
        embeddings_128d = embedder.embed_texts_custom_dim([text1], embedding_dim=128, are_queries=False)
        embeddings_64d = embedder.embed_texts_custom_dim([text1], embedding_dim=64, are_queries=False)

        print(f"512D embedding length: {len(embeddings_512d[0])} chars")
        print(f"128D embedding length: {len(embeddings_128d[0])} chars")
        print(f"64D embedding length: {len(embeddings_64d[0])} chars")

        print("Testing")
        test_entity = "Advanced Enterprises #9531"
        embs = embedder.embed_texts_custom_dim([test_entity], embedding_dim=256, are_queries=False)

    except Exception as e:
        print(f"Error loading Qwen3 local model: {e}")
        print("Make sure your QWEN3_EMBEDDING environment variable is set correctly in your .env file.")