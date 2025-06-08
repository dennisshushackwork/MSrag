"""
Embedding model: Alibaba-NLP/gte-multilingual-base (loaded from local directory)
Uses MRL to truncate embeddings to 256. https://huggingface.co/Alibaba-NLP/gte-multilingual-base
Apache-2.0 License: https://apache.org/licenses/LICENSE-2.0.
Model loaded from local directory to avoid HuggingFace rate limits.
"""
# External imports:
import torch
import logging
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


class Embedder:
    """
    Embedding Service using the Singleton Design Pattern.
    Loads Alibaba-NLP/gte-multilingual-base from local directory.
    """
    _instance = None  # Holds the instance of the embedder
    _is_initialized = False  # Flag for embedding initialization
    TRUNCATION_DIM = 256

    def __new__(cls, model_path: str = None):
        """Creates a new embedder instance."""
        if cls._instance is None:
            logger.info("Creating new embedder instance.")
            cls._instance = super(Embedder, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None):
        """
        Initializes a new embedder instance.

        Args:
            model_path: Path to local model directory. If None, uses environment variable.
        """
        if Embedder._is_initialized:
            return

        # Use provided path, environment variable, or fallback to default
        if model_path is None:
            model_path = os.getenv("EMBEDDING")
            print(model_path)

        # Convert relative path to absolute path
        if model_path and not os.path.isabs(model_path):
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Resolve the relative path from the script directory
            model_path = os.path.abspath(os.path.join(script_dir, model_path))
            print(model_path)

        # Verify the model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at: {model_path}")

        # Check for required files
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                logger.warning(f"Required file {file} not found in {model_path}")

        logger.info(f"Initializing embedder with local model from: {model_path}")
        self.model_path = model_path
        self.max_tokens = 8192  # Maximum tokens the embedder can handle at once

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

        # 2) Load config from local directory - FIXED VERSION:
        logger.info("Loading model configuration from local directory...")
        self.config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        # 3) Load tokenizer and model from local directory:
        logger.info("Loading tokenizer from local directory...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        logger.info("Loading model from local directory...")
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            add_pooling_layer=False,
        ).to(self.device)

        Embedder._is_initialized = True
        logger.info("Embedder initialization complete using local model.")

    def tokenize_to_ids(self, text: str) -> List[int]:
        """Returns a list of integers representing the token ids of a given `text`."""
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_tokens,
            skip_special_tokens=True
        )
        return tokens["input_ids"].squeeze(0).tolist()

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decodes the tokens back"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a text document"""
        return len(self.tokenize_to_ids(text))

    def _get_raw_embeddings(self, texts: list[str], are_queries: bool = False) -> torch.Tensor:
        """
        Embeds the input texts into 768-dimensional embeddings in full float32 precision.
        """
        # If we have a query vector, we add the prefix: query
        if are_queries:
            query_prefix = 'query: '
            texts = ["{}{}".format(query_prefix, i) for i in texts]

        # 1) Tokenizer:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_tokens
        ).to(self.device)

        # 2) Compute token embeddings
        with torch.no_grad():
            text_embeddings_768d = self.model(**tokens)[0][:, 0]

        return text_embeddings_768d  # Shape: (batch_size, 768)

    def cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculates the cosine similarity between two embeddings."""
        query = torch.tensor(emb1, device=self.device)
        document = torch.tensor(emb2, device=self.device)
        return F.cosine_similarity(query, document, dim=0).item()

    @staticmethod
    def _normalize_embeddings_torch(embeddings_matrix: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings to unit norm along axis 1."""
        return F.normalize(embeddings_matrix, p=2, dim=1)

    def embed_texts(self, texts: list[str], are_queries: bool = False) -> List[List[float]]:
        """Generates 768D embeddings, truncates them to 256D """

        # 1. Get 768D normalized float embeddings (as torch.Tensor on self.device)
        embeddings_768d_torch = self._get_raw_embeddings(texts=texts, are_queries=are_queries)

        # 2. Truncate to TRUNCATION_DIM (e.g., 256):
        truncated_embeddings_torch = embeddings_768d_torch[:, :self.TRUNCATION_DIM]

        # 3. Re-normalize the 256D embeddings:
        normalized_truncated_embeddings_torch = self._normalize_embeddings_torch(truncated_embeddings_torch)
        return normalized_truncated_embeddings_torch.cpu().detach().numpy().tolist()


    def verify_gpu_usage(self):
        """Verify that the model and tensors are actually on GPU"""
        print(f"\n=== GPU Verification ===")
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
                embeddings = self.model(**tokens)[0][:, 0]
                print(f"Output Embeddings Device: {embeddings.device}")

            # Check memory after inference
            final_memory = torch.cuda.memory_allocated(0)
            memory_used = (final_memory - initial_memory) / 1024**2
            print(f"Memory used for inference: {memory_used:.2f} MB")

        else:
            print("CUDA not available - running on CPU")
        print(f"========================\n")

# Add this to your main section:
if __name__ == "__main__":
    try:
        embedder = Embedder()

        # Verify GPU usage
        embedder.verify_gpu_usage()

        text1 = "Colin Shushack is born in Hedingen"
        text2 = "Dennis Shushack"

        print("Generating embeddings...")

        # Time the embedding generation
        import time
        start_time = time.time()

        embeddings = embedder.embed_texts(texts=[text1, text2], are_queries=False)

        end_time = time.time()
        print(f"Embedding generation took: {end_time - start_time:.3f} seconds")

        emb1 = embeddings[0]
        emb2 = embeddings[1]

        cosine = embedder.cosine_similarity(emb1, emb2)
        print(f"Cosine similarity: {cosine}")
        print("Local model loading successful!")

    except Exception as e:
        print(f"Error loading local model: {e}")
        print("Make sure your EMBEDDING environment variable is set correctly in your .env file.")