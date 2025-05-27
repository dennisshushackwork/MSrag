"""
Embedding model: Alibaba-NLP/gte-multilingual-base
Uses MRL to truncate embeddings to 256. https://huggingface.co/Alibaba-NLP/gte-multilingual-base
Apache-2.0 License: https://apache.org/licenses/LICENSE-2.0
"""
# External imports:
import torch
import logging
from typing import List
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Initialising the Logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedding Service using the Singleton Design Pattern.
    https://huggingface.co/Alibaba-NLP/gte-multilingual-base
    """
    _instance = None # Holds the instance of the embedder
    _is_initialized = False # Flag for embedding initalisation
    TRUNCATION_DIM = 256

    def __new__(cls, model_id: str = "Alibaba-NLP/gte-multilingual-base"):
        """Creates a new embedder instance."""
        if cls._instance is None:
            logger.info("Creating new embedder instance.")
            cls._instance = super(Embedder, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_id: str = "Alibaba-NLP/gte-multilingual-base"):
        """Initializes a new embedder instance."""
        if Embedder._is_initialized:
            return
        logger.info(f"Initializing new embedder instance with id {model_id}.")
        self.model_id = model_id
        self.max_tokens = 8192  # This is the maximum number of tokens the embedder can handle at once.

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

        # 2) Load and add the correct config:
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if self.device.type == "cuda":
            # Enable xFormers memory-efficient kernels on CUDA (may require pip install xformers)
            self.config.use_memory_efficient_attention = True
            # Also force the right attention implementation
            self.config.attn_implementation = "memory_efficient"
        else:
            # MPS/CPU: fallback to eager
            self.config.use_memory_efficient_attention = False
            self.config.attn_implementation = "eager"

        # 3) Tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            config=self.config,
            add_pooling_layer=False,
            trust_remote_code=True
        ).to(self.device)

        Embedder._is_initialized = True
        logger.info("Embedder initialization complete.")

    def tokenize_to_ids(self, text: str) -> List[int]:
        """Returns a list of integers representing the token ids of a given `text`."""
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_tokens
        )
        return tokens["input_ids"].squeeze(0).tolist()

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decodes the tokens back"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a text document"""
        return len(self.tokenize_to_ids(text))

    def _get_raw_embeddings(self, texts: list[str], are_queries: bool = False) -> torch.Tensor:
        """
        Embedds the input texts into 768-dimensional embeddings in full float32 precision.
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

    def cosine_similarity(self, query: torch.Tensor , document: torch.Tensor) -> float:
        """Calculates the cosine similarity between `query` and `document`."""
        query = torch.tensor(emb1, device=self.device)
        document = torch.tensor(emb2, device=self.device)
        return F.cosine_similarity(query, document, dim=0).item()

    @staticmethod
    def _normalize_embeddings_torch(embeddings_matrix: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings to unit norm along axis 1."""
        return F.normalize(embeddings_matrix, p=2, dim=1)

    def embed_texts(self, texts: list[str], are_queries: bool = False) -> torch.Tensor:
        """Generates 768D embeddings, truncates them to 256D """

        # 1. Get 768D normalized float embeddings (as torch.Tensor on self.device)
        embeddings_768d_torch = self._get_raw_embeddings(texts=texts, are_queries=are_queries)

        # 2. Truncate to TRUNCATION_DIM (e.g., 256):
        truncated_embeddings_torch = embeddings_768d_torch[:, :self.TRUNCATION_DIM]

        # 3. Re-normalize the 256D embeddings:
        normalized_truncated_embeddings_torch = self._normalize_embeddings_torch(truncated_embeddings_torch)
        return normalized_truncated_embeddings_torch.cpu().detach().numpy().tolist()

# For testing purposes:
if __name__ == "__main__":
    embedder = Embedder(model_id="Alibaba-NLP/gte-multilingual-base")
    text1 = "Colin Shushack is born in Hedingen"
    text2 = "Dennis Shushack"
    embeddings = embedder.embed_texts(texts=[text1, text2], are_queries=False)
    emb1 = embeddings[0]
    emb2 = embeddings[1]
    cosine = embedder.cosine_similarity(emb1, emb2)
    print(cosine)


