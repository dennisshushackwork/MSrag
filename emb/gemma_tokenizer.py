"""
This is the tokenizer used by gemma models, which is needed for calculating the accurate amount of tokens
for the LLM. If you use a different model, please change the tokenizer accordingly.
"""
# External imports:
import os
from dotenv import load_dotenv
import logging
from typing import List
from transformers import AutoTokenizer

# Set up basic logging to see the singleton pattern in action.
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GemmaSingletonTokenizer:
    """
    A Singleton class to provide a single, globally accessible instance of the
    Gemma tokenizer loaded from local directory.
    """
    _instance = None
    _is_initialized = False
    _model_path = os.getenv("TOKENIZER")

    def __new__(cls, model_path: str = None):
        """
        Controls the creation of the instance. If an instance does not exist,
        it creates one. Otherwise, it returns the existing instance.
        """
        if cls._instance is None:
            logger.info("Creating new GemmaSingletonTokenizer instance.")
            cls._instance = super(GemmaSingletonTokenizer, cls).__new__(cls)
        else:
            logger.info("Returning existing GemmaSingletonTokenizer instance.")
        return cls._instance

    def __init__(self, model_path: str = None):
        """
        Initializes the tokenizer. A flag `_is_initialized` prevents this
        expensive operation from running more than once.

        Args:
            model_path: Path to local Gemma model directory
        """
        if self._is_initialized:
            return

        # Use provided path or default
        if model_path is not None:
            self._model_path = model_path

        # Convert relative path to absolute path
        if self._model_path and not os.path.isabs(self._model_path):
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Resolve the relative path from the script directory
            self._model_path = os.path.abspath(os.path.join(script_dir, self._model_path))

        # Set model name for the name property
        self._model_name = "gemma-3-1b-it"

        # Verify the model path exists
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"Gemma model directory not found at: {self._model_path}")

        # Check for required tokenizer files
        required_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(self._model_path, file)):
                logger.warning(f"Required tokenizer file {file} not found in {self._model_path}")

        logger.info(f"Performing one-time initialization of the tokenizer from '{self._model_path}'...")

        try:
            # Load the tokenizer from local directory
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                local_files_only=True  # This prevents downloading from HuggingFace
            )
            logger.info(f"Tokenizer for Gemma-3-1B-IT loaded successfully from local directory.")
            self._is_initialized = True
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {self._model_path}. Error: {e}")
            logger.error("Please ensure the model directory exists and contains all required tokenizer files.")
            self._is_initialized = False

    @property
    def name(self) -> str:
        """Returns the model name, similar to tiktoken.Encoding.name."""
        return self._model_name  # Fixed: use _model_name instead of _model_id

    @property
    def model_path(self) -> str:
        """Returns the model path."""
        return self._model_path

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string of text into a list of token IDs.
        """
        if not self._is_initialized:
            logger.warning("Tokenizer is not initialized. Returning empty list.")
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        if not self._is_initialized:
            logger.warning("Tokenizer is not initialized. Returning empty string.")
            return ""
        return self.tokenizer.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        """
        A utility function to count the number of tokens in a given string,
        without adding special tokens.
        """
        if not self._is_initialized:
            logger.warning("Tokenizer is not initialized. Returning 0.")
            return 0
        return len(self.encode(text))

if __name__ == "__main__":
    tokenizer_instance_1 = GemmaSingletonTokenizer()
    sample_text = "Hedingen is a municipality in the district of Affoltern."
    print(f"Text to process: '{sample_text}'")
    print(f"Tokenizer name property: {tokenizer_instance_1.name}\n")
    tokens = tokenizer_instance_1.encode(sample_text)
    print(f"Tokens: {tokens}")
    text = tokenizer_instance_1.decode(tokens)
    print(text)

