
# This script is adapted from the LangChain package, developed by LangChain AI.
# Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py
# License: MIT License
# It has been further adjusted/optimised to fit the chunking method needed for this application

# External imports:
from abc import ABC, abstractmethod
from attr import dataclass
import logging
from typing import (
    Callable,
    List,
    TypeVar,
)
# Internal imports:
from emb.gemma_tokenizer import GemmaSingletonTokenizer
from chunkers.base import BaseChunker

# Configuring logging:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
TS = TypeVar("TS", bound="TextSplitter")


class TextSplitter(BaseChunker, ABC):
    """Interface for splitting text into chunks."""
    def __init__(
        self,
        doc_id: int,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> None:
        """Create a new TextSplitter."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._doc_id = doc_id
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[tuple]:
        """Split text into multiple components and return tuples."""

@dataclass(frozen=True)
class TokenizerConfig:
    """Helper data class for token splitting."""
    chunk_overlap: int
    tokens_per_chunk: int
    decode: Callable[[List[int]], str]
    encode: Callable[[str], List[int]]


class FixedTokenChunker(TextSplitter):
    """Splitting text to tokens using the Gemma tokenizer."""
    def __init__(
            self,
            doc_id: int,
            chunk_size: int = 400,
            chunk_overlap: int = 200,
    ) -> None:
        """Create a new FixedTokenChunker."""
        super().__init__(doc_id=doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Get the singleton instance of our Gemma tokenizer
        self._tokenizer = GemmaSingletonTokenizer()

    def split_text(self, text: str) -> List[tuple]:
        """Split text and return tuples in format: (doc_id, chunk_text, token_count, "token", True)"""
        # Define the tokenizer helper class for the splitting function
        tokenizer_config = TokenizerConfig(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=lambda t: self._tokenizer.encode(t),
        )

        # Get the raw text chunks based on token counts
        chunk_texts = self._split_text_on_tokens(text=text, tokenizer=tokenizer_config)

        # Convert to the required output format
        result = []
        for chunk_text in chunk_texts:
            token_count = self._tokenizer.count_tokens(chunk_text)
            result.append((self._doc_id, chunk_text, token_count, "token", True))

        return result

    def _split_text_on_tokens(self, *, text: str, tokenizer: TokenizerConfig) -> List[str]:
        """Split incoming text and return chunks using a sliding window of tokens."""
        splits: List[str] = []
        input_ids = tokenizer.encode(text)
        start_idx = 0
        while start_idx < len(input_ids):
            # Find the end of the current chunk
            end_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
            # Extract the token IDs for the chunk and decode them
            chunk_ids = input_ids[start_idx:end_idx]
            splits.append(tokenizer.decode(chunk_ids))
            # If we've reached the end, we're done
            if end_idx == len(input_ids):
                break
            # Move the start index forward for the next chunk, accounting for overlap
            start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        return splits


# --- Example Usage ---
if __name__ == "__main__":
    # Instantiate the chunker directly. It will use the Gemma tokenizer internally.
    chunker = FixedTokenChunker(
        doc_id=1,
        chunk_size=20, # Using a small chunk size to demonstrate overlap
        chunk_overlap=5  # Overlap of 5 tokens
    )

    # Test text
    test_text = (
        "Barack Hussein Obama II (born August 4, 1961) is an American politician who was the 44th president of the "
        "United States from 2009 to 2017. A member of the Democratic Party, he was the first African American "
        "president in American history."
    )

    print(f"--- Splitting text with FixedTokenChunker (using Gemma Tokenizer) ---")
    print(f"Original Text: '{test_text}'")
    print(f"Chunk Size: {chunker._chunk_size} tokens, Overlap: {chunker._chunk_overlap} tokens\n")

    # Get chunks in your required format
    chunks_with_metadata = chunker.split_text(test_text)

    print("--- Results ---")
    for i, chunk_data in enumerate(chunks_with_metadata):
        doc_id, chunk_text, token_count, chunk_type, _ = chunk_data
        print(f"Chunk { i +1}:")
        print(f"  Text: '{chunk_text}'")
        print(f"  Token Count: {token_count} (Doc ID: {doc_id}, Type: {chunk_type})")
        print("-" * 20)
