# This script is adapted from the LangChain package, developed by LangChain AI.
# Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py
# License: MIT License
# It has been further adjusted/optimised to fit the chunking method needed for this application
"""
Token Chunker with Position Tracking

This chunker splits text based on token counts using a sliding window approach
with configurable overlap. Enhanced with position tracking to map chunks back
to their exact locations in the original document.

The chunker uses the Gemma tokenizer for consistent token counting and splitting,
ensuring chunks respect token boundaries while maintaining semantic coherence
through overlapping windows.
"""

# External imports:
from abc import ABC, abstractmethod
from attr import dataclass
import logging
from typing import (
    Callable,
    List,
    TypeVar,
    Tuple,
    Optional,
)
import difflib

# Internal imports:
from emb.gemma_tokenizer import GemmaSingletonTokenizer
from chunkers.base import BaseChunker

# Configuring logging:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
TS = TypeVar("TS", bound="TextSplitter")


class ChunkWithPosition:
    """Helper class to store chunk data with original position information."""
    
    def __init__(self, doc_id: int, text: str, token_count: int, chunk_type: str, 
                 is_processed: bool, start_index: int = None, end_index: int = None):
        self.doc_id = doc_id
        self.text = text
        self.token_count = token_count
        self.chunk_type = chunk_type
        self.is_processed = is_processed
        self.start_index = start_index
        self.end_index = end_index
    
    def to_tuple(self) -> Tuple:
        """Convert to the expected tuple format with position info."""
        return (self.doc_id, self.text, self.token_count, self.chunk_type, 
               self.is_processed, self.start_index, self.end_index)


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
    """
    Splitting text to tokens using the Gemma tokenizer with position tracking.
    
    This chunker creates fixed-size chunks based on token counts, using a sliding
    window approach with configurable overlap. Each chunk's position in the original
    document is tracked for precise mapping.
    
    The chunker ensures that:
    - Chunks respect token boundaries
    - Overlapping provides context continuity
    - Position tracking enables accurate document mapping
    - Output format is consistent with other chunkers
    """
    
    def __init__(
            self,
            doc_id: int,
            chunk_size: int = 400,
            chunk_overlap: int = 200,
    ) -> None:
        """
        Create a new FixedTokenChunker with position tracking.
        
        Args:
            doc_id: Document ID for tracking chunks
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between consecutive chunks
        """
        super().__init__(doc_id=doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Get the singleton instance of our Gemma tokenizer
        self._tokenizer = GemmaSingletonTokenizer()
        
        logger.info(f"Initialized FixedTokenChunker with position tracking - chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def _find_chunk_position(self, original_text: str, chunk_text: str, search_start: int = 0) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the position of a chunk in the original text with fuzzy matching.
        
        This method handles cases where token decoding might introduce minor
        differences from the original text, using similarity-based matching.
        
        Args:
            original_text: The full original text
            chunk_text: The chunk text to locate
            search_start: Position to start searching from
            
        Returns:
            Tuple of (start_index, end_index) or (None, None) if not found
        """
        # First try exact match
        chunk_start = original_text.find(chunk_text, search_start)
        if chunk_start != -1:
            return chunk_start, chunk_start + len(chunk_text)
        
        # If exact match fails, try fuzzy matching
        # This handles cases where tokenization/detokenization changes text slightly
        chunk_len = len(chunk_text)
        best_pos = None
        best_ratio = 0
        
        # Search in a reasonable window
        search_end = min(len(original_text), search_start + len(original_text) // 2)
        
        # Try different positions with sliding window
        for pos in range(search_start, max(search_start + 1, search_end - chunk_len + 1)):
            if pos + chunk_len > len(original_text):
                break
                
            candidate = original_text[pos:pos + chunk_len]
            ratio = difflib.SequenceMatcher(None, chunk_text.strip(), candidate.strip()).ratio()
            
            if ratio > best_ratio and ratio > 0.8:  # High similarity threshold
                best_ratio = ratio
                best_pos = pos
        
        if best_pos is not None:
            return best_pos, best_pos + chunk_len
        
        # Last resort: try to find a substantial substring
        words = chunk_text.split()[:5]  # First 5 words
        if words:
            search_phrase = ' '.join(words)
            phrase_start = original_text.find(search_phrase, search_start)
            if phrase_start != -1:
                # Estimate the full chunk position based on the phrase
                estimated_end = min(len(original_text), phrase_start + len(chunk_text))
                return phrase_start, estimated_end
        
        return None, None

    def split_text_with_positions(self, text: str) -> List[ChunkWithPosition]:
        """
        Split text and return chunks with their original positions tracked.
        
        This method creates token-based chunks while maintaining precise position
        information for each chunk in the original document.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of ChunkWithPosition objects with start/end indices
        """
        # Define the tokenizer helper class for the splitting function
        tokenizer_config = TokenizerConfig(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=lambda t: self._tokenizer.encode(t),
        )

        # Get the raw text chunks based on token counts
        chunk_texts = self._split_text_on_tokens(text=text, tokenizer=tokenizer_config)

        # Track positions for each chunk
        result = []
        search_start = 0
        
        for i, chunk_text in enumerate(chunk_texts):
            # Calculate token count
            token_count = self._tokenizer.count_tokens(chunk_text)
            
            # Find position in original text
            start_idx, end_idx = self._find_chunk_position(text, chunk_text, search_start)
            
            if start_idx is not None:
                # Update search start for next chunk, accounting for overlap
                # Don't go backwards to handle overlapping chunks
                search_start = max(search_start + 1, start_idx + len(chunk_text) // 2)
            else:
                logger.warning(f"Could not find position for token chunk {i+1}: {chunk_text[:50]}...")
            
            # Create chunk with position info
            chunk_obj = ChunkWithPosition(
                doc_id=self._doc_id,
                text=chunk_text,
                token_count=token_count,
                chunk_type="token",
                is_processed=True,
                start_index=start_idx,
                end_index=end_idx
            )
            result.append(chunk_obj)
        
        logger.info(f"Created {len(result)} token chunks with position tracking")
        return result

    def split_text(self, text: str) -> List[tuple]:
        """
        Split text and return tuples in the consistent format with position tracking.
        
        This is the main public interface that returns chunks in the format:
        (doc_id, chunk_text, token_count, "token", True, start_index, end_index)
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of tuples containing chunk information with position data
        """
        # Use position tracking for consistent output format
        chunks_with_positions = self.split_text_with_positions(text)
        
        # Convert to the required output format with positions
        result = []
        for chunk in chunks_with_positions:
            result.append(chunk.to_tuple())

        logger.info(f"Created {len(result)} token chunks with position tracking")
        return result

    def _split_text_on_tokens(self, *, text: str, tokenizer: TokenizerConfig) -> List[str]:
        """
        Split incoming text and return chunks using a sliding window of tokens.
        
        This method implements the core token-based splitting algorithm using
        a sliding window approach with configurable overlap.
        
        Args:
            text: Input text to split
            tokenizer: Tokenizer configuration with encode/decode functions
            
        Returns:
            List of text chunks based on token boundaries
        """
        splits: List[str] = []
        input_ids = tokenizer.encode(text)
        start_idx = 0
        
        while start_idx < len(input_ids):
            # Find the end of the current chunk
            end_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
            
            # Extract the token IDs for the chunk and decode them
            chunk_ids = input_ids[start_idx:end_idx]
            chunk_text = tokenizer.decode(chunk_ids)
            splits.append(chunk_text)
            
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
        chunk_size=20,  # Using a small chunk size to demonstrate overlap
        chunk_overlap=5  # Overlap of 5 tokens
    )

    # Test text
    test_text = (
        "Barack Hussein Obama II (born August 4, 1961) is an American politician who was the 44th president of the "
        "United States from 2009 to 2017. A member of the Democratic Party, he was the first African American "
        "president in American history. Prior to his presidency, he served as a United States senator from Illinois "
        "from 2005 to 2008 and as an Illinois state senator from 1997 to 2004."
    )

    print(f"--- Splitting text with FixedTokenChunker (using Gemma Tokenizer with Position Tracking) ---")
    print(f"Original Text: '{test_text}'")
    print(f"Chunk Size: {chunker._chunk_size} tokens, Overlap: {chunker._chunk_overlap} tokens\n")

    # Get chunks in the consistent format with position tracking
    chunks_with_metadata = chunker.split_text(test_text)

    print("--- Results ---")
    for i, chunk_data in enumerate(chunks_with_metadata):
        # All chunkers now return consistent 7-element tuples
        doc_id, chunk_text, token_count, chunk_type, is_processed, start_idx, end_idx = chunk_data
        print(f"Chunk {i + 1}:")
        print(f"  Doc ID: {doc_id}")
        print(f"  Token Count: {token_count}")
        print(f"  Chunk Type: {chunk_type}")
        print(f"  Is Processed: {is_processed}")
        
        if start_idx is not None and end_idx is not None:
            print(f"  Start Index: {start_idx}")
            print(f"  End Index: {end_idx}")
            print(f"  Text: '{chunk_text}'")
            print(f"  Verification: '{test_text[start_idx:end_idx]}'")
        else:
            print(f"  Position: Not available")
            print(f"  Text: '{chunk_text}'")
            
        print("-" * 40)
    
    print(f"\nSummary:")
    print(f"Total chunks: {len(chunks_with_metadata)}")
    total_tokens = sum(chunk_data[2] for chunk_data in chunks_with_metadata)
    print(f"Total tokens: {total_tokens}")
    avg_tokens = total_tokens / len(chunks_with_metadata) if chunks_with_metadata else 0
    print(f"Average tokens per chunk: {avg_tokens:.1f}")
    
    # Verify position tracking accuracy
    print(f"\nPosition Tracking Verification:")
    for i, chunk_data in enumerate(chunks_with_metadata):
        doc_id, chunk_text, token_count, chunk_type, is_processed, start_idx, end_idx = chunk_data
        
        if start_idx is not None and end_idx is not None:
            original_text = test_text[start_idx:end_idx]
            
            # Check if the positions accurately map back to original text
            similarity = difflib.SequenceMatcher(None, chunk_text.strip(), original_text.strip()).ratio()
            if similarity > 0.8:
                print(f"✓ Chunk {i+1}: Position mapping verified (similarity: {similarity:.2f})")
            else:
                print(f"✗ Chunk {i+1}: Position mapping issue detected (similarity: {similarity:.2f})")
                print(f"  Chunk text: '{chunk_text[:50]}...'")
                print(f"  Original text: '{original_text[:50]}...'")
        else:
            print(f"⚠ Chunk {i+1}: No position information available")