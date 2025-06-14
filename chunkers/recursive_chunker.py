# This script is adapted from the LangChain package, developed by LangChain AI.
# Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py
# License: MIT License
"""
Recursive Token Chunker with Position Tracking

This chunker recursively tries to split text using a list of separators
in order of preference (from coarse to fine-grained). If a separator
results in chunks that are still too large, it recurses with the next
separator in the list.

Enhanced with position tracking capabilities to map chunks back to original document positions.
Ideal for maintaining semantic boundaries while respecting token limits.
"""

from typing import Any, List, Optional, Tuple
from chunkers.utils import Language
from chunkers.token_chunker import TextSplitter
from emb.gemma_tokenizer import GemmaSingletonTokenizer
import re
import logging
import difflib

logger = logging.getLogger(__name__)


def split_text_with_regex(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    """
    Split text using regex with option to keep or remove separators.

    Args:
        text: The input text to split
        separator: The regex pattern to split on
        keep_separator: Whether to keep the separator in the resulting chunks

    Returns:
        List of text segments after splitting
    """
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


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
        if self.start_index is not None and self.end_index is not None:
            return (self.doc_id, self.text, self.token_count, self.chunk_type,
                   self.is_processed, self.start_index, self.end_index)
        else:
            return (self.doc_id, self.text, self.token_count, self.chunk_type,
                   self.is_processed)


class RecursiveTokenChunker(TextSplitter):
    """
    Recursive text splitter that tries different separators hierarchically.

    This chunker recursively tries to split text using a list of separators
    in order of preference (from coarse to fine-grained). If a separator
    results in chunks that are still too large, it recurses with the next
    separator in the list.

    Ideal for maintaining semantic boundaries while respecting token limits.
    """

    def __init__(
            self,
            doc_id: int,  # Added doc_id parameter for tracking
            chunk_size: int = 400,
            chunk_overlap: int = 100,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = False,
            **kwargs: Any,
    ) -> None:
        """
        Create a new RecursiveTokenChunker.

        Args:
            doc_id: Document ID for tracking chunks
            chunk_size: Maximum size of chunks to return (in tokens)
            chunk_overlap: Overlap in tokens between chunks for context preservation
            separators: List of separators to try in descending order of preference
            keep_separator: Whether to keep the separator in the chunks
            is_separator_regex: Whether to treat separators as regex patterns
            **kwargs: Additional arguments passed to parent TextSplitter
        """
        super().__init__(
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )

        # Default separators: try paragraphs, then sentences, then words, then characters
        self._separators = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]
        self._is_separator_regex = is_separator_regex
        self._keep_separator = keep_separator

        # Initialize Gemma tokenizer for token counting
        self._tokenizer = GemmaSingletonTokenizer()

        # Set length function to use token counting
        self._length_function = self._tokenizer.count_tokens

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the provided list of separators.

        This is the core recursive algorithm that:
        1. Finds the first separator that exists in the text
        2. Splits the text using that separator
        3. For chunks that are still too large, recursively splits with finer separators
        4. Merges smaller chunks together with overlap

        Args:
            text: Input text to be split
            separators: List of separators, ordered from coarse to fine-grained

        Returns:
            List of text chunks that respect size limits
        """
        final_chunks = []

        # Find the first separator that actually exists in the text
        separator = separators[-1]  # Default to last (finest) separator
        new_separators = []

        for i, _s in enumerate(separators):
            # Escape separator for regex unless it's already a regex pattern
            _separator = _s if self._is_separator_regex else re.escape(_s)

            # Empty string means split into individual characters (last resort)
            if _s == "":
                separator = _s
                break

            # Check if this separator exists in the text
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]  # Remaining separators for recursion
                break

        # Perform the actual text splitting
        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = split_text_with_regex(text, _separator, self._keep_separator)

        # Process each split: merge small ones, recursively split large ones
        _good_splits = []  # Accumulator for chunks that fit within size limit
        _separator = "" if self._keep_separator else separator

        for s in splits:
            # If this split fits within our chunk size, add it to the accumulator
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                # Current split is too large
                # First, process any accumulated good splits
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []  # Reset accumulator

                # Handle the oversized split
                if not new_separators:
                    # No more separators to try, add as-is (will trigger warning)
                    final_chunks.append(s)
                else:
                    # Recursively split with finer-grained separators
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

        # Process any remaining good splits
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)

        return final_chunks

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        """Join documents with separator and strip whitespace."""
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge splits into chunks respecting size and overlap constraints.

        This method combines smaller text segments into larger chunks while
        ensuring they don't exceed the maximum chunk size and maintaining
        the specified overlap between consecutive chunks.

        Args:
            splits: List of text segments to merge
            separator: String to use when joining segments

        Returns:
            List of merged text chunks
        """
        separator_len = self._length_function(separator)
        docs = []
        current_doc: List[str] = []
        total = 0

        for d in splits:
            _len = self._length_function(d)

            # Check if adding this split would exceed chunk size
            if (
                    total + _len + (separator_len if len(current_doc) > 0 else 0)
                    > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )

                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)

                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                            total + _len + (separator_len if len(current_doc) > 0 else 0)
                            > self._chunk_size
                            and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)

        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)

        return docs

    def _find_best_match_position(self, text: str, chunk_text: str, start_pos: int) -> int:
        """
        Find the best matching position for a chunk, handling minor differences.

        This method helps locate chunks even when there are small differences
        due to text processing, using fuzzy matching with a high similarity threshold.

        Args:
            text: Original text to search in
            chunk_text: Chunk text to find
            start_pos: Starting position for search

        Returns:
            Best matching position, or -1 if no good match found
        """
        # Try to find a substring that's very similar
        chunk_len = len(chunk_text)
        best_pos = -1
        best_ratio = 0

        # Search in a reasonable window
        search_end = min(len(text), start_pos + len(text) // 4)

        for pos in range(start_pos, search_end - chunk_len + 1):
            candidate = text[pos:pos + chunk_len]
            ratio = difflib.SequenceMatcher(None, chunk_text, candidate).ratio()

            if ratio > best_ratio and ratio > 0.9:  # High similarity threshold
                best_ratio = ratio
                best_pos = pos

        return best_pos

    def split_text_with_positions(self, text: str) -> List[ChunkWithPosition]:
        """
        Split text and return chunks with their original positions tracked.

        This enhanced method tracks the exact position of each chunk in the original
        text, enabling precise mapping for downstream processing.

        Args:
            text: Input text to be chunked

        Returns:
            List of ChunkWithPosition objects with start/end indices
        """
        # Get raw text chunks using the recursive algorithm
        chunk_texts = self._split_text(text, self._separators)

        # Track positions by finding each chunk in the original text
        result = []
        search_start = 0

        for chunk_text in chunk_texts:
            # Find this chunk's position in the original text
            chunk_start = text.find(chunk_text, search_start)

            if chunk_start == -1:
                # If exact match fails, try to find the best match
                # This handles cases where chunking may have modified whitespace
                chunk_start = self._find_best_match_position(text, chunk_text, search_start)

            if chunk_start != -1:
                chunk_end = chunk_start + len(chunk_text)
                search_start = chunk_start + 1  # Start next search after this chunk

                token_count = self._tokenizer.count_tokens(chunk_text)
                chunk_obj = ChunkWithPosition(
                    doc_id=self._doc_id,
                    text=chunk_text,
                    token_count=token_count,
                    chunk_type="recursive",
                    is_processed=True,
                    start_index=chunk_start,
                    end_index=chunk_end
                )
                result.append(chunk_obj)
            else:
                # Fallback: create chunk without position info
                logger.warning(f"Could not find position for chunk: {chunk_text[:50]}...")
                token_count = self._tokenizer.count_tokens(chunk_text)
                chunk_obj = ChunkWithPosition(
                    doc_id=self._doc_id,
                    text=chunk_text,
                    token_count=token_count,
                    chunk_type="recursive",
                    is_processed=True
                )
                result.append(chunk_obj)

        return result

    def split_text(self, text: str) -> List[tuple]:
        """
        Split text and return tuples in the required format with position tracking.

        This is the main public interface that returns chunks in the format:
        (doc_id, chunk_text, token_count, "recursive", True, start_index, end_index)

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

        logger.info(f"Created {len(result)} recursive chunks with position tracking")
        return result

    @classmethod
    def from_language(
            cls,
            doc_id: int,
            language: Language,
            chunk_size: int = 400,
            chunk_overlap: int = 100,
            **kwargs: Any,
    ) -> "RecursiveTokenChunker":
        """
        Create a RecursiveTokenChunker with language-specific separators.

        This factory method provides language-specific separators for better
        semantic chunking of code and markup.

        Args:
            doc_id: Document ID for tracking
            language: Programming language for automatic separator selection
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            **kwargs: Additional arguments for the chunker

        Returns:
            Configured RecursiveTokenChunker instance
        """
        separators = cls.get_separators_for_language(language)
        return cls(
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            is_separator_regex=True,
            **kwargs
        )

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        """
        Get language-specific separators for better semantic chunking.

        Different programming languages and markup formats have different
        logical boundaries. This method returns separators optimized for
        each language to maintain semantic coherence.

        Args:
            language: The programming/markup language

        Returns:
            List of separators ordered from coarse to fine-grained
        """
        if language == Language.CPP:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.GO:
            return [
                # Split along function definitions
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JAVA:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PYTHON:
            return [
                # First, try to split along class definitions
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                # Now split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JS:
            return [
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.MARKDOWN:
            return [
                # First, try to split along Markdown headings (starting with level 2)
                "\n#{1,6} ",
                # End of code block
                "```\n",
                # Horizontal lines
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # Split by paragraphs and lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        # Add other languages as needed...
        else:
            # For unsupported languages, use general text separators
            return ["\n\n", "\n", ".", "?", "!", " ", ""]


# Enhanced version with position tracking for use by cluster chunker
class PositionTrackingRecursiveChunker(RecursiveTokenChunker):
    """
    Extended recursive chunker that tracks original positions in the text.

    This specialized version is designed for use by the cluster chunker to maintain
    precise position tracking through the semantic clustering process.
    """

    def split_text_with_positions(self, text: str) -> List[ChunkWithPosition]:
        """
        Split text and return chunks with their original positions tracked.

        This method is the main interface for position-aware chunking, used
        by downstream chunkers that need to track original text positions.

        Args:
            text: Input text to be chunked

        Returns:
            List of ChunkWithPosition objects with start/end indices
        """
        return super().split_text_with_positions(text)


# Example usage demonstrating both general and language-specific chunking
if __name__ == "__main__":
    # Option 1: General text chunking
    general_chunker = RecursiveTokenChunker(
        doc_id=1,
        chunk_size=100,  # Small size for demonstration
        chunk_overlap=0
    )

    # Test with different types of content
    general_text = """Good evening. If I were smart, I'd go home now. Mr. Speaker, Madam Vice President, 
    members of Congress, my fellow Americans. In January 1941, Franklin Roosevelt came 
    to this chamber to speak to the nation. And he said, "I address you at a moment 
    unprecedented in the history of the Union". Hitler was on the march. War was raging 
    in Europe. President Roosevelt's purpose was to wake up Congress and alert the 
    American people that this was no ordinary time.
    """

    # Test general text chunking with positions
    print("=== General Text Chunking with Positions ===")
    general_chunks = general_chunker.split_text(general_text)
    for i, chunk_data in enumerate(general_chunks):
        if len(chunk_data) == 7:  # With position info
            doc_id, chunk_text, token_count, chunk_type, is_processed, start_idx, end_idx = chunk_data
            print(f"Chunk {i + 1}: Doc {doc_id}, {token_count} tokens, Type: {chunk_type}")
            print(f"Position: {start_idx}-{end_idx}")
            print(f"Text: {chunk_text[:100]}...")
            print(f"Verification: {repr(general_text[start_idx:end_idx][:50])}...")
        else:  # Fallback without position info
            doc_id, chunk_text, token_count, chunk_type, is_processed = chunk_data
            print(f"Chunk {i + 1}: Doc {doc_id}, {token_count} tokens, Type: {chunk_type}")
            print(f"Text: {chunk_text[:100]}... (no position info)")
        print("-" * 40)

    # Option 2: Position tracking chunker
    print("\n=== Position Tracking Chunking ===")
    position_chunker = PositionTrackingRecursiveChunker(
        doc_id=2,
        chunk_size=100,
        chunk_overlap=0
    )

    position_chunks = position_chunker.split_text_with_positions(general_text)
    for i, chunk in enumerate(position_chunks):
        print(f"Chunk {i + 1}: Doc {chunk.doc_id}, {chunk.token_count} tokens")
        if chunk.start_index is not None:
            print(f"Position: {chunk.start_index}-{chunk.end_index}")
            print(f"Verification: {repr(general_text[chunk.start_index:chunk.end_index][:50])}...")
        print(f"Text: {chunk.text[:50]}...")
        print("-" * 40)