"""
Recursive Token Chunker.
This chunker recursively splits the text based on
 pre-defined separators and creates overlapping chunks.
"""

# External imports:
import re
from typing import List

# Internal imports:
from chunkers.base import BaseChunker

class RecursiveChunker(BaseChunker):
    """
    Initializes the recursive chunker class, which
    recursively splits the text based on seperators.
    """

    # Initialises the variables:
    def __init__(self, document_id: int, chunk_size: int = 400, chunk_overlap: int = 200):
        super().__init__(
            document_id,
            chunk_size,
            chunk_overlap
        )

        # Default separators used:
        self.separators = [
            "\n\n",  # Paragraphs
            "\n",  # Line breaks
            ". ",  # Sentences
            ", ",  # Clauses
            " ",  # Words
            ""  # Characters
        ]

    @staticmethod
    def split_by_separator(text: str, separator: str) -> List[str]:
        """Split text using a specified separator pattern"""
        if not text:
            return []

        # Regex-Pattern for splitting:
        # Example: text = "Hello. How are you?" => separator = ". " => splits = ['Hello', '. ', 'How are you?']
        pattern = f"({separator})"
        splits = re.split(pattern, text)

        # Process splits to maintain separators
        results = []
        for i in range(0, len(splits) - 1, 2): # Iterates only over the words (not separators)
            piece = splits[i]
            if i + 1 < len(splits):
                piece += splits[i + 1]  # Append the separator
            if piece:
                results.append(piece)

        # Handle the last piece if it exists
        if len(splits) % 2 == 1 and splits[-1]:
            results.append(splits[-1])

        return results if results else [text]

    def split_text(self, text: str) -> List[str]:
        """
        Entrypoint for splitting the text.
        """
        if not text.strip():
            return []

        # Get chunks using recursive splitting
        chunks = self.recursive_split(text, self.separators)

        # Apply overlap handling to create final chunks
        overlapping_chunks = self.create_overlapping_chunks(chunks)

        # Validate that all chunks are within the size limit and not empty
        validated_chunks = self.validate_chunks(overlapping_chunks)

        return validated_chunks

    def recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using multiple separators until chunks are small enough."""
        if not text.strip():
            return []

        # Base case: if we're at the last separator or text is small enough
        if len(separators) == 1 or self.count_tokens(text) <= self.chunk_size:
            return [text] if text else []

        # Find first separator that splits the text
        separator = None
        for sep in separators:
            if sep in text:
                separator = sep
                break

        # If no separator found, move to next level
        if not separator:
            return self.recursive_split(text, separators[1:])

        # Split using chosen separator
        splits = self.split_by_separator(text, separator)

        # Process each split
        final_chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = self.count_tokens(split)

            # Length validation: If this split alone exceeds chunk size, try splitting further
            if split_length > self.chunk_size:
                sub_chunks = self.recursive_split(split, separators[1:])
                if sub_chunks:  # Only add if non-empty
                    final_chunks.extend(sub_chunks)
            # If adding this split would exceed chunk size, start a new chunk
            elif current_length + split_length > self.chunk_size:
                if current_chunk:
                    combined_chunk = self.join_docs(current_chunk)
                    if combined_chunk.strip():  # Ensure chunk is not empty
                        final_chunks.append(combined_chunk)
                current_chunk = [split]
                current_length = split_length
            # The current chunk fits with the split
            else:
                current_chunk.append(split)
                current_length += split_length

        # Add any remaining text
        if current_chunk:
            combined_chunk = self.join_docs(current_chunk)
            if combined_chunk.strip():  # Ensure chunk is not empty
                final_chunks.append(combined_chunk)

        return final_chunks

    def create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """Create overlapping chunks based on chunk_overlap parameter"""
        if not chunks:
            return []

        if len(chunks) == 1:
            return chunks

        result = []

        for i in range(len(chunks)):
            if i == 0:
                # First chunk remains as is
                result.append(chunks[i])
            else:
                # Calculate overlap with previous chunk
                prev_chunk = chunks[i-1]
                current_chunk = chunks[i]

                # Get tokens from previous chunk to create overlap
                prev_tokens = self.tokenize(prev_chunk)
                overlap_token_count = min(self.chunk_overlap, len(prev_tokens))

                if overlap_token_count > 0:
                    # Get the last N tokens from previous chunk
                    overlap_tokens = prev_tokens[-overlap_token_count:]
                    overlap_text = self.detokenize(overlap_tokens)

                    # Only add overlap if it doesn't make the chunk too large
                    if self.count_tokens(overlap_text + current_chunk) <= self.chunk_size:
                        current_chunk = overlap_text + current_chunk

                result.append(current_chunk)

        return result

    def validate_chunks(self, chunks: List[str]) -> List[str]:
        """
        Validates chunks to ensure they are:
        1. Not empty or containing only meaningless characters
        2. Not exceeding maximum token size
        3. Have sufficient content to be useful
        """
        validated = []
        MIN_CHUNK_LENGTH = 10  # Minimum characters for a meaningful chunk
        MIN_TOKENS = 5         # Minimum tokens for a meaningful chunk

        for chunk in chunks:
            # Skip empty or whitespace-only chunks
            if not chunk.strip():
                continue

            # Skip chunks that are too short or just punctuation/special characters
            cleaned_content = re.sub(r'[^\w\s]', '', chunk).strip()
            if len(cleaned_content) < MIN_CHUNK_LENGTH or len(chunk) < MIN_CHUNK_LENGTH:
                continue

            # Check token count (skip chunks with too few tokens)
            chunk_tokens = self.count_tokens(chunk)
            if chunk_tokens < MIN_TOKENS:
                continue

            # Check if chunk is too large
            if chunk_tokens > self.chunk_size:
                # If too large, try to split it at a lower level
                for separator in self.separators[2:]:  # Start with sentence-level separators
                    if separator in chunk:
                        smaller_chunks = self.split_by_separator(chunk, separator)
                        smaller_processed = []

                        for small_chunk in smaller_chunks:
                            # Skip small chunks that don't meet minimum requirements
                            if not small_chunk.strip():
                                continue

                            cleaned_small = re.sub(r'[^\w\s]', '', small_chunk).strip()
                            if len(cleaned_small) < MIN_CHUNK_LENGTH or len(small_chunk) < MIN_CHUNK_LENGTH:
                                continue

                            small_tokens = self.count_tokens(small_chunk)
                            if small_tokens < MIN_TOKENS:
                                continue

                            # Accept valid small chunks
                            if small_tokens <= self.chunk_size:
                                smaller_processed.append(small_chunk)
                            else:
                                # Try character-level splitting as last resort, with size validation
                                for i in range(0, len(small_chunk), self.chunk_size // 4):
                                    fragment = small_chunk[i:i + self.chunk_size // 4]
                                    if fragment.strip() and len(fragment) >= MIN_CHUNK_LENGTH:
                                        fragment_tokens = self.count_tokens(fragment)
                                        if fragment_tokens >= MIN_TOKENS:
                                            smaller_processed.append(fragment)

                        if smaller_processed:
                            validated.extend(smaller_processed)
                            break
                else:
                    # If we couldn't split further, truncate as a last resort
                    tokens = self.tokenize(chunk)[:self.chunk_size]
                    truncated = self.detokenize(tokens)
                    if truncated.strip() and len(truncated) >= MIN_CHUNK_LENGTH:
                        truncated_tokens = self.count_tokens(truncated)
                        if truncated_tokens >= MIN_TOKENS:
                            validated.append(truncated)
            else:
                validated.append(chunk)

        return validated

    def process_document(self, document: str) -> List[tuple]:
        """Process text into recursive token-based chunks with overlap."""
        if not document.strip():
            return []

        final_chunks = []
        text_chunks = self.split_text(document)

        for i, chunk_text in enumerate(text_chunks):
            # Skip empty chunks
            if not chunk_text.strip():
                continue

            # Calculate token count
            token_count = self.count_tokens(chunk_text)

            # Skip chunks that are too large (this should not happen with proper validation)
            if token_count > self.chunk_size:
                print(f"Warning: Chunk {i} exceeds size limit with {token_count} tokens.")
                continue

            chunk = (self.document_id, chunk_text, token_count, "recursive", True)
            final_chunks.append(chunk)

        return final_chunks