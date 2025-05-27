"""
Robust Recursive Token Chunker - Fixed Version
This chunker recursively splits text based on pre-defined separators
and creates precise token-based overlapping chunks with proper validation.
Fixed to respect chunk_overlap=0 setting.
"""

# External imports:
import re
from typing import List, Tuple

# Internal imports:
from chunkers.base import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Robust recursive chunker that ensures precise token-based overlap
    and proper chunk size validation.
    """

    def __init__(self, document_id: int, chunk_size: int = 400, chunk_overlap: int = 200):
        super().__init__(document_id, chunk_size, chunk_overlap)

        # Validate overlap ratio
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

        # Calculate maximum effective chunk size after overlap
        self.max_effective_chunk_size = chunk_size
        self.min_chunk_tokens = max(5, chunk_overlap // 4) if chunk_overlap > 0 else 5  # Minimum meaningful chunk size

        # Hierarchical separators for better text splitting
        self.separators = [
            "\n\n\n",  # Section breaks
            "\n\n",  # Paragraphs
            "\n",  # Line breaks
            ". ",  # Sentences
            "! ",  # Exclamations
            "? ",  # Questions
            "; ",  # Semicolons
            ", ",  # Clauses
            " ",  # Words
            ""  # Characters (fallback)
        ]

    def split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text using separator while preserving the separator and natural boundaries."""
        if not text or not text.strip():
            return []

        # Handle character-level splitting
        if separator == "":
            return list(text)

        # If separator not in text, return the whole text
        if separator not in text:
            return [text]

        # Split and preserve separators
        parts = text.split(separator)
        results = []

        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Last part - no separator to add
                if part:
                    results.append(part)
            else:
                # Add separator back to maintain original text structure
                if part or separator.strip():  # Keep empty parts if separator has meaning
                    results.append(part + separator)

        # Filter out completely empty results
        return [r for r in results if r]

    def get_token_boundaries(self, text: str, max_tokens: int) -> Tuple[str, str]:
        """
        Split text at token boundary, returning (first_part, remaining_part).
        Ensures the first part doesn't exceed max_tokens.
        """
        if not text:
            return "", ""

        tokens = self.tokenize(text)

        if len(tokens) <= max_tokens:
            return text, ""

        # Find the split point
        split_tokens = tokens[:max_tokens]
        remaining_tokens = tokens[max_tokens:]

        # Convert back to text
        first_part = self.detokenize(split_tokens)
        remaining_part = self.detokenize(remaining_tokens)

        return first_part, remaining_part

    def create_token_overlap(self, prev_chunk: str, current_chunk: str) -> str:
        """
        Create overlap by taking exact token count from previous chunk
        and prepending to current chunk, ensuring size limits.
        Only applies overlap if chunk_overlap > 0.
        """
        # FIXED: Check if overlap is disabled
        if self.chunk_overlap == 0:
            return current_chunk

        if not prev_chunk or not current_chunk:
            return current_chunk

        # Get tokens from previous chunk for overlap
        prev_tokens = self.tokenize(prev_chunk)

        if len(prev_tokens) <= self.chunk_overlap:
            # If previous chunk is smaller than overlap, use entire chunk
            overlap_tokens = prev_tokens
        else:
            # Take last N tokens for overlap
            overlap_tokens = prev_tokens[-self.chunk_overlap:]

        # Convert overlap tokens back to text
        overlap_text = self.detokenize(overlap_tokens)

        # Combine overlap with current chunk
        combined_text = overlap_text + current_chunk

        # Ensure combined chunk doesn't exceed size limit
        combined_tokens = self.tokenize(combined_text)

        if len(combined_tokens) <= self.chunk_size:
            return combined_text
        else:
            # Truncate current chunk to fit within size limit
            available_tokens = self.chunk_size - len(overlap_tokens)
            if available_tokens > 0:
                current_tokens = self.tokenize(current_chunk)
                truncated_current_tokens = current_tokens[:available_tokens]
                truncated_current = self.detokenize(truncated_current_tokens)
                return overlap_text + truncated_current
            else:
                # If overlap is too large, just return current chunk truncated
                current_tokens = self.tokenize(current_chunk)
                if len(current_tokens) > self.chunk_size:
                    truncated_tokens = current_tokens[:self.chunk_size]
                    return self.detokenize(truncated_tokens)
                return current_chunk

    def recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchical separators."""
        if not text or not text.strip():
            return []

        current_tokens = self.count_tokens(text)

        # Base case: text fits within chunk size
        if current_tokens <= self.chunk_size:
            return [text] if text.strip() else []

        # Find appropriate separator
        separator = None
        for sep in separators:
            if sep in text and sep != "":
                separator = sep
                break

        # If no separator found and we're not at character level, try next level
        if not separator and len(separators) > 1:
            return self.recursive_split(text, separators[1:])

        # If we're at character level or found separator, proceed with splitting
        if separator == "" or separator:
            splits = self.split_by_separator(text, separator or "")
        else:
            # Fallback: force split at token boundary
            first_part, remaining = self.get_token_boundaries(text, self.chunk_size)
            splits = [first_part, remaining] if remaining else [first_part]

        # Process splits
        final_chunks = []
        current_chunk_parts = []
        current_chunk_tokens = 0

        for split in splits:
            if not split.strip():
                continue

            split_tokens = self.count_tokens(split)

            # If single split is too large, recursively split it further
            if split_tokens > self.chunk_size:
                # Save current accumulated chunk
                if current_chunk_parts:
                    chunk_text = "".join(current_chunk_parts)
                    if chunk_text.strip():
                        final_chunks.append(chunk_text)
                    current_chunk_parts = []
                    current_chunk_tokens = 0

                # Recursively split the large piece
                if len(separators) > 1:
                    sub_chunks = self.recursive_split(split, separators[1:])
                else:
                    # Force split at token boundary
                    sub_chunks = []
                    remaining_text = split
                    while remaining_text and self.count_tokens(remaining_text) > self.chunk_size:
                        part, remaining_text = self.get_token_boundaries(remaining_text, self.chunk_size)
                        if part.strip():
                            sub_chunks.append(part)
                    if remaining_text and remaining_text.strip():
                        sub_chunks.append(remaining_text)

                final_chunks.extend([chunk for chunk in sub_chunks if chunk.strip()])

            # If adding this split would exceed chunk size, finalize current chunk
            elif current_chunk_tokens + split_tokens > self.chunk_size:
                if current_chunk_parts:
                    chunk_text = "".join(current_chunk_parts)
                    if chunk_text.strip():
                        final_chunks.append(chunk_text)

                current_chunk_parts = [split]
                current_chunk_tokens = split_tokens

            # Add split to current chunk
            else:
                current_chunk_parts.append(split)
                current_chunk_tokens += split_tokens

        # Add any remaining chunk
        if current_chunk_parts:
            chunk_text = "".join(current_chunk_parts)
            if chunk_text.strip():
                final_chunks.append(chunk_text)

        return final_chunks

    def create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """Create overlapping chunks with precise token-based overlap."""
        if not chunks:
            return []

        if len(chunks) == 1:
            return chunks

        # FIXED: Skip overlap creation if overlap is disabled
        if self.chunk_overlap == 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no overlap needed
                overlapped_chunks.append(chunk)
            else:
                # Create overlap with previous chunk
                prev_chunk = chunks[i - 1]
                overlapped_chunk = self.create_token_overlap(prev_chunk, chunk)
                overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def validate_chunks(self, chunks: List[str]) -> List[str]:
        """Validate chunks for size, content quality, and token limits."""
        validated = []

        for chunk in chunks:
            # Skip empty or whitespace-only chunks
            if not chunk or not chunk.strip():
                continue

            chunk = chunk.strip()
            chunk_tokens = self.count_tokens(chunk)

            # Skip chunks that are too small to be meaningful
            if chunk_tokens < self.min_chunk_tokens:
                continue

            # Handle chunks that are too large (shouldn't happen with proper splitting)
            if chunk_tokens > self.chunk_size:
                # Try to truncate at sentence boundary
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                truncated = ""

                for sentence in sentences:
                    test_chunk = truncated + (" " if truncated else "") + sentence
                    if self.count_tokens(test_chunk) <= self.chunk_size:
                        truncated = test_chunk
                    else:
                        break

                if truncated and self.count_tokens(truncated) >= self.min_chunk_tokens:
                    validated.append(truncated)
                else:
                    # Force truncate at token boundary as last resort
                    tokens = self.tokenize(chunk)
                    if len(tokens) > self.chunk_size:
                        truncated_tokens = tokens[:self.chunk_size]
                        truncated_text = self.detokenize(truncated_tokens)
                        if self.count_tokens(truncated_text) >= self.min_chunk_tokens:
                            validated.append(truncated_text)
            else:
                validated.append(chunk)

        return validated

    def split_text(self, text: str) -> List[str]:
        """Main entry point for splitting text into robust chunks."""
        if not text or not text.strip():
            return []

        # Step 1: Recursive splitting
        chunks = self.recursive_split(text, self.separators)

        # Step 2: Apply token-based overlap (only if chunk_overlap > 0)
        overlapping_chunks = self.create_overlapping_chunks(chunks)

        # Step 3: Validate all chunks
        validated_chunks = self.validate_chunks(overlapping_chunks)

        return validated_chunks

    def process_document(self, document: str) -> List[Tuple]:
        """Process document into robust token-based chunks with precise overlap."""
        if not document or not document.strip():
            return []

        final_chunks = []
        text_chunks = self.split_text(document)

        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue

            token_count = self.count_tokens(chunk_text)

            # Final validation
            if token_count > self.chunk_size:
                print(f"Warning: Chunk {i} still exceeds size limit with {token_count} tokens after validation.")
                continue

            if token_count < self.min_chunk_tokens:
                print(f"Warning: Chunk {i} is too small with {token_count} tokens.")
                continue

            chunk = (self.document_id, chunk_text, token_count, "robust_recursive", True)
            final_chunks.append(chunk)

        return final_chunks


# Example usage and testing
if __name__ == '__main__':
    test_text = """Born in Honolulu, Hawaii, Obama graduated from Columbia University in 1983 with a Bachelor of Arts degree in political science and later worked as a community organizer in Chicago. In 1988, Obama enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review. He became a civil rights attorney and an academic, teaching constitutional law at the University of Chicago Law School from 1992 to 2004. In 1996, Obama was elected to represent the 13th district in the Illinois Senate, a position he held until 2004, when he successfully ran for the U.S. Senate. In the 2008 presidential election, after a close primary campaign against Hillary Clinton, he was nominated by the Democratic Party for president. Obama selected Joe Biden as his running mate and defeated Republican nominee John McCain and his running mate Sarah Palin.

Obama was awarded the 2009 Nobel Peace Prize for efforts in international diplomacy, a decision which drew both criticism and praise. During his first term, his administration responded to the 2008 financial crisis with measures including the American Recovery and Reinvestment Act of 2009, a major stimulus package to guide the economy in recovering from the Great Recession; a partial extension of the Bush tax cuts; legislation to reform health care; and the Dodd–Frank Wall Street Reform and Consumer Protection Act, a major financial regulation reform bill. Obama also appointed Supreme Court justices Sonia Sotomayor and Elena Kagan, the former being the first Hispanic American on the Supreme Court. He oversaw the end of the Iraq War and ordered Operation Neptune Spear, the raid that killed Osama bin Laden, who was responsible for the September 11 attacks. Obama downplayed Bush's counterinsurgency model, expanding air strikes and making extensive use of special forces, while encouraging greater reliance on host-government militaries. He also ordered the 2011 military intervention in Libya to implement United Nations Security Council Resolution 1973, contributing to the overthrow of Muammar Gaddafi and the outbreak of the Libyan crisis."""

    chunker = RobustRecursiveChunker(document_id=1, chunk_size=200, chunk_overlap=0)
    chunks = chunker.process_document(test_text)

    for i, chunk in enumerate(chunks):
        print(f"=== Chunk {i + 1} ({chunk[2]} tokens) ===")
        print(chunk[1])
        print()
