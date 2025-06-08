# This script is adapted from the LangChain package, developed by LangChain AI.
# Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py
# License: MIT License

from typing import Any, List, Optional
from chunkers.utils import Language
from chunkers.token_chunker import TextSplitter
from emb.gemma_tokenizer import GemmaSingletonTokenizer
import re

def _split_text_with_regex(
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
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

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
        """
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0

        for d in splits:
            _len = self._length_function(d)
            if (
                    total + _len + (separator_len if len(current_doc) > 0 else 0)
                    > self._chunk_size
            ):
                if total > self._chunk_size:
                    import logging
                    logger = logging.getLogger(__name__)
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

    def split_text(self, text: str) -> List[tuple]:
        """
        Split text and return tuples in the required format.

        This is the main public interface that returns chunks in the format:
        (doc_id, chunk_text, token_count, "recursive", True)

        Args:
            text: Input text to be chunked

        Returns:
            List of tuples containing chunk information
        """
        # Get raw text chunks using the recursive algorithm
        chunk_texts = self._split_text(text, self._separators)

        # Convert to the required output format: (doc_id, chunk_text, token_count, "recursive", True)
        result = []
        for chunk_text in chunk_texts:
            token_count = self._tokenizer.count_tokens(chunk_text)
            result.append((self._doc_id, chunk_text, token_count, "recursive", True))

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


# Example usage demonstrating both general and language-specific chunking
if __name__ == "__main__":
    # Option 1: General text chunking
    general_chunker = RecursiveTokenChunker(
        doc_id=1,
        chunk_size=100,  # Small size for demonstration
        chunk_overlap=10
    )

    # Test with different types of content
    general_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """

    # Test general text chunking
    print("=== General Text Chunking ===")
    general_chunks = general_chunker.split_text(general_text)
    for i, (doc_id, chunk_text, token_count, chunk_type, is_processed) in enumerate(general_chunks):
        print(f"Chunk {i + 1}: Doc {doc_id}, {token_count} tokens, Type: {chunk_type}")
        print(f"Text: {chunk_text}...")
        print("-" * 40)

