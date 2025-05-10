# External imports:
from typing import List
from abc import ABC, abstractmethod

# Internal imports:
from embedder.embedder import Embedder

class BaseChunker(ABC):
    """
    Base Chunker class (Abstract Class) for the chunking method implementation.
    Arguments:
        - document_id: id of the document (int)
        - chunk_size: size of the chunk (int)
        - chunk_overlap: overlap between chunks (int)
    """
    def __init__(self,
                 document_id: int,
                 chunk_size: int,
                 chunk_overlap: int,
                 ):
        if chunk_overlap > chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.document_id = document_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = Embedder()

    def tokenize(self, text: str) -> List[int]:
        """Returns the token ids"""
        return self.embedder.tokenize_to_ids(text)

    def detokenize(self, tokens: List[int]) -> str:
        """Returns the detokenized text"""
        return self.embedder.decode_tokens(tokens)

    def count_tokens(self, text: str) -> int:
        """Returns the number of tokens"""
        return self.embedder.count_tokens(text)

    @staticmethod
    def join_docs(parts: List[str]) -> str:
        """Joins seperate documents to one"""
        return " ".join(parts)

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Splits the text document into multiple chunks"""
        pass

