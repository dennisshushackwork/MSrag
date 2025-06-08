"""
Base Chunker Abstract class: Defines the minimum of what a chunker has to implement.
"""

from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass