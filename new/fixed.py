# This script is adapted from the LangChain package, developed by LangChain AI.
# Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py
# License: MIT License

from abc import ABC, abstractmethod
from enum import Enum
import logging
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)
from base_chunker import BaseChunker

from attr import dataclass

logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")


class TextSplitter(BaseChunker, ABC):
    """Interface for splitting text into chunks."""

    def __init__(
            self,
            chunk_size: int = 4000,
            chunk_overlap: int = 200,
            length_function: Callable[[str], int] = len,
            keep_separator: bool = False,
            add_start_index: bool = False,
            strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator in the chunks
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                              every document
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
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

    # @classmethod
    # def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> TextSplitter:
    #     """Text splitter that uses HuggingFace tokenizer to count length."""
    #     try:
    #         from transformers import PreTrainedTokenizerBase

    #         if not isinstance(tokenizer, PreTrainedTokenizerBase):
    #             raise ValueError(
    #                 "Tokenizer received was not an instance of PreTrainedTokenizerBase"
    #             )

    #         def _huggingface_tokenizer_length(text: str) -> int:
    #             return len(tokenizer.encode(text))

    #     except ImportError:
    #         raise ValueError(
    #             "Could not import transformers python package. "
    #             "Please install it with `pip install transformers`."
    #         )
    #     return cls(length_function=_huggingface_tokenizer_length, **kwargs)

    @classmethod
    def from_tiktoken_encoder(
            cls: Type[TS],
            encoding_name: str = "gpt2",
            model_name: Optional[str] = None,
            allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
            disallowed_special: Union[Literal["all"], Collection[str]] = "all",
            **kwargs: Any,
    ) -> TS:
        """Text splitter that uses tiktoken encoder to count length."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate max_tokens_for_prompt. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)

        def _tiktoken_encoder(text: str) -> int:
            return len(
                enc.encode(
                    text,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )

        if issubclass(cls, FixedTokenChunker):
            extra_kwargs = {
                "encoding_name": encoding_name,
                "model_name": model_name,
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_tiktoken_encoder, **kwargs)


class FixedTokenChunker(TextSplitter):
    """Splitting text to tokens using model tokenizer."""

    def __init__(
            self,
            encoding_name: str = "cl100k_base",
            model_name: Optional[str] = None,
            chunk_size: int = 4000,
            chunk_overlap: int = 200,
            allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
            disallowed_special: Union[Literal["all"], Collection[str]] = "all",
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for FixedTokenChunker. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(
                _text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)


@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: Callable[[List[int]], str]
    """ Function to decode a list of token ids to a string"""
    encode: Callable[[str], List[int]]
    """ Function to encode a string to a list of token ids"""


def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: List[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits

if __name__ == "__main__":
    chunker = FixedTokenChunker(
        chunk_size=100,  # Safe limit below 8192
        chunk_overlap=50  # 200 token overlap for context
    )

    # Example: Process a long document
    long_document = """
        Barack Hussein Obama II[a] (born August 4, 1961) is an American politician who was the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African American president in American history. Obama previously served as a U.S. senator representing Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.

    Born in Honolulu, Hawaii, Obama graduated from Columbia University in 1983 with a Bachelor of Arts degree in political science and later worked as a community organizer in Chicago. In 1988, Obama enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review. He became a civil rights attorney and an academic, teaching constitutional law at the University of Chicago Law School from 1992 to 2004. In 1996, Obama was elected to represent the 13th district in the Illinois Senate, a position he held until 2004, when he successfully ran for the U.S. Senate. In the 2008 presidential election, after a close primary campaign against Hillary Clinton, he was nominated by the Democratic Party for president. Obama selected Joe Biden as his running mate and defeated Republican nominee John McCain and his running mate Sarah Palin.

    Obama was awarded the 2009 Nobel Peace Prize for efforts in international diplomacy, a decision which drew both criticism and praise. During his first term, his administration responded to the 2008 financial crisis with measures including the American Recovery and Reinvestment Act of 2009, a major stimulus package to guide the economy in recovering from the Great Recession; a partial extension of the Bush tax cuts; legislation to reform health care; and the Dodd–Frank Wall Street Reform and Consumer Protection Act, a major financial regulation reform bill. Obama also appointed Supreme Court justices Sonia Sotomayor and Elena Kagan, the former being the first Hispanic American on the Supreme Court. He oversaw the end of the Iraq War and ordered Operation Neptune Spear, the raid that killed Osama bin Laden, who was responsible for the September 11 attacks. Obama downplayed Bush's counterinsurgency model, expanding air strikes and making extensive use of special forces, while encouraging greater reliance on host-government militaries. He also ordered the 2011 military intervention in Libya to implement United Nations Security Council Resolution 1973, contributing to the overthrow of Muammar Gaddafi and the outbreak of the Libyan crisis.
        """
  # Step 1: Split the document into chunks
    chunks = chunker.split_text(long_document)
    i = 0
    for chunk in chunks:
        print(f"CHUNK {i}:")
        i += 1
        print(chunk)
        print("\n")