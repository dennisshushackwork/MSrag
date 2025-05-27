# This script is adapted from the LangChain package, developed by LangChain AI.
# Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py
# License: MIT License

from typing import Any, List, Optional
from base_chunker import BaseChunker
from utils import Language
from fixed_token_chunker import TextSplitter
import re


def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
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
    """Splitting text by recursively looking at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
            self,
            chunk_size: int = 6000,
            chunk_overlap: int = 200,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = False,
            **kwargs: Any,
    ) -> None:
        """Create a new RecursiveTokenChunker."""
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=keep_separator,
            **kwargs
        )
        self._separators = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)

    @classmethod
    def from_language(
            cls,
            language: Language,
            embedder=None,
            chunk_size: int = 6000,
            chunk_overlap: int = 200,
            **kwargs: Any
    ) -> "RecursiveTokenChunker":
        """Create a RecursiveTokenChunker configured for a specific language."""
        separators = cls.get_separators_for_language(language)

        if embedder is not None:
            # Use embedder's token counting function
            kwargs.pop('length_function', None)  # Remove if exists
            return cls(
                separators=separators,
                is_separator_regex=True,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=embedder.count_tokens,
                **kwargs
            )
        else:
            return cls(
                separators=separators,
                is_separator_regex=True,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )

    @classmethod
    def from_custom_embedder(
            cls,
            embedder,
            separators: Optional[List[str]] = None,
            language: Optional[Language] = None,
            chunk_size: int = 6000,
            chunk_overlap: int = 200,
            **kwargs: Any,
    ) -> "RecursiveTokenChunker":
        """Create a RecursiveTokenChunker that uses your custom embedder."""

        # Get separators from language if specified
        if language is not None:
            separators = cls.get_separators_for_language(language)
            kwargs['is_separator_regex'] = True
        elif separators is None:
            # Default separators for general text
            separators = ["\n\n", "\n", ".", "?", "!", " ", ""]

        # Remove length_function from kwargs if it exists to avoid conflict
        kwargs.pop('length_function', None)

        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=embedder.count_tokens,
            separators=separators,
            **kwargs
        )

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
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
        elif language == Language.KOTLIN:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\ninternal ",
                "\ncompanion ",
                "\nfun ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nwhen ",
                "\ncase ",
                "\nelse ",
                # Split by the normal type of lines
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
        elif language == Language.TS:
            return [
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\ntype ",
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
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
        elif language == Language.PHP:
            return [
                # Split along function definitions
                "\nfunction ",
                # Split along class definitions
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nforeach ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PROTO:
            return [
                # Split along message definitions
                "\nmessage ",
                # Split along service definitions
                "\nservice ",
                # Split along enum definitions
                "\nenum ",
                # Split along option definitions
                "\noption ",
                # Split along import statements
                "\nimport ",
                # Split along syntax declarations
                "\nsyntax ",
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
        elif language == Language.RST:
            return [
                # Split along section titles
                "\n=+\n",
                "\n-+\n",
                "\n\\*+\n",
                # Split along directive markers
                "\n\n.. *\n\n",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUBY:
            return [
                # Split along method definitions
                "\ndef ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\nfor ",
                "\ndo ",
                "\nbegin ",
                "\nrescue ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUST:
            return [
                # Split along function definitions
                "\nfn ",
                "\nconst ",
                "\nlet ",
                # Split along control flow statements
                "\nif ",
                "\nwhile ",
                "\nfor ",
                "\nloop ",
                "\nmatch ",
                "\nconst ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SCALA:
            return [
                # Split along class definitions
                "\nclass ",
                "\nobject ",
                # Split along method definitions
                "\ndef ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nmatch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SWIFT:
            return [
                # Split along function definitions
                "\nfunc ",
                # Split along class definitions
                "\nclass ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
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
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n",
                # Horizontal lines
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.LATEX:
            return [
                # First, try to split along Latex sections
                "\n\\\\chapter{",
                "\n\\\\section{",
                "\n\\\\subsection{",
                "\n\\\\subsubsection{",
                # Now split by environments
                "\n\\\\begin{enumerate}",
                "\n\\\\begin{itemize}",
                "\n\\\\begin{description}",
                "\n\\\\begin{list}",
                "\n\\\\begin{quote}",
                "\n\\\\begin{quotation}",
                "\n\\\\begin{verse}",
                "\n\\\\begin{verbatim}",
                # Now split by math environments
                "\n\\\begin{align}",
                "$$",
                "$",
                # Now split by the normal type of lines
                " ",
                "",
            ]
        elif language == Language.HTML:
            return [
                # First, try to split along HTML tags
                "<body",
                "<div",
                "<p",
                "<br",
                "<li",
                "<h1",
                "<h2",
                "<h3",
                "<h4",
                "<h5",
                "<h6",
                "<span",
                "<table",
                "<tr",
                "<td",
                "<th",
                "<ul",
                "<ol",
                "<header",
                "<footer",
                "<nav",
                # Head
                "<head",
                "<style",
                "<script",
                "<meta",
                "<title",
                "",
            ]
        elif language == Language.CSHARP:
            return [
                "\ninterface ",
                "\nenum ",
                "\nimplements ",
                "\ndelegate ",
                "\nevent ",
                # Split along class definitions
                "\nclass ",
                "\nabstract ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\nreturn ",
                # Split along control flow statements
                "\nif ",
                "\ncontinue ",
                "\nfor ",
                "\nforeach ",
                "\nwhile ",
                "\nswitch ",
                "\nbreak ",
                "\ncase ",
                "\nelse ",
                # Split by exceptions
                "\ntry ",
                "\nthrow ",
                "\nfinally ",
                "\ncatch ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SOL:
            return [
                # Split along compiler information definitions
                "\npragma ",
                "\nusing ",
                # Split along contract definitions
                "\ncontract ",
                "\ninterface ",
                "\nlibrary ",
                # Split along method definitions
                "\nconstructor ",
                "\ntype ",
                "\nfunction ",
                "\nevent ",
                "\nmodifier ",
                "\nerror ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo while ",
                "\nassembly ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.COBOL:
            return [
                # Split along divisions
                "\nIDENTIFICATION DIVISION.",
                "\nENVIRONMENT DIVISION.",
                "\nDATA DIVISION.",
                "\nPROCEDURE DIVISION.",
                # Split along sections within DATA DIVISION
                "\nWORKING-STORAGE SECTION.",
                "\nLINKAGE SECTION.",
                "\nFILE SECTION.",
                # Split along sections within PROCEDURE DIVISION
                "\nINPUT-OUTPUT SECTION.",
                # Split along paragraphs and common statements
                "\nOPEN ",
                "\nCLOSE ",
                "\nREAD ",
                "\nWRITE ",
                "\nIF ",
                "\nELSE ",
                "\nMOVE ",
                "\nPERFORM ",
                "\nUNTIL ",
                "\nVARYING ",
                "\nACCEPT ",
                "\nDISPLAY ",
                "\nSTOP RUN.",
                # Split by the normal type of lines
                "\n",
                " ",
                "",
            ]

        else:
            raise ValueError(
                f"Language {language} is not supported! "
                f"Please choose from {list(Language)}"
            )


# Example usage showing how to use RecursiveTokenChunker:
if __name__ == "__main__":
    from emb.embedder import Embedder

    # Create embedder instance
    embedder = Embedder()

    # Method 1: General recursive chunker
    chunker = RecursiveTokenChunker.from_custom_embedder(
        embedder=embedder,
        chunk_size=100,
        chunk_overlap=50
    )


    # Example: Process a Python code document
    python_code = '''
    Barack Hussein Obama II[a] (born August 4, 1961) is an American politician who was the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African American president in American history. Obama previously served as a U.S. senator representing Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.
    Born in Honolulu, Hawaii, Obama graduated from Columbia University in 1983 with a Bachelor of Arts degree in political science and later worked as a community organizer in Chicago. In 1988, Obama enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review. He became a civil rights attorney and an academic, teaching constitutional law at the University of Chicago Law School from 1992 to 2004. In 1996, Obama was elected to represent the 13th district in the Illinois Senate, a position he held until 2004, when he successfully ran for the U.S. Senate. In the 2008 presidential election, after a close primary campaign against Hillary Clinton, he was nominated by the Democratic Party for president. Obama selected Joe Biden as his running mate and defeated Republican nominee John McCain and his running mate Sarah Palin.
    Obama was awarded the 2009 Nobel Peace Prize for efforts in international diplomacy, a decision which drew both criticism and praise. During his first term, his administration responded to the 2008 financial crisis with measures including the American Recovery and Reinvestment Act of 2009, a major stimulus package to guide the economy in recovering from the Great Recession; a partial extension of the Bush tax cuts; legislation to reform health care; and the Dodd–Frank Wall Street Reform and Consumer Protection Act, a major financial regulation reform bill. Obama also appointed Supreme Court justices Sonia Sotomayor and Elena Kagan, the former being the first Hispanic American on the Supreme Court. He oversaw the end of the Iraq War and ordered Operation Neptune Spear, the raid that killed Osama bin Laden, who was responsible for the September 11 attacks. Obama downplayed Bush's counterinsurgency model, expanding air strikes and making extensive use of special forces, while encouraging greater reliance on host-government militaries. He also ordered the 2011 military intervention in Libya to implement United Nations Security Council Resolution 1973, contributing to the overthrow of Muammar Gaddafi and the outbreak of the Libyan crisis.
    """
    '''

    print("🐍 Python-specific chunking:")
    python_chunks = chunker.split_text(python_code)

    for i, chunk in enumerate(python_chunks):
        token_count = embedder.count_tokens(chunk)
        print(f"Chunk {i + 1} ({token_count} tokens):")
        print(chunk)
        print("-" * 40)

    print(f"\n📊 Total chunks: {len(python_chunks)}")