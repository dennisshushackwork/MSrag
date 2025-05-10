# External imports:
import re
import logging
from typing import List

# Internal imports:
from chunkers.base import BaseChunker

# Attempt to import nltk for sentence tokenization:
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except nltk.download.DownloadError:
        logging.warning("NLTK dataset not found. Downloading now.")
        nltk.download("punkt")
    NLTK_AVAILABLE = True
    logging.info("NLTK dataset available.")
except ImportError:
    NLTK_AVAILABLE = False

# Initialising the logger:
logging.basicConfig(level=logging.INFO,format='%(asctime)s - '
                                              '%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentenceTokenChunker(BaseChunker):
    """
    Chunks text by grouping sentences to approximate a target token count (chunk_size),
    with a token-based overlap. Ensures chunks meet minimum requirements and merges
    undersized chunks.
    """

    # Thresholds:
    MIN_CHUNK_LENGTH = 10  # Limit in characters
    MIN_TOKENS = 5  # Limit in tokens

    # Initialising the class:
    def __init__(self,
                 document_id: int,
                 chunk_size: int = 400,
                 chunk_overlap: int = 100,  # Default token overlap for sentence chunker
                 **kwargs):
        super().__init__(document_id, chunk_size, chunk_overlap, **kwargs)

        logger.info(f"SentenceTokenChunker initialized with chunk_size={self.chunk_size}, "
                    f"chunk_overlap={self.chunk_overlap}, NLTK_AVAILABLE={NLTK_AVAILABLE}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences. Uses NLTK if available, otherwise a regex-based fallback.
        """
        if not text or not text.strip():
            return []

        if NLTK_AVAILABLE:
            logger.debug("Using NLTK for sentence splitting.")
            return nltk.sent_tokenize(text)
        else:
            # Fallback to regex-based approach
            logger.debug("Using regex-based sentence splitting (NLTK not available or 'punkt' missing).")
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s for s in sentences if s.strip()]

    def split_text(self, text: str) -> List[str]:
        """
        Splits text by grouping sentences to fit chunk_size (tokens) and applying token_overlap.
        """
        if not text.strip():
            logger.warning("Input text is empty or whitespace only.")
            return []

        # Splits the input text into sentences:
        source_sentences_texts = self._split_into_sentences(text)
        if not source_sentences_texts:
            logger.warning("No sentences found after splitting the text.")
            return []

        # Adds the sentence metadata for further processing:
        sentences_data = []
        for i, s_text in enumerate(source_sentences_texts):
            # Gets the tokens per sentence:
            token_ids = self.tokenize(s_text)
            if not token_ids:
                logger.debug(f"Sentence {i} ('{s_text[:30]}...') resulted in no tokens.")
                continue
            sentences_data.append({
                'text': s_text,
                'original_idx': i,
                'token_count': len(token_ids)
            })

        if not sentences_data:
            logger.warning("No valid sentences with tokens to process.")
            return []

        all_candidate_chunk_texts: List[str] = [] # Will store all the text chunks.
        current_s_cursor = 0 # indicating the starting sentence index in `sentences_data` for new chunk


        while current_s_cursor < len(sentences_data):
            chunk_sentences_for_current_pass = [] # Sentences in the current pass
            chunk_tokens_for_current_pass = 0 # Number of tokens currently used up
            last_s_idx_in_this_chunk = -1 # Tracks sentence index

            # Inner loop: Greedily add sentences to the current chunk.
            for i_build_chunk in range(current_s_cursor, len(sentences_data)):
                sentence = sentences_data[i_build_chunk]

                # Case A: Handle a single sentence that is already larger than `self.chunk_size`.
                # If the current chunk is empty (no sentences added yet) and this one sentence
                # by itself exceeds the desired `chunk_size`, it forms a chunk on its own.
                if not chunk_sentences_for_current_pass and sentence['token_count'] > self.chunk_size:
                    chunk_sentences_for_current_pass.append(sentence)
                    last_s_idx_in_this_chunk = i_build_chunk
                    logger.debug(f"Chunk contains single oversized sentence {sentence['original_idx']} "
                                 f"with {sentence['token_count']} tokens.")
                    break

                # Case B: Check if adding the current sentence would make the chunk too large.
                # If the chunk already has sentences, and adding the current sentence's tokens
                # would push the total `chunk_tokens_for_current_pass` over `self.chunk_size`,
                # then don't add this sentence. The current chunk is considered complete with
                # the sentences added so far.
                if chunk_sentences_for_current_pass and \
                        (chunk_tokens_for_current_pass + sentence['token_count'] > self.chunk_size):
                    break

                # If neither of the above conditions is met, add the current sentence to the chunk.
                chunk_sentences_for_current_pass.append(sentence)
                chunk_tokens_for_current_pass += sentence['token_count']
                last_s_idx_in_this_chunk = i_build_chunk

                # Case C: If this was the last available sentence in the document.
                if i_build_chunk == len(sentences_data) - 1:
                    break #  Finalize chunk, as there are no more sentences.

            # If, for some reason, no sentences were added to the current pass (e.g., all remaining sentences
            # were filtered out or an edge case in logic), advance the cursor to prevent an infinite loop.
            if not chunk_sentences_for_current_pass:
                current_s_cursor += 1
                continue

            # Join the text of the collected sentences to form the candidate chunk's text.
            current_chunk_text_joined = " ".join([s['text'] for s in chunk_sentences_for_current_pass]).strip()

            # Add the formed chunk text to our list of candidates if it's not empty.
            if current_chunk_text_joined:
                logger.info(f"Formed candidate chunk from sentences original_idx "
                            f"{chunk_sentences_for_current_pass[0]['original_idx']} to "
                            f"{chunk_sentences_for_current_pass[-1]['original_idx']} "
                            f"with ~{chunk_tokens_for_current_pass} tokens.")
                all_candidate_chunk_texts.append(current_chunk_text_joined)

            #    Overlap Calculation and Advancing the Main Cursor:
            #    This section determines where the *next* chunk should start building from,
            #    to achieve the desired `self.chunk_overlap` in tokens.

            # If all sentences have been included in the last built chunk, we're done.
            if last_s_idx_in_this_chunk < 0 or last_s_idx_in_this_chunk >= len(sentences_data) - 1:
                break

            accumulated_overlap_tokens = 0
            next_chunk_start_s_cursor = last_s_idx_in_this_chunk + 1  # Default if no suitable overlap found

            # Iterate backwards from the end of the current chunk to find overlap start
            for i_overlap in range(last_s_idx_in_this_chunk, current_s_cursor - 1, -1):
                sentence_for_overlap = sentences_data[i_overlap]
                accumulated_overlap_tokens += sentence_for_overlap['token_count']
                next_chunk_start_s_cursor = i_overlap
                if accumulated_overlap_tokens >= self.chunk_overlap:
                    logger.debug(f"Overlap target of {self.chunk_overlap} tokens met. "
                                 f"Next chunk to start at sentence original_idx {sentences_data[i_overlap]['original_idx']}.")
                    break

            if next_chunk_start_s_cursor <= current_s_cursor and current_s_cursor < len(sentences_data) - 1:
                logger.debug(f"Overlap calculation would stall or regress cursor (next_start_original_idx="
                             f"{sentences_data[next_chunk_start_s_cursor]['original_idx'] if next_chunk_start_s_cursor < len(sentences_data) else 'OOB'}, "
                             f"current_original_idx={sentences_data[current_s_cursor]['original_idx']}). Forcing advance.")
                current_s_cursor += 1
            else:
                current_s_cursor = next_chunk_start_s_cursor

            # Additional check for the single-sentence chunk causing no progress with overlap
            if len(chunk_sentences_for_current_pass) == 1 and \
                    current_s_cursor == last_s_idx_in_this_chunk and \
                    current_s_cursor < len(sentences_data) - 1:
                logger.debug(
                    f"Single sentence chunk overlap points to itself (original_idx {sentences_data[current_s_cursor]['original_idx']}). Forcing advance.")
                current_s_cursor += 1

        final_merged_chunks: List[str] = []
        if not all_candidate_chunk_texts:
            logger.warning("No candidate chunks were formed.")
            return []

        for cand_chunk_text in all_candidate_chunk_texts:
            stripped_cand_text = cand_chunk_text.strip()
            if not stripped_cand_text:
                continue

            actual_token_count = len(self.tokenize(stripped_cand_text))
            logger.debug(
                f"Validating candidate chunk (approx {len(stripped_cand_text)} chars, {actual_token_count} tokens): '{stripped_cand_text[:100]}...'")

            if len(stripped_cand_text) >= self.MIN_CHUNK_LENGTH and \
                    actual_token_count >= self.MIN_TOKENS:
                logger.debug("Chunk is valid. Adding.")
                final_merged_chunks.append(stripped_cand_text)
            else:
                logger.debug(
                    f"Chunk is too small (chars: {len(stripped_cand_text)}, tokens: {actual_token_count}). Attempting merge.")
                if final_merged_chunks:
                    logger.debug("Merging with previous chunk.")
                    final_merged_chunks[-1] = f"{final_merged_chunks[-1]} {stripped_cand_text}".strip()
                else:
                    logger.debug("It's the first chunk and too small; adding anyway.")
                    final_merged_chunks.append(stripped_cand_text)

        logger.info(f"Produced {len(final_merged_chunks)} final chunks after merging.")
        return [chunk for chunk in final_merged_chunks if chunk]

    def process_document(self, document: str) -> List[tuple]:
        final_chunks_data: List[tuple] = []
        text_chunks = self.split_text(text=document)
        for chunk_text in text_chunks:
            tokens = self.tokenize(chunk_text)
            final_chunks_data.append(
                (
                    self.document_id,
                    chunk_text,
                    len(tokens),
                    "sentence_nltk_token_grouped" if NLTK_AVAILABLE else "sentence_regex_token_grouped",
                    True
                )
            )
        return final_chunks_data

if __name__ == "__main__":
    document = """Hello my name is Dennis. I am a cool guy, looking for an improvement of my life.
    Please make sure to include the best things into my name. What is the derivative of 5?
    Born in Honolulu, Hawaii, Obama graduated from Columbia University in 1983 with a Bachelor of 
    Arts degree in political science and later worked as a community organizer in Chicago. In 1988, 
    Obama enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review.
    He became a civil rights attorney and an academic, teaching constitutional law at the University 
    of Chicago Law School from 1992 to 2004. In 1996, Obama was elected to represent the 13th district 
    in the Illinois Senate, a position he held until 2004, when he successfully ran for the U.S. Senate. 
    In the 2008 presidential election, after a close primary campaign against Hillary Clinton, he was 
    nominated by the Democratic Party for president. Obama selected Joe Biden as his running mate and 
    defeated Republican nominee John McCain and his running mate Sarah Palin.

    Obama was awarded the 2009 Nobel Peace Prize for efforts in international diplomacy, a decision which drew both criticism and praise. During his first term, his administration responded to the 2008 financial crisis with measures including the American Recovery and Reinvestment Act of 2009, a major stimulus package to guide the economy in recovering from the Great Recession; a partial extension of the Bush tax cuts; legislation to reform health care; and the Dodd–Frank Wall Street Reform and Consumer Protection Act, a major financial regulation reform bill. Obama also appointed Supreme Court justices Sonia Sotomayor and Elena Kagan, the former being the first Hispanic American on the Supreme Court. He oversaw the end of the Iraq War and ordered Operation Neptune Spear, the raid that killed Osama bin Laden, who was responsible for the September 11 attacks. Obama downplayed Bush's counterinsurgency model, expanding air strikes and making extensive use of special forces, while encouraging greater reliance on host-government militaries. He also ordered the 2011 military intervention in Libya to implement United Nations Security Council Resolution 1973, contributing to the overthrow of Muammar Gaddafi and the outbreak of the Libyan crisis.

    Obama defeated Republican opponent Mitt Romney and his running mate Paul Ryan in the 2012 presidential election. In his second term, Obama advocated for gun control in the wake of the Sandy Hook Elementary School shooting, took steps to combat climate change, signing the Paris Agreement, a major international climate agreement, and an executive order to limit carbon emissions. Obama also presided over the implementation of the Affordable Care Act and other legislation passed in his first term. He initiated sanctions against Russia following the invasion in Ukraine and again after Russian interference in the 2016 U.S. elections, ordered military intervention in Iraq in response to gains made by ISIL following the 2011 withdrawal from Iraq, negotiated the Joint Comprehensive Plan of Action (a nuclear agreement with Iran), and normalized relations with Cuba. The number of American soldiers in Afghanistan decreased during Obama's second term, though U.S. soldiers remained in the country throughout the remainder of his presidency. Obama promoted inclusion for LGBT Americans, becoming the first sitting U.S. president to publicly support same-sex marriage.

    Obama left office in 2017 with high approval ratings both within the United States and among foreign advisories. He continues to reside in Washington D.C. and remains politically active, campaigning for candidates in various American elections, including in Biden's successful presidential bid in the 2020 presidential election. Outside of politics, Obama has published three books: Dreams from My Father (1995), The Audacity of Hope (2006), and A Promised Land (2020). His presidential library began construction in the South Side of Chicago in 2021. Historians and political scientists rank Obama among the upper tier in historical rankings of U.S. presidents."""

    logger.info("\n--- Testing SentenceTokenChunker ---")
    # Ensure your BaseChunker is correctly imported and initialized
    # For the demo, I'll assume the placeholder BaseChunker is used if you uncomment it,
    # OR that your actual chunkers.base.BaseChunker works with the embedder.
    sentence_chunker = SentenceTokenChunker(document_id=1, chunk_size=400, chunk_overlap=100)

    processed_sentence_chunks = sentence_chunker.process_document(document)

    logger.info(f"\n--- Generated {len(processed_sentence_chunks)} Chunks (SentenceTokenChunker) ---")
    for i, chunk_data in enumerate(processed_sentence_chunks):
        print(chunk_data)