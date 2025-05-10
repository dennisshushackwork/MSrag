# External imports:
import re
import logging
from typing import List

# Internal imports:
from chunkers.base import BaseChunker

# Initialising the Logger:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenChunker(BaseChunker):
    """
    Chunks the text based on the number of tokens and uses an overlap (of tokens).
    Ensures that each of the chunks meets the minimum requirements and token threshold,
    and merges any undersized chunks into the previous one to avoid data loss.
    """
    # Thresholds:
    MIN_CHUNK_LENGTH = 10 # Limit in characters
    MIN_TOKENS = 5 # Limit in tokens

    # Initialising the class:
    def __init__(self,
                 document_id: int,
                 chunk_size: int = 400,
                 chunk_overlap: int = 200
                 ):
        super().__init__(document_id, chunk_size, chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        """Splits the text into chunks by token count with token-level overlap and merges small tails."""

        if not text.strip():
            return []

        # Gets the tokens/input_ids of the text:
        input_ids = self.tokenize(text)
        chunks: List[str] = []
        start = 0

        while start < len(input_ids):
            end = start + self.chunk_size
            chunk_ids = input_ids[start:end]

            # It's good to print the current window being processed
            logger.info(f"Processing window: start={start}, end={end}, chunk_ids (first 10): {chunk_ids[:10]}")

            if not chunk_ids:  # If start is at or beyond the end of input_ids, this could be empty
                break

            # Decode and trim whitespace (consider adding .strip() back if needed)
            chunk_text = self.detokenize(chunk_ids).strip()
            token_count = len(chunk_ids)

            # If chunk meets thresholds, add it; otherwise merge into previous
            if len(chunk_text.strip()) >= self.MIN_CHUNK_LENGTH and token_count >= self.MIN_TOKENS:  # Added .strip() for len check
                chunks.append(chunk_text.strip())  # Store stripped version
            else:
                # Ensure chunk_text is stripped before concatenation if it contains only whitespace
                stripped_chunk_text = chunk_text.strip()
                if chunks:  # If there's a previous chunk to merge into
                    if stripped_chunk_text:  # Only append if there's non-whitespace content
                        chunks[
                            -1] = f"{chunks[-1]} {stripped_chunk_text}".strip()  # Ensure previous chunk also remains stripped
                else:  # The first chunk is small
                    if stripped_chunk_text or not chunks:  # Add if it has content, or if it's truly the first (even if empty, to signify processing)
                        chunks.append(stripped_chunk_text)

            # Advance by chunk size minus overlap:
            start += (self.chunk_size - self.chunk_overlap)

        # Filter out any completely empty chunks that might have resulted from merging empty strings
        return [chunk for chunk in chunks if chunk]

    def process_document(self, document: str) -> List[tuple]:
        """Processes the document and returns a list of tuple chunks (document_id, text, token_count, method, ready)."""
        final_chunks: List[tuple] = []
        text_chunks = self.split_text(text=document)
        for chunk_text in text_chunks:
            tokens = self.tokenize(chunk_text)
            final_chunks.append(
                (
                    self.document_id,
                    chunk_text,
                    len(tokens),
                    "token",
                    True
                )
            )
        return final_chunks


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
    chunker = TokenChunker(document_id=1)
    chunks = chunker.process_document(document)
    for chunk in chunks:
        print("\n")
        print(chunk)