"""
Performs the different retrieval tasks based on the provided type of retrieval
asked for: i.e. RAG, Graph-RAG, Local Search, and Drift Search
"""
# External imports:
import os
import time
import logging
from typing import List
from dotenv import load_dotenv

# Define the logging handler:
load_dotenv(verbose=False)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# External imports:
from llm.llm import LLMClient
from llm.prompts.rag.reponse import RagPrompt
from llm.prompts.graphrag.entities import EntityExtractor
from emb.embedder import Embedder
from postgres.retrieval import RetrievalQueries

class Retriever:
    """
    The user formulates a query. With this query we find the top-k most similar chunks
    These chunks are then used to create the answer.
    """
    def __init__(self, model: str):
        self.embedder = Embedder()
        self.max_length = int(os.getenv("CONTEXT"))
        self.model = model

    def context_size_chunk(self, context_chunks, max_length) -> List[tuple]:
        """
        This function is required to calculate the context length given some text chunks. If it is too large,
        we cut off some context (the least relevant chunks).
        """
        context = 0
        for chunk in context_chunks:
            context += chunk[2]
        if context > max_length:
            logger.info("Context too large, removing least relevant chunk")
            context_chunks = context_chunks[:-1]
            return self.context_size_chunk(context_chunks, max_length)
        return context_chunks

    # --------------- Performs RAG ------------------- #

    def chunk_retrieval(self, query: str, chunking_method: str):
        """Simple RAG retrieval (normal RAG)"""
        chunks = []
        start = time.time()

        # Embed the input:
        embedding = str(self.embedder.embed_texts([query])[0])

        # Performs hybrid search to gather the context for RAG:
        with RetrievalQueries() as db:
            chunks = db.hybrid_search(query, embedding, chunking_method)

        # Makes sure the context lenght is below 8000:
        context_chunks = self.context_size_chunk(chunks, int(os.getenv("CONTEXT")))
        text_chunks = [chunk[1] for chunk in context_chunks]

        # We now perform the RAG query with the provided context:
        rag_prompt = RagPrompt(query, context=text_chunks)
        client = LLMClient(rag_prompt.system_prompt,rag_prompt.human_prompt, temperature=0.7, provider=self.model)
        response = client.send_message()
        end = time.time()
        print(end - start)






if __name__ == '__main__':
    rag = Retriever(model="openai")
    response = rag.chunk_retrieval(query="Who is obama's wife?", chunking_method="recursive")

    #rag.graph_retrieval('Who is Obamas wife?', 'recursive', limit=60)        text_chunks = [chunk['content'] for chunk in context_chunks]



