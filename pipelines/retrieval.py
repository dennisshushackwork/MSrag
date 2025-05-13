"""
Performs the different retrieval tasks based on the provided type of retrieval
asked for: i.e. RAG, Graph-RAG, Local Search, and Drift Search
"""
# External imports:
import os
import time
import logging
import asyncio
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
from graphdb.bfs import BFS

def define_top_k(entity_count: int):
    """Defines the number of top-k similar entities extracted from the database, based
    on the number of entities"""
    if entity_count < 3: # Up to 15 starting entities
        top_k = 5
    elif 3 < entity_count < 5: # Up to 15 starting entities
        top_k = 3
    else:
        top_k = 2
    return top_k

async def run_dfs(entity_ids: List[int], max_depth: int):
    """Runs the dfs search iteratively"""
    bfs_service = BFS(recreate_db=False)
    return await bfs_service.run_multiple_bfs_searches(
        entity_ids=entity_ids,
        max_depth=max_depth,
    )

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
        print(response)
        end = time.time()
        print(end - start)

    # --------------- Graph-RAG ------------------- #

    async def graph_retrieval(self, query: str):
        """Performs normal graphrag on the database"""

        # Step 1: Extract all the entities from the user query:
        extractor = EntityExtractor(query=query, model=self.model)
        entities = extractor.extract_entities()
        entity_count = len(entities)
        top_k = define_top_k(entity_count)
        logger.info(f"Extracted {entity_count} entities: {entities}")


        # Step 2: Embedd the entities and the query:
        entities.append(query)
        entities_embedding = self.embedder.embed_texts(entities)
        query_embedding = entities_embedding[-1]
        entities_embedding.pop()

        # Step 3: Get the entity_ids from the database (top-k similar per entity):
        entity_dict = {entity: str(emb) for entity, emb in zip(entities, entities_embedding)}
        with RetrievalQueries() as db:
            similar_entities = db.find_similar_entities(entity_dict, top_k=top_k)
            logger.info(f"Found {len(similar_entities)} similar entities: {similar_entities}")

        # Step 4: Perform Async BFS inside of Kuzu:
        max_depth = 1

        # Run the multiple BFS searches asynchronously
        bfs_results = await run_dfs(
            entity_ids=similar_entities,
            max_depth=max_depth,
        )

        logger.info(f"BFS search completed. Results obtained for {len(bfs_results)} starting entities.")

        # Get only the unique values:
        all_found_relationship_ids = set()
        for rel_list in bfs_results.values():
            all_found_relationship_ids.update(rel_list)

        logger.info(f"Found {len(all_found_relationship_ids)} unique relationships")


async def main():
    """Main entry point for running the retriever"""
    rag = Retriever(model="openai")

    # Uncomment the line you want to run:
    # response = rag.chunk_retrieval(query="Who is obama's wife?", chunking_method="recursive")
    response = await rag.graph_retrieval(query="Who is Michelle Obama? What is the wife of Obama?")
    return response

if __name__ == '__main__':
    asyncio.run(main())


