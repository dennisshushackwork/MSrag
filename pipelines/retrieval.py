"""
Performs the different retrieval tasks based on the provided type of retrieval
asked for: i.e. RAG, Graph-RAG, Local Search, and Drift Search
"""
# External imports:
import os
import time
import logging
import asyncio
from dotenv import load_dotenv
from typing import Optional

# Define the logging handler:
load_dotenv(verbose=False)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# External imports:
from llm.prompts.rag.reponse import RagPrompt
from llm.prompts.graphrag.agent import RelationshipAgent
from llm.prompts.graphrag.entities import EntityExtractor
from emb.embedder import Qwen3Embedder
from postgres.retrieval import RetrievalQueries
from graphdatabase.dfs import DFS

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

class Retriever:
    """
    The user formulates a query. With this query we find the top-k most similar chunks
    These chunks are then used to create the answer.
    """
    def __init__(self, model: str):
        self.embedder = Qwen3Embedder()
        self.max_length = int(os.getenv("CONTEXT"))
        self.model = model

    def context_size_chunk(self, context_chunks, max_length):
        """
        This function is required to calculate the context length given some text chunks. If it is too large,
        we cut off some context (the least relevant chunks).
        """
        context_length = 0
        for chunk in context_chunks:
            context_length += chunk["tokens"]
        if context_length> max_length:
            logger.info("Context too large, removing least relevant chunk")
            context_chunks = context_chunks[:-1]
            return self.context_size_chunk(context_chunks, max_length)
        return context_chunks, context_length

    # --------------- Performs RAG ------------------- #
    def semantic_retrieval(self, query: str, chunking_method: str, top_k: Optional[int] = None, chroma: Optional[bool] = False, re_ranker: Optional[bool] = False):
        """Performs semantic retrieval with or without re-ranking"""
        start = time.time()
        return_response = []

        # Embed the input:
        custom_instruction = "Retrieve relevant documents that answer the user's question"
        embedding = str(self.embedder.embed_texts([query], are_queries=True, custom_instruction=custom_instruction)[0])

        context_chunks = []
        if re_ranker == False:
            # Performs hybrid search to gather the context for RAG:
            with RetrievalQueries() as db:
                # Retrieve the chunks:
                chunks = db.semantic_search(embedding, chunking_method, limit=top_k)
                # Checks that it does not go over the max context:
                context_chunks, context_length = self.context_size_chunk(chunks, int(os.getenv("CONTEXT")))
                if chroma:
                    return chunks
        else:
              with RetrievalQueries() as db:
                # Semantic search and Rerank the documents:
                instruction="Find information that directly answers the user's question"
                reranked_chunks = db.semantic_search_with_reranking(query, embedding, chunking_method, final_top_k=top_k, instruction=instruction, initial_limit=20)
                # Checks that it does not go over the max context:
                context_chunks, context_length = self.context_size_chunk(reranked_chunks, int(os.getenv("CONTEXT")))
                if chroma:
                    return context_chunks

        text_chunks = [chunk["content"] for chunk in context_chunks]
        # We now perform the RAG query with the provided context:
        rag_prompt = RagPrompt(query, context=text_chunks, model=self.model)
        response = rag_prompt.generate_response()
        end = time.time()
        request_time = end - start
        print(f"Request time: {request_time}")
        return_response.extend([response, request_time, context_length])
        return return_response

    def hybrid_search(self, query: str, chunking_method: str, top_k: Optional[int] = None, chroma: Optional[bool] = False, re_ranker: Optional[bool] = False):
        """Performs semantic retrieval with or without re-ranking"""
        start = time.time()
        return_response = []

        # Embed the input:
        custom_instruction = "Retrieve relevant documents that answer the user's question"
        embedding = str(self.embedder.embed_texts([query], are_queries=True, custom_instruction=custom_instruction)[0])
        context_chunks = []
        if re_ranker == False:
            # Performs hybrid search to gather the context for RAG:
            with RetrievalQueries() as db:
                # Retrieve the chunks:
                chunks = db.hybrid_search(query_text=query, query_embedding=embedding, chunk_type=chunking_method, final_limit=top_k)
                # Checks that it does not go over the max context:
                context_chunks, context_length = self.context_size_chunk(chunks, int(os.getenv("CONTEXT")))
                if chroma:
                    return context_chunks
        else:
              with RetrievalQueries() as db:
                # Semantic search and Rerank the documents:
                instruction="Find information that directly answers the user's question."
                reranked_chunks = db.semantic_search_with_reranking(query, embedding, chunking_method, final_top_k=top_k, instruction=instruction, initial_limit=20)
                # Checks that it does not go over the max context:
                context_chunks, context_length = self.context_size_chunk(reranked_chunks, int(os.getenv("CONTEXT")))
                if chroma:
                    return context_chunks

        text_chunks = [chunk["content"] for chunk in context_chunks]
        # We now perform the RAG query with the provided context:
        rag_prompt = RagPrompt(query, context=text_chunks, model=self.model)
        response = rag_prompt.generate_response()
        end = time.time()
        request_time = end - start
        print(f"Request time: {request_time}")
        return_response.extend([response, request_time, context_length])
        return return_response

    # --------------- Graph-RAG ------------------- #
    async def graph_retrieval(self, query: str):
        """Performs normal graphrag on the database"""

        logger.info(f"Starting graph retrieval for query: '{query}'")
        start_time = time.time()

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
        entity_dict_for_search = {entity: str(emb) for entity, emb in zip(entities, entities_embedding)}
        with RetrievalQueries() as db:
            similar_entity_ids = db.find_similar_entities(entity_dict_for_search, top_k=top_k)
            logger.info(f"Found {len(similar_entity_ids)} potentially relevant entity IDs in DB: {similar_entity_ids}")

        sufficient_context = "NO"
        dfs_depth = 1
        max_depth = 4
        context_relationships = []
        context_length = 0

        while sufficient_context == "NO" and dfs_depth <= max_depth:
            # Perform DFS
            dfs = await DFS.get_relationships_async(
                entity_ids=similar_entity_ids,
                max_depth=dfs_depth
            )

            # Gets the ranked relationships:
            relationships = []
            with RetrievalQueries() as db:
                relationships = db.extract_rel_given_ids(query_embedding, dfs)

            # Get only the relevant relationships (max size = 7000 tokens):
            context_relationships, context_length = self.context_size_chunk(relationships, int(os.getenv("CONTEXT")))
            context_relationships = [rel[0] for rel in context_relationships]

            # Define an agent to test if we have enough context to answer the question:
            agent = RelationshipAgent(user_query=query, relationships=context_relationships, model=self.model)
            sufficient_context = agent.check_context()
            logger.info("The context is sufficient to answer the query.")

            if sufficient_context == "NO":
                logger.info(f"The context is not sufficient to answer the question: {query}, Trying depth: {dfs_depth}")
                dfs_depth += 1

        # Generate the response using the context
        rag_prompt = RagPrompt(query, context=context_relationships, model=self.model)
        llm_response = rag_prompt.generate_response()

        # Calculate the total response time
        response_time = time.time() - start_time

        # Return a dictionary that matches the GraphRetrievalAPIResponse model
        return {
            "query": query,
            "llm_response": llm_response,
            "response_time": response_time,
            "context_length": int(context_length)
        }

async def main():
    """Main entry point for running the retriever"""
    rag = Retriever(model="openai")

    # Uncomment the line you want to run:
    #response = rag.chunk_retrieval(query="What is the tower building?", chunking_method="token", chroma=True, top_k=5)
    response = rag.hybrid_search(query=" did compensation expense related to the company 2019s employee stock purchase plan grow from 2004 to 2005?", chunking_method="token", chroma=False, top_k=5, re_ranker=True)
    #response = await rag.graph_retrieval(query="Who is Michelle Obama? What is the wife of Obama?")
    print(response)

if __name__ == '__main__':
    asyncio.run(main())

