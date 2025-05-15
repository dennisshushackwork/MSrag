# External imports:
import os
import asyncio
import logging
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Body
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Internal imports:
from pipelines.retrieval import Retriever
from graphdatabase.kuzudb import KuzuDB
from emb.embedder import Embedder

# Load environmental variables:
load_dotenv(verbose=True)

# Define the model used:
model = os.getenv("MODEL")

# Set logging:
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Pydantic Models for Request and Response ---
class ChunkRetrievalRequest(BaseModel):
    query: str = Field(...)
    chunking_method: str = Field(...)


class ChunkRetrievalResponse(BaseModel):
    query: str
    llm_response: str
    response_time: float
    context_length: int


class GraphRetrievalRequest(BaseModel):
    query: str = Field(...)


class GraphRetrievalAPIResponse(BaseModel):
    query: str
    llm_response: str
    response_time: float
    context_length: int

# ------------------------------------------------ #


# Initializes the Database and closes the connection:
kuzu_db = KuzuDB(create=True)
kuzu_db.close_connection()
# Initializes the embedder used later:
embedder = Embedder()

app = FastAPI(

    title="Graph-RAG API",
    description="API for performing advanced RAG and Graph-RAG operations.",
    version="1.0.1"
)

# --- Normal RAG --- #
@app.post("/retrieve/chunk", response_model=ChunkRetrievalResponse, tags=["RAG"])
async def api_perform_chunk_retrieval(request: ChunkRetrievalRequest = Body(...)):
    """
    Performs standard RAG by retrieving relevant text chunks based on the query
    and generating an answer using an LLM.
    """
    try:
        logger.info(f"Received chunk retrieval request for query: '{request.query}' with method: '{request.chunking_method}'")
        retriever = Retriever(model) # Use the model defined in the .env
        response = retriever.chunk_retrieval(query=request.query, chunking_method=request.chunking_method)
        if response is None:
            logger.error("chunk_retrieval did not return a response. This should not happen.")
            raise HTTPException(status_code=500, detail="Server error: Chunk retrieval failed to produce a response.")
        else:
            logger.info(f"Chunk retrieval for query '{request.query}' successful.")
            return ChunkRetrievalResponse(query=request.query, llm_response=response[0], response_time=response[1], context_length=response[2])
    except Exception as e:
        logger.error(f"Error during chunk retrieval for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during chunk retrieval: {str(e)}")

# --- Graph RAG --- #
@app.post("/retrieve/graph", response_model=GraphRetrievalAPIResponse, tags=["Graph-RAG"])
async def api_perform_graph_retrieval(request: GraphRetrievalRequest = Body(...)):
    """
    Performs Graph-RAG:
    1. Extracts entities from the query.
    2. Find similar entities in the knowledge graph (PostgreSQL for entity info, Kuzu for graph structure).
    3. Performs DFS from these entities in Kuzu to find related relationships.
    4. Ranks these relationships based on their cosine similarity to the user query.
    5. Use an agent to evaluate if the context is enough for the query.
    6. Creates the response
    """
    try:
        logger.info(f"Received graph retrieval request for query: '{request.query}', BFS depth: ")
        retriever = Retriever(model="openai")
        graph_rag_internal_results = await retriever.graph_retrieval(query=request.query)

        if not isinstance(graph_rag_internal_results, dict):
            logger.error(f"graph_retrieval returned an unexpected type: {type(graph_rag_internal_results)}. Expected dict.")
            raise HTTPException(status_code=500, detail="Server error: Graph retrieval produced an unexpected result format.")

        return GraphRetrievalAPIResponse(**graph_rag_internal_results)

    except Exception as e:
        logger.error(f"Error during graph retrieval for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during graph retrieval: {str(e)}")





# --- Health Check Endpoint ---
@app.get("/health", tags=["System"])
async def health_check():
    """
    Simple health check endpoint.
    Verifies basic operational status (e.g., Retriever instance available).
    """
    return {"status": "healthy", "message": "API is operational."}



if __name__ == "__main__":
    uvicorn.run(app,
                host="localhost",
                port=7000,
                reload=True)
