"""
Debug script to check if evaluation is working properly
"""

from pipelines.retrieval import Retriever
import json

def debug_single_query():
    """Debug a single query to see what's happening"""

    # Initialize retriever
    retriever = Retriever(model="openai")

    # Test query
    test_query = "What are the symptoms of COVID-19?"

    print("="*60)
    print("DEBUGGING SINGLE QUERY")
    print("="*60)
    print(f"Query: {test_query}")
    print()

    # Test all 4 strategies
    strategies = [
        ("semantic_basic", "semantic", False),
        ("semantic_reranked", "semantic", True),
        ("hybrid_basic", "hybrid", False),
        ("hybrid_reranked", "hybrid", True)
    ]

    for name, method, use_reranker in strategies:
        print(f"\n--- Testing {name} ---")

        try:
            if method == "semantic":
                result = retriever.semantic_retrieval(
                    query=test_query,
                    chunking_method="token",
                    top_k=5,
                    chroma=True,
                    re_ranker=use_reranker
                )
            else:  # hybrid
                result = retriever.hybrid_search(
                    query=test_query,
                    chunking_method="token",
                    top_k=5,
                    chroma=True,
                    re_ranker=use_reranker
                )

            print(f"Result type: {type(result)}")
            print(f"Result length: {len(result) if isinstance(result, list) else 'Not a list'}")

            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if (len(result) == 3 and
                    isinstance(result[1], (int, float)) and
                    isinstance(result[2], (int, float))):
                    # This is [response, time, context_length] format
                    chunks = result[0] if isinstance(result[0], list) else []
                    print(f"Format: [response, time, context_length]")
                    print(f"Number of chunks: {len(chunks)}")
                else:
                    # Direct chunks list
                    chunks = result
                    print(f"Format: Direct chunks list")
                    print(f"Number of chunks: {len(chunks)}")

                # Examine first chunk
                if chunks and len(chunks) > 0:
                    first_chunk = chunks[0]
                    print(f"First chunk type: {type(first_chunk)}")
                    if isinstance(first_chunk, dict):
                        print(f"First chunk keys: {list(first_chunk.keys())}")
                        content = (first_chunk.get('content') or
                                 first_chunk.get('text') or
                                 first_chunk.get('chunk_text') or
                                 first_chunk.get('passage_text', ''))
                        print(f"First chunk content preview: {content[:100]}...")
                    else:
                        print(f"First chunk: {str(first_chunk)[:100]}...")
                else:
                    print("❌ No chunks returned!")
            else:
                print("❌ Unexpected result format!")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


def debug_database_content():
    """Check what's actually in the database"""

    from postgres.retrieval import RetrievalQueries

    print("\n" + "="*60)
    print("DEBUGGING DATABASE CONTENT")
    print("="*60)

    try:
        with RetrievalQueries() as db:
            # Check total number of documents
            doc_count = db.get_document_count()  # You might need to implement this
            print(f"Total documents in database: {doc_count}")

            # Check total number of chunks
            chunk_count = db.get_chunk_count()  # You might need to implement this
            print(f"Total chunks in database: {chunk_count}")

            # Get sample chunks
            sample_chunks = db.get_sample_chunks(limit=3)  # You might need to implement this
            for i, chunk in enumerate(sample_chunks):
                print(f"\nSample chunk {i+1}:")
                print(f"  Content: {chunk.get('content', 'No content')[:100]}...")
                print(f"  Tokens: {chunk.get('tokens', 'No token count')}")

    except Exception as e:
        print(f"❌ Error accessing database: {str(e)}")
        print("You may need to implement get_document_count, get_chunk_count, get_sample_chunks methods")


def debug_relevance_calculation():
    """Test the relevance calculation logic"""

    print("\n" + "="*60)
    print("DEBUGGING RELEVANCE CALCULATION")
    print("="*60)

    # Mock data to test relevance calculation
    retrieved_chunks = [
        {"content": "COVID-19 symptoms include fever, cough, and difficulty breathing."},
        {"content": "The weather today is sunny and warm."},
        {"content": "Coronavirus can cause loss of taste and smell."}
    ]

    relevant_documents = [
        "COVID-19 commonly causes fever, cough, and shortness of breath.",
        "Loss of taste and smell are common symptoms of coronavirus."
    ]

    print(f"Retrieved chunks: {len(retrieved_chunks)}")
    print(f"Relevant documents: {len(relevant_documents)}")

    # Simulate the relevance calculation
    retrieved_texts = [chunk['content'].lower().strip() for chunk in retrieved_chunks]
    relevant_docs_lower = [doc.lower().strip() for doc in relevant_documents]

    relevance_scores = []
    for i, retrieved_text in enumerate(retrieved_texts):
        print(f"\nChunk {i+1}: {retrieved_text}")

        max_relevance = 0.0
        retrieved_words = set(retrieved_text.split())

        for j, relevant_doc in enumerate(relevant_docs_lower):
            relevant_words = set(relevant_doc.split())

            if len(relevant_words) > 0:
                overlap = len(retrieved_words & relevant_words) / len(relevant_words)
                substring_match = (retrieved_text in relevant_doc or relevant_doc in retrieved_text)

                relevance = max(overlap, 1.0 if substring_match else 0.0)
                max_relevance = max(max_relevance, relevance)

                print(f"  vs Relevant doc {j+1}: overlap={overlap:.3f}, substring={substring_match}, relevance={relevance:.3f}")

        is_relevant = max_relevance >= 0.3
        relevance_scores.append(1.0 if is_relevant else 0.0)
        print(f"  Final relevance: {max_relevance:.3f} -> {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")

    print(f"\nRelevance scores: {relevance_scores}")
    print(f"Hit rate: {1.0 if any(score > 0 for score in relevance_scores) else 0.0}")
    print(f"Precision@3: {sum(relevance_scores) / len(relevance_scores):.3f}")


def debug_ragbench_data():
    """Check what the RAGBench data actually looks like"""

    from datasets import load_dataset

    print("\n" + "="*60)
    print("DEBUGGING RAGBENCH DATA")
    print("="*60)

    try:
        # Load a small sample
        dataset = load_dataset("rungalileo/ragbench", "covidqa", split="test")

        print(f"Dataset size: {len(dataset)}")

        # Look at first example
        if len(dataset) > 0:
            example = dataset[0]
            print(f"\nFirst example:")
            print(f"ID: {example.get('id', 'No ID')}")
            print(f"Question: {example['question']}")
            print(f"Number of documents: {len(example['documents'])}")

            for i, doc in enumerate(example['documents'][:2]):  # First 2 docs
                print(f"\nDocument {i+1}:")
                print(f"  Length: {len(doc)} characters")
                print(f"  Preview: {doc[:200]}...")

    except Exception as e:
        print(f"❌ Error loading RAGBench data: {str(e)}")


if __name__ == "__main__":
    print("🔍 EVALUATION DEBUGGING SCRIPT")
    print("This will help identify why you're getting perfect scores")

    # Run all debug functions
    debug_ragbench_data()
    debug_single_query()
    debug_relevance_calculation()
    debug_database_content()

    print("\n" + "="*60)
    print("🔍 ANALYSIS")
    print("="*60)
    print("Look for these issues:")
    print("1. Are chunks actually being returned by your retriever?")
    print("2. Is the database properly populated?")
    print("3. Is the relevance calculation too lenient?")
    print("4. Are the RAGBench questions too easy for your data?")