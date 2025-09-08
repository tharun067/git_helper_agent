from typing import List, Dict, Any
import numpy as np

# Import necessary classes and constants from the ingestion script
# This assumes 'vector_store_creator.py' is in the same directory.
from vector_store import EmbeddingManager, VectorStore, CHROMADB_PATH, COLLECTION_NAME, EMBEDDING_MODEL


class Retriever:
    """Handles query-based document retrieval from the ChromaDB vector store."""

    def __init__(self, persist_directory: str = CHROMADB_PATH, collection_name: str = COLLECTION_NAME,
                 model_name: str = EMBEDDING_MODEL):
        """
        Initializes the retriever by setting up the necessary managers.
        """
        print("--- Initializing RAG Retriever ---")
        self.embedding_manager = EmbeddingManager(model_name=model_name)
        self.vector_store_manager = VectorStore(persist_directory=persist_directory,
                                                       collection_name=collection_name)

    def retrieve(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the vector store for relevant documents based on the query text.

        Args:
            query_text: The user's query.
            n_results: The number of top results to return.

        Returns:
            A list of dictionaries, each containing a retrieved document and its metadata.
        """
        print(f"\n--- Performing a query: '{query_text}' ---")

        # Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embeddings([query_text])

        # Query the collection
        results = self.vector_store_manager.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )

        # Format and print results
        print(f"\nTop {n_results} results:")
        formatted_results = []
        if results.get('documents') and results['documents'][0]:
            for i, (metadata, doc_content, distance) in enumerate(
                    zip(results['metadatas'][0], results['documents'][0], results['distances'][0])):
                result_item = {
                    "rank": i + 1,
                    "source": metadata.get('source', 'N/A'),
                    "distance": distance,
                    "content": doc_content
                }
                formatted_results.append(result_item)

                print(f"\nResult {i + 1}:")
                print(f"  Source: {result_item['source']}")
                print(f"  Distance: {result_item['distance']:.4f} (lower is better)")
                print(f"  Content: \n{result_item['content'][:500]}...")
        else:
            print("No results found.")

        return formatted_results

"""
def main():
    #Example usage of the RAGRetriever.
    print(">>> Running retriever example... <<<")

    # Check if the database exists before trying to query
    if not os.path.exists(CHROMADB_PATH):
        print("\nERROR: ChromaDB directory not found.")
        print("Please run 'vector_store_creator.py' first to build the database.")
        return

    retriever = Retriever()

    # Example Query
    retriever.retrieve("What are the key components of NLP?")


if __name__ == "__main__":
    import os

    main()
"""
