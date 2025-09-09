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
        self.vector_store = VectorStore(persist_directory=persist_directory,
                                                       collection_name=collection_name)

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    max_distance = max(distances)
                    min_distance = min(distances)
                    similarity_score = 1 - ((distance - min_distance) / (max_distance - min_distance))  # Normalize to [0,1]
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


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