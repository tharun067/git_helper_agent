## Imports
import os
import uuid
from typing import List, Any, Dict
import numpy as np
from git import Repo

## Langchain and Community Imports
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Embedding and VectorDB Imports
from sentence_transformers import SentenceTransformer
import chromadb

CHROMADB_PATH = "./vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "git_documents"



## Embedding data
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """ Initialize the embedding manager
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """ Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f" Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f" Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """ Generate embedding for a list of texts

        Args:
            texts: List of texts to generate embeddings
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """

        if not self.model:
            raise ValueError("Model not loaded")

        print(f" Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f" Generated embeddings with shape {embeddings.shape}")
        return embeddings

## Vector store
class VectorStore:
    """ Manages document embeddings in a ChormaDb vector store"""
    def __init__(self, collection_name: str = "git_documents", persist_directory: str = "./vector_store"):
        """ Initialize the vector store
        Args:
            collection_name: Name of the chormadb collection
            persist_directory: Directory to save the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """ Initialize ChormaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":"Git documents embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f" Error initializing chormadb client {self.collection_name}: {e}")
            raise


    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """ Add documents and their embeddings to the vector store

        Args:
            documents: List of git documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f" Adding: {len(documents)} documents to vector store...")

        # prepare data for chormaDb
        ids =[]
        metadatas = []
        document_text = []
        embeddings_list = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            document_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())


        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=document_text,
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f" Error adding {len(documents)} documents to vector store: {e}")
            raise

    def clear_collection(self):
        # Deletes all embeddings and documents from the collection.
        print(f"--- Clearing collection {self.collection_name}---")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"Collections cleared successfully.\n")

# Pipeline to add and retrieve from vector store
class GitVectorStore:
    """ A Complete pipeline to create and query a vector store from a git repository"""

    def __init__(self, repo_url: str, clone_path:str, persist_dir: str = "./vector_store", collection_name: str = "git_documents",model_name: str = "all-MiniLM-L6-v2", chunk_size:int = 1000, chunk_overlap:int =200):
        self.repo_url = repo_url
        self.clone_path = clone_path
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name

        self.embedding_manager = None
        self.vector_store_manager = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _initialize_manager(self):
        """ Initialize the embedding and vector store when they are needed."""
        if not self.embedding_manager:
            self.embedding_manager = EmbeddingManager(model_name=self.model_name)
        if not self.vector_store_manager:
            self.vector_store_manager = VectorStore(persist_directory=self.persist_dir,collection_name=self.collection_name)

    def _setup_repository(self):
        """ Clones the repository if it doesn't exist, or pulls the latest changes if it does."""
        if os.path.exists(self.clone_path):
            print(f" Repository {self.clone_path} already exists, Pulling latest changes...")
            try:
                repo = Repo(self.clone_path)
                origin = repo.remote('origin')
                origin.pull()
                print(f" Successfully pulled {self.clone_path}.")
            except Exception as e:
                print(f" Error pulling {self.clone_path}: {e}")
        else:
            print(f" Repository {self.clone_path} does not exist, creating it...")
            Repo.clone_from(self.repo_url, self.clone_path)
            print(f" Successfully Cloned {self.clone_path}.")

    def _load_from_local_repo(self) ->  List[Any]:
        """ Loads files from the local repository and returns them as a list of documents. """
        print(f" Loading documents from local path: {self.clone_path}")
        loader = GitLoader(clone_url=self.repo_url, repo_path=self.clone_path)
        documents = loader.load()
        print(f" Successfully loaded {len(documents)} documents from the repository.")
        return documents

    def _split_documents(self, documents: List[Any]) -> List[Any]:
        """ Splits documents into chunks and returns them as a list of chunked documents. """
        print(" Splitting documents into chunks...")
        chunked_docs = self.text_splitter.split_documents(documents)
        print(f" Split documents into {len(chunked_docs)} chunks.\n")
        return chunked_docs


    def run_pipeline(self):
        self._initialize_manager()
        self._setup_repository()

        documents = self._load_from_local_repo()
        chunked_documents = self._split_documents(documents)
        doc_texts = [doc.page_content for doc in chunked_documents]
        embeddings = self.embedding_manager.generate_embeddings(doc_texts)
        self.vector_store_manager.add_documents(chunked_documents,embeddings)
        return " Successfully added documents to vector store."

## Serves as example how to run it
def main():
    pipeline = GitVectorStore(
        repo_url="https://github.com/tharun067/NLP_assignments.git",
        clone_path="V:\\Clones\\NLP_assignments",
        persist_dir="./vector_store",
        collection_name= "git_documents",
        model_name= "all-MiniLM-L6-v2",
        chunk_size= 1000,
        chunk_overlap= 200
    )
    pipeline.run_pipeline()
if __name__ == "__main__":
    main()