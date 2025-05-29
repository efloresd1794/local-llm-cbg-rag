import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document
import uuid

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str, collection_name: str = "documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def clear_collection(self):
        """Clear all documents from collection."""
        try:
            # Delete and recreate collection to clear it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.warning(f"Could not clear collection: {str(e)}")
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents with their embeddings to the vector store."""
        try:
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            # Generate unique IDs using UUID to avoid conflicts
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata
            )
            
            documents_with_scores = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (higher is better)
                    similarity_score = 1 - distance
                    
                    document = Document(
                        page_content=doc_text,
                        metadata=metadata
                    )
                    
                    documents_with_scores.append((document, similarity_score))
            
            logger.info(f"Found {len(documents_with_scores)} similar documents")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            return {
                "total_documents": self.collection.count(),
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")