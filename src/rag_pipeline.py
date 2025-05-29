import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import re

from .config import settings
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .llm_client import OllamaClient

logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """Response from hybrid chat system."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    query: str
    mode: str  # 'rag' or 'chat'
    reasoning: str

class HybridRAGPipeline:
    """Hybrid pipeline supporting both general chat and document-specific RAG."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embedding_manager = EmbeddingManager(settings.EMBEDDING_MODEL)
        self.vector_store = VectorStore(
            persist_directory=settings.VECTOR_DB_PATH,
            collection_name="documents"
        )
        self.llm_client = OllamaClient(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL
        )
        self.is_initialized = False
        self.conversation_history: List[Dict[str, str]] = []
    
    def initialize_knowledge_base(self, document_paths: List[Path]) -> Dict[str, Any]:
        """Initialize the knowledge base with documents."""
        try:
            logger.info("Starting knowledge base initialization...")
            
            # Process documents
            documents = self.document_processor.process_documents(document_paths)
            
            if not documents:
                raise ValueError("No documents were processed successfully")
            
            # Generate embeddings
            texts = [doc.page_content for doc in documents]
            embeddings = self.embedding_manager.embed_texts(texts)
            
            # Store in vector database
            self.vector_store.add_documents(documents, embeddings)
            
            self.is_initialized = True
            
            stats = {
                "total_documents": len(document_paths),
                "total_chunks": len(documents),
                "avg_chunk_size": sum(len(doc.page_content) for doc in documents) / len(documents),
                "embedding_dimension": self.embedding_manager.embedding_dimension
            }
            
            logger.info(f"Knowledge base initialized successfully: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            raise
    
    def chat(
        self, 
        message: str, 
        force_mode: Optional[str] = None,
        top_k: int = None,
        similarity_threshold: float = None
    ) -> ChatResponse:
        """Main chat interface that decides between RAG and general conversation."""
        try:
            # Determine chat mode
            mode, reasoning = self._determine_chat_mode(message, force_mode)
            
            if mode == "rag" and self.is_initialized:
                response = self._handle_rag_query(message, top_k, similarity_threshold)
                response.mode = mode
                response.reasoning = reasoning
            else:
                response = self._handle_general_chat(message)
                response.mode = "chat"
                response.reasoning = reasoning
            
            # Update conversation history
            self._update_conversation_history(message, response.answer)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise
    
    def _determine_chat_mode(self, message: str, force_mode: Optional[str] = None) -> Tuple[str, str]:
        """Determine whether to use RAG or general chat."""
        if force_mode:
            return force_mode, f"Mode forced to {force_mode}"
        
        if not self.is_initialized:
            return "chat", "No documents loaded - using general conversation"
        
        # Keywords that suggest document-specific queries
        document_keywords = [
            "document", "documents", "file", "files", "pdf", "paper", "report", 
            "according to", "based on", "what does it say", "in the text",
            "summary", "summarize", "key points", "main points", "findings",
            "content", "information", "data", "evidence", "source", "sources"
        ]
        
        # Questions that typically relate to documents
        question_patterns = [
            r"what.*(?:document|file|text|paper|report)",
            r"(?:tell me|explain).*(?:about|from).*(?:document|file|text)",
            r"(?:summary|summarize|key points|main points)",
            r"according to.*(?:document|text|file|paper)",
            r"what.*(?:say|mention|state|explain|describe).*(?:about|in|on)",
            r"(?:find|search|look for).*(?:information|data|content)"
        ]
        
        message_lower = message.lower()
        
        # Check for document keywords
        for keyword in document_keywords:
            if keyword in message_lower:
                return "rag", f"Document-related keyword detected: '{keyword}'"
        
        # Check for question patterns
        for pattern in question_patterns:
            if re.search(pattern, message_lower):
                return "rag", f"Document-related pattern detected"
        
        # General conversation indicators
        general_patterns = [
            r"^(hi|hello|hey|greetings)",
            r"how are you",
            r"what can you do",
            r"tell me about yourself",
            r"^(thanks|thank you)",
            r"^(bye|goodbye|see you)",
            r"what.*weather",
            r"tell me.*joke",
            r"^(help|what|how)(?!.*document)"
        ]
        
        for pattern in general_patterns:
            if re.search(pattern, message_lower):
                return "chat", "General conversation pattern detected"
        
        # Default to RAG if documents are available and query seems like a question
        if message.strip().endswith('?') or any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return "rag", "Question detected with documents available - trying RAG first"
        
        return "chat", "Default to general conversation"
    
    def _handle_rag_query(
        self, 
        query: str, 
        top_k: int = None, 
        similarity_threshold: float = None
    ) -> ChatResponse:
        """Handle document-specific queries using RAG."""
        top_k = top_k or settings.TOP_K_RESULTS
        similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        
        logger.info(f"Processing RAG query: {query[:100]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Retrieve relevant documents
        similar_docs = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k
        )
        
        # Log similarity scores for debugging
        logger.info(f"Retrieved {len(similar_docs)} documents with scores: {[round(score, 3) for _, score in similar_docs]}")
        
        # Filter by similarity threshold
        filtered_docs = [
            (doc, score) for doc, score in similar_docs 
            if score >= similarity_threshold
        ]
        
        # If no docs pass threshold, try with lower threshold
        if not filtered_docs and similar_docs:
            logger.warning(f"No documents above threshold {similarity_threshold}. Using lower threshold.")
            min_threshold = 0.3
            filtered_docs = [
                (doc, score) for doc, score in similar_docs 
                if score >= min_threshold
            ]
            
            if not filtered_docs:
                filtered_docs = [similar_docs[0]]
                logger.warning(f"Using top document with score: {similar_docs[0][1]}")
        
        if not filtered_docs:
            # Fallback to general chat if no relevant documents
            logger.info("No relevant documents found, falling back to general chat")
            return self._handle_general_chat(query)
        
        # Prepare context for LLM
        context_texts = [doc.page_content for doc, _ in filtered_docs]
        
        # Generate answer using RAG
        answer = self.llm_client.generate_rag_response(
            prompt=query,
            context=context_texts
        )
        
        # Prepare source information
        sources = []
        for doc, score in filtered_docs:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
                "similarity_score": round(score, 3)
            })
        
        # Calculate confidence score
        confidence_score = sum(score for _, score in filtered_docs) / len(filtered_docs)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence_score=round(confidence_score, 3),
            query=query,
            mode="rag",
            reasoning="RAG mode used with document context"
        )
    
    def _handle_general_chat(self, message: str) -> ChatResponse:
        """Handle general conversation."""
        logger.info(f"Processing general chat: {message[:100]}...")
        
        # Generate response using conversation history
        answer = self.llm_client.generate_chat_response(
            prompt=message,
            conversation_history=self.conversation_history
        )
        
        return ChatResponse(
            answer=answer,
            sources=[],
            confidence_score=1.0,  # High confidence for general chat
            query=message,
            mode="chat",
            reasoning="General conversation mode"
        )
    
    def _update_conversation_history(self, user_message: str, assistant_response: str):
        """Update conversation history."""
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Keep only last 20 messages to manage memory
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "knowledge_base_initialized": self.is_initialized,
            "conversation_history_length": len(self.conversation_history),
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "available_llm_models": self.llm_client.get_available_models()
        }
        
        if self.is_initialized:
            stats.update(self.vector_store.get_collection_stats())
        
        return stats

# Backward compatibility alias
RAGPipeline = HybridRAGPipeline