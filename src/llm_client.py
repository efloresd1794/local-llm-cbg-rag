import logging
import requests
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama LLM."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2:7b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found. Available models: {model_names}")
            else:
                logger.info(f"Connected to Ollama. Using model: {self.model}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            raise
    
    def generate_rag_response(
        self, 
        prompt: str, 
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate response from the LLM using RAG context."""
        try:
            # Build the RAG prompt with context
            full_prompt = self._build_rag_prompt(prompt, context)
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            raise
    
    def generate_chat_response(
        self, 
        prompt: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.8,
        max_tokens: int = 500
    ) -> str:
        """Generate response for general conversation."""
        try:
            # Build conversational prompt
            full_prompt = self._build_chat_prompt(prompt, conversation_history)
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            raise
    
    def _build_rag_prompt(self, query: str, context: List[str]) -> str:
        """Build prompt with document context for RAG."""
        context_text = "\n\n".join(context)
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context_text}

Question: {query}

Answer the question based on the provided context. Be informative and helpful. If the answer cannot be found in the context, say "I don't have enough information in the provided documents to answer this question."

Answer:"""
        
        return prompt
    
    def _build_chat_prompt(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Build prompt for general conversation."""
        system_prompt = """You are a friendly and helpful AI assistant. You can have natural conversations about any topic, answer questions, help with tasks, and provide information. Be conversational, engaging, and helpful."""
        
        prompt = f"{system_prompt}\n\n"
        
        # Add conversation history if available
        if conversation_history:
            for message in conversation_history[-6:]:  # Keep last 6 messages for context
                role = message.get('role', '')
                content = message.get('content', '')
                if role == 'user':
                    prompt += f"Human: {content}\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n"
        
        prompt += f"Human: {query}\n\nAssistant:"
        
        return prompt
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get('models', [])
            return [model['name'] for model in models]
            
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []