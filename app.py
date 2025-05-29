import streamlit as st
import logging
from pathlib import Path
from typing import List
import os
import tempfile
import shutil

from src.config import settings
from src.rag_pipeline import HybridRAGPipeline
from src.evaluation import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🤖 Hybrid RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize RAG pipeline (cached)."""
    return HybridRAGPipeline()

@st.cache_data
def load_sample_questions():
    """Load sample questions for testing."""
    return [
        "What is the main topic of the documents?",
        "Can you summarize the key findings?",
        "What are the most important points mentioned?",
        "Are there any specific recommendations provided?",
        "What conclusions can be drawn from the information?"
    ]

def main():
    """Main Streamlit application."""
    st.title("🤖 Hybrid RAG Chatbot")
    st.markdown("**Chat naturally + Ask about your documents**")
    
    # Initialize RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = initialize_rag_pipeline()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents (Optional)",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload documents to enable document-specific Q&A. You can still chat without documents!"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                process_documents(uploaded_files)
        
        # Add clear knowledge base button
        if st.session_state.rag_pipeline.is_initialized:
            if st.button("🗑️ Clear Documents", type="secondary"):
                clear_knowledge_base()
        
        # Add clear conversation button
        if st.session_state.rag_pipeline.conversation_history:
            if st.button("💬 Clear Chat History", type="secondary"):
                clear_conversation_history()
        
        st.divider()
        
        # Chat mode selection
        st.header("🎯 Chat Mode")
        
        mode_options = ["Auto (Smart Detection)", "Force RAG Mode", "Force Chat Mode"]
        selected_mode = st.selectbox(
            "Mode Selection",
            mode_options,
            help="Auto mode intelligently detects when to use documents vs general chat"
        )
        
        # Map selection to mode
        force_mode = None
        if selected_mode == "Force RAG Mode":
            force_mode = "rag"
        elif selected_mode == "Force Chat Mode":
            force_mode = "chat"
        
        st.divider()
        
        # System configuration
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = st.session_state.rag_pipeline.llm_client.get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "LLM Model",
                available_models,
                index=0 if settings.LLM_MODEL not in available_models else available_models.index(settings.LLM_MODEL)
            )
            if selected_model != settings.LLM_MODEL:
                st.session_state.rag_pipeline.llm_client.model = selected_model
        
        # Retrieval settings (only show when documents are loaded)
        if st.session_state.rag_pipeline.is_initialized:
            st.subheader("🔍 RAG Settings")
            top_k = st.slider("Top K Results", 1, 10, settings.TOP_K_RESULTS)
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, settings.SIMILARITY_THRESHOLD, 0.1)
        else:
            top_k = settings.TOP_K_RESULTS
            similarity_threshold = settings.SIMILARITY_THRESHOLD
        
        st.divider()
        
        # System stats
        st.header("📊 System Stats")
        stats = st.session_state.rag_pipeline.get_system_stats()
        
        # Display key stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", "✅" if stats['knowledge_base_initialized'] else "❌")
            if stats['knowledge_base_initialized']:
                st.metric("Total Documents", stats.get('total_documents', 0))
        
        with col2:
            st.metric("Chat History", stats['conversation_history_length'])
            st.metric("Model", stats['llm_model'].split(':')[0])
        
        # Debug mode toggle
        if st.checkbox("🐛 Debug Mode"):
            st.session_state.debug_mode = True
        else:
            st.session_state.debug_mode = False
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                if message["role"] == "assistant":
                    # Show mode indicator
                    mode_emoji = "🔍" if message.get("mode") == "rag" else "💬"
                    mode_text = "Document Q&A" if message.get("mode") == "rag" else "General Chat"
                    st.caption(f"{mode_emoji} {mode_text}")
                    
                    # Show debug info if enabled
                    if getattr(st.session_state, 'debug_mode', False) and "reasoning" in message:
                        with st.expander("🐛 Debug Info"):
                            st.write(f"**Mode:** {message.get('mode', 'unknown')}")
                            st.write(f"**Reasoning:** {message.get('reasoning', 'N/A')}")
                            if message.get('confidence_score'):
                                st.write(f"**Confidence:** {message['confidence_score']}")
                    
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander("📄 Sources"):
                            for i, source in enumerate(message["sources"]):
                                st.write(f"**Source {i+1}** (Score: {source['similarity_score']})")
                                st.write(f"*{source['metadata'].get('filename', 'Unknown')}*")
                                st.write(source["content"])
    
    with col2:
        st.header("🧪 Testing")
        
        # Quick start examples
        st.subheader("💡 Try These Examples")
        
        # General chat examples
        st.write("**General Chat:**")
        general_examples = [
            "Hello! How are you?",
            "Tell me a joke",
            "What can you help me with?",
            "How does machine learning work?",
            "What's the weather like?"
        ]
        
        for example in general_examples:
            if st.button(example, key=f"general_{hash(example)}", help="General conversation"):
                handle_chat_input(example, force_mode, top_k, similarity_threshold)
        
        # Document examples (only show if documents are loaded)
        if st.session_state.rag_pipeline.is_initialized:
            st.write("**Document Questions:**")
            doc_examples = load_sample_questions()
            
            for example in doc_examples:
                if st.button(example, key=f"doc_{hash(example)}", help="Document-specific question"):
                    handle_chat_input(example, force_mode, top_k, similarity_threshold)
        else:
            st.info("Upload documents to see document-specific examples!")
        
        st.divider()
        
        # Evaluation section
        if st.session_state.rag_pipeline.is_initialized:
            st.subheader("📊 RAG Evaluation")
            if st.button("🔍 Run Evaluation"):
                run_evaluation()
    
    # Chat input - MUST be outside columns due to Streamlit limitation
    if prompt := st.chat_input("Chat naturally or ask about your documents..."):
        handle_chat_input(prompt, force_mode, top_k, similarity_threshold)

def clear_conversation_history():
    """Clear conversation history."""
    try:
        st.session_state.rag_pipeline.clear_conversation_history()
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing chat history: {str(e)}")

def clear_knowledge_base():
    """Clear the knowledge base."""
    try:
        with st.spinner("Clearing knowledge base..."):
            st.session_state.rag_pipeline.vector_store.clear_collection()
            st.session_state.rag_pipeline.is_initialized = False
            
        st.success("Documents cleared successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error clearing knowledge base: {str(e)}")
        logger.error(f"Clear knowledge base error: {str(e)}")

def process_documents(uploaded_files):
    """Process uploaded documents."""
    try:
        with st.spinner("Processing documents..."):
            # Save uploaded files temporarily
            temp_dir = Path(tempfile.mkdtemp())
            document_paths = []
            
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                document_paths.append(file_path)
            
            # Initialize knowledge base
            stats = st.session_state.rag_pipeline.initialize_knowledge_base(document_paths)
            
            # Clean up temporary files
            shutil.rmtree(temp_dir)
            
            st.success("Documents processed successfully!")
            st.json(stats)
            
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        logger.error(f"Document processing error: {str(e)}")

def handle_chat_input(prompt: str, force_mode: str, top_k: int, similarity_threshold: float):
    """Handle user chat input."""
    try:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_pipeline.chat(
                    message=prompt,
                    force_mode=force_mode,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
            
            st.write(response.answer)
            
            # Show mode indicator
            mode_emoji = "🔍" if response.mode == "rag" else "💬"
            mode_text = "Document Q&A" if response.mode == "rag" else "General Chat"
            st.caption(f"{mode_emoji} {mode_text}")
            
            # Show debug info if enabled
            if getattr(st.session_state, 'debug_mode', False):
                with st.expander("🐛 Debug Information"):
                    st.write(f"**Mode:** {response.mode}")
                    st.write(f"**Reasoning:** {response.reasoning}")
                    st.write(f"**Confidence Score:** {response.confidence_score}")
                    st.write(f"**Number of Sources:** {len(response.sources)}")
                    
                    if response.sources:
                        st.write("**Source Scores:**")
                        for i, source in enumerate(response.sources):
                            st.write(f"- Source {i+1}: {source['similarity_score']}")
            
            # Display sources
            if response.sources:
                with st.expander("📄 Sources"):
                    for i, source in enumerate(response.sources):
                        st.write(f"**Source {i+1}** (Score: {source['similarity_score']})")
                        st.write(f"*{source['metadata'].get('filename', 'Unknown')}*")
                        st.write(source["content"])
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "mode": response.mode,
            "reasoning": response.reasoning,
            "confidence_score": response.confidence_score
        })
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        logger.error(f"Chat response error: {str(e)}")

def run_evaluation():
    """Run system evaluation."""
    try:
        if not st.session_state.rag_pipeline.is_initialized:
            st.error("Please upload documents first!")
            return
        
        with st.spinner("Running evaluation..."):
            evaluator = RAGEvaluator(st.session_state.rag_pipeline)
            sample_questions = load_sample_questions()
            
            # Convert to old query format for compatibility
            responses = []
            for question in sample_questions:
                response = st.session_state.rag_pipeline.chat(question, force_mode="rag")
                # Convert to old format for evaluator
                old_response = type('RAGResponse', (), {
                    'confidence_score': response.confidence_score,
                    'sources': response.sources,
                    'answer': response.answer
                })()
                responses.append(old_response)
            
            # Calculate basic metrics
            confidence_scores = [r.confidence_score for r in responses]
            successful_retrievals = sum(1 for r in responses if r.sources)
            high_confidence_responses = sum(1 for score in confidence_scores if score > 0.7)
            
            st.success("Evaluation completed!")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            with col2:
                retrieval_coverage = successful_retrievals / len(sample_questions) if sample_questions else 0
                st.metric("Retrieval Coverage", f"{retrieval_coverage:.3f}")
            with col3:
                response_relevance = high_confidence_responses / len(sample_questions) if sample_questions else 0
                st.metric("Response Relevance", f"{response_relevance:.3f}")
            
            # Sample responses
            st.subheader("📝 Sample Responses")
            for i, (question, response) in enumerate(zip(sample_questions[:3], responses[:3])):
                with st.expander(f"Q: {question}"):
                    st.write(f"**Answer:** {response.answer[:200]}...")
                    st.write(f"**Confidence:** {response.confidence_score}")
                    st.write(f"**Sources Found:** {len(response.sources)}")
            
    except Exception as e:
        st.error(f"Error running evaluation: {str(e)}")
        logger.error(f"Evaluation error: {str(e)}")

if __name__ == "__main__":
    main()