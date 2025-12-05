"""
Streamlit user interface
"""
import sys
import os
from pathlib import Path

# CRITICAL: Add parent directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Verify path was added
print(f"📁 Project root: {project_root}")
print(f"🐍 Python path includes: {str(project_root) in sys.path}")

import streamlit as st

# Import modules
try:
    from src.llm import get_llm
    from src.document_processor import process_documents
    from src.vector_store import create_vectorstore, load_vectorstore
    from src.rag_chain import create_rag_chain
    from src.chat_manager import ChatManager
    from config.settings import UIConfig, DocumentConfig, ModelConfig
    from langchain_core.messages import HumanMessage, AIMessage
    print("✅ All imports successful!")
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=UIConfig.PAGE_TITLE,
    page_icon=UIConfig.PAGE_ICON,
    layout="wide"
)

st.title(UIConfig.PAGE_TITLE)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR - SETTINGS & CONTROLS
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_option = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "deepseek-r1-distill-llama-70b"
        ],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.5,
        value=ModelConfig.TEMPERATURE,
        step=0.1,
        help="Lower = more factual, Higher = more creative"
    )
    
    st.divider()
    
    # Document info
    st.header("📚 Documents")
    pdf_folder = Path(DocumentConfig.PDF_FOLDER)
    
    if pdf_folder.exists():
        pdf_count = len(list(pdf_folder.glob("*.pdf")))
        st.info(f"📄 {pdf_count} PDF(s) loaded")
    else:
        st.warning("⚠️ No PDF folder found")
    
    # Reindex button
    if st.button("🔄 Reindex Documents", help="Rebuild vector database"):
        if "vectorstore_manager" in st.session_state:
            del st.session_state.vectorstore_manager
        st.success("✅ Will reindex on next message")
        st.rerun()
    
    st.divider()
    
    # Chat controls
    st.header("💬 Chat Controls")
    
    if st.button("🗑️ Clear Chat", help="Start new conversation"):
        st.session_state.chat_history = [
            AIMessage(content=UIConfig.WELCOME_MESSAGE)
        ]
        st.success("✅ Chat cleared")
        st.rerun()
    
    # Export chat
    if st.button("💾 Export Chat", help="Download conversation"):
        if "chat_manager" in st.session_state:
            history = st.session_state.chat_manager.export_history()
            st.download_button(
                "Download JSON",
                data=str(history),
                file_name="chat_history.json",
                mime="application/json"
            )
    
    st.divider()
    
    # Statistics
    st.header("📊 Statistics")
    if "chat_history" in st.session_state:
        msg_count = len(st.session_state.chat_history)
        st.metric("Total Messages", msg_count)
        
        user_msgs = sum(1 for msg in st.session_state.chat_history 
                       if isinstance(msg, HumanMessage))
        st.metric("Your Questions", user_msgs)


# ═══════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content=UIConfig.WELCOME_MESSAGE)
        ]
    
    # Chat manager
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
        st.session_state.chat_manager.chat_history = st.session_state.chat_history
    
    # LLM
    if "llm" not in st.session_state:
        st.session_state.llm = get_llm(model_option, temperature)
    
    # Vector store manager
    if "vectorstore_manager" not in st.session_state:
        st.session_state.vectorstore_manager = None
    
    # RAG chain
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None


initialize_session_state()


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_or_create_vectorstore():
    """
    Load existing vector store or create new one
    Uses Streamlit caching for performance
    """
    vector_db_path = Path(DocumentConfig.VECTOR_DB_PATH)
    
    # Try to load existing
    if vector_db_path.exists():
        try:
            st.info("📂 Loading existing vector database...")
            manager = load_vectorstore()
            st.success("✅ Vector database loaded")
            return manager
        except Exception as e:
            st.warning(f"⚠️ Could not load vector DB: {e}")
    
    # Create new
    st.info("🔨 Creating new vector database...")
    
    with st.spinner(UIConfig.LOADING_DOCUMENTS):
        # Process documents
        chunks = process_documents()
        
        if not chunks:
            st.error("❌ No documents found to process")
            return None
        
        # Create vector store
        manager = create_vectorstore(chunks, save=True)
        st.success("✅ Vector database created")
        return manager


def get_or_create_rag_chain():
    """Get or create RAG chain"""
    
    # Load vector store if not loaded
    if st.session_state.vectorstore_manager is None:
        st.session_state.vectorstore_manager = load_or_create_vectorstore()
        
        if st.session_state.vectorstore_manager is None:
            return None
    
    # Create RAG chain if not created
    if st.session_state.rag_chain is None:
        retriever = st.session_state.vectorstore_manager.get_retriever()
        st.session_state.rag_chain = create_rag_chain(
            st.session_state.llm,
            retriever,
            industry="banking"  # Can be made configurable
        )
    
    return st.session_state.rag_chain


# ═══════════════════════════════════════════════════════════════
# DISPLAY CHAT HISTORY
# ═══════════════════════════════════════════════════════════════

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)


# ═══════════════════════════════════════════════════════════════
# CHAT INPUT & RESPONSE
# ═══════════════════════════════════════════════════════════════

user_input = st.chat_input(UIConfig.INPUT_PLACEHOLDER)

if user_input:
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Add to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Get response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner(UIConfig.LOADING_RESPONSE):
            # Get RAG chain
            rag_chain = get_or_create_rag_chain()
            
            if rag_chain is None:
                st.error("❌ Could not initialize RAG system. Please check your documents.")
            else:
                # Get response
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                
                # Extract answer
                answer = response["answer"]
                
                # Clean <think> tags
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()
                
                # Display answer
                st.markdown(answer)
                
                # Add to history
                st.session_state.chat_history.append(AIMessage(content=answer))
                
                # Show sources (if enabled)
                if UIConfig.SHOW_SOURCES and "context" in response:
                    with st.expander("📄 View Sources"):
                        for i, doc in enumerate(response["context"], 1):
                            st.write(f"**Source {i}** (Page {doc.metadata.get('page', 'N/A')})")
                            st.write(doc.page_content[:300] + "...")
                            st.divider()


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.divider()
st.caption("🤖 Powered by RAG • Built with Streamlit & LangChain")