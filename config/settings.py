"""
Application configuration and constants
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# load environmental variables
load_dotenv()

#================================================================
# API CONFIGURATION
#================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API_KEY exists
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables")

#================================================================
# MODEL CONFIGURATION
#================================================================

class ModelConfig:
    """LLM model configuration"""

    # Model selection
    # Model selection
    MODEL_NAME = "llama-3.1-8b-instant"  # Fast & accurate
    # MODEL_NAME = "llama-3.1-70b-versatile"  # More powerful
    # MODEL_NAME = "deepseek-r1-distill-llama-70b"  # Reasoning model
    
    # Generation parameters
    TEMPERATURE = 0.7  # 0.0 = deterministic, 1.5 = creative
    MAX_TOKENS = None  # No limit
    TIMEOUT = None     # No timeout
    MAX_RETRIES = 2    # Retry failed requests

#================================================================
# DOCUMENT PROCESSING CONFIGURATION
#================================================================

class DocumentConfig:
    """Document processing configuration"""

    # Chunking parameters
    CHUNK_SIZE = 1000  # Characters per chunk
    CHUNK_OVERLAP = 200  # Overlap between chunks
    # CHUNK_SEPERATOR = "\n\n" #Seperator between chunks
    # CHUNK_EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Embedding model for chunks
    #CHUNK_EMBEDDING_DIM = 384 # Embedding dimension
    #CHUNK_EMBEDDING_MODEL_KWARGS = {"normalize_embeddings": True} # Embedding model kwargs
    #CHUNK_EMBEDDING_MODEL_KWARGS = {"normalize_embeddings": True} # Embedding model kwargs


    # File paths
    PDF_FOLDER = Path("data/pdf")                 # <-- used by app & processor
    PDF_PATH = PDF_FOLDER                         # <-- alias for backwards compatibility
    VECTOR_DB_PATH = Path("data/index_faiss")

    # Supported file types
    SUPPORTED_EXTENSIONS = [".pdf"]


# ========================================================================
# EMBEDDING CONFIGURATION
# ========================================================================

class EmbeddingConfig:
    """Embedding model configuration"""
    
    # Model selection
    MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Balanced
    # MODEL_NAME = "BAAI/bge-m3"  # More accurate but slower
    # MODEL_NAME = "all-MiniLM-L6-v2"  # Faster but less accurate
    
    # Dimensions (auto-detected by model)
    # bge-large-en-v1.5: 1024
    # bge-m3: 1024
    # all-MiniLM-L6-v2: 384

# ═══════════════════════════════════════════════════════════════
# RETRIEVAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class RetrieverConfig:
    """Vector search configuration"""

    # Search type
    SEARCH_TYPE = "mmr" # Maximal Marginal Relevance
    # SEARCH_TYPE = "similarity" # Similarity search

    # Search parameters
    K = 3 # Number of results to return
    FETCH_K = 4 # Number of candidates before MMR filtering

# ═══════════════════════════════════════════════════════════════
# UI CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class UIConfig:
    """Streamlit UI configuration"""
    
    PAGE_TITLE = "SafeBank Support 🤖"
    PAGE_ICON = "🤖"
    
    # Initial bot message
    WELCOME_MESSAGE = "Hi, I'm your virtual assistant! How can I help you?"
    
    # Chat input placeholder
    INPUT_PLACEHOLDER = "Enter your message here..."
    
    # Loading messages
    LOADING_DOCUMENTS = "Loading documents... (first message takes longer)"
    LOADING_RESPONSE = "Thinking..."
    
    # UI feature toggles (aliases for Features.* to remain backwards-compatible)
    # These keep the UI checks (UIConfig.SHOW_SOURCES, UIConfig.SHOW_THINKING, etc.)
    # working while preserving the Features class as the single source of truth.
    SHOW_SOURCES = True
    SHOW_THINKING = False
    ENABLE_FILE_UPLOAD = False
    SAVE_CONVERSATIONS = False


# ═══════════════════════════════════════════════════════════════
# FEATURE FLAGS
# ═══════════════════════════════════════════════════════════════

class Features:
    """Enable/disable features"""
    
    SHOW_SOURCES = True          # Show source documents
    SHOW_THINKING = False        # Show <think> tags (for reasoning models)
    ENABLE_FILE_UPLOAD = False   # Allow users to upload PDFs
    SAVE_CONVERSATIONS = False   # Save chat history to database


# make UIConfig flags follow Features (keeps them in sync)
UIConfig.SHOW_SOURCES = Features.SHOW_SOURCES
UIConfig.SHOW_THINKING = Features.SHOW_THINKING
UIConfig.ENABLE_FILE_UPLOAD = Features.ENABLE_FILE_UPLOAD
UIConfig.SAVE_CONVERSATIONS = Features.SAVE_CONVERSATIONS    