"""
Vector store management (embeddings + FAISS)
"""
from typing import List, Optional
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config.settings import EmbeddingConfig, DocumentConfig, RetrieverConfig


class VectorStoreManager:
    """Manages embeddings and FAISS vector store"""
    
    def __init__(self, embedding_model: str = None):
        """
        Initialize vector store manager
        
        Args:
            embedding_model: Override default embedding model
        """
        self.embedding_model_name = embedding_model or EmbeddingConfig.MODEL_NAME
        self.embeddings = self._initialize_embeddings()
        self.vectorstore = None
    
    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Create embedding model
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        print(f"📥 Loading embedding model: {self.embedding_model_name}")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )
        print("✅ Embedding model loaded")
        return embeddings
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create new vector store from documents
        
        Args:
            documents: List of chunked documents
        
        Returns:
            FAISS vectorstore
        """
        if not documents:
            raise ValueError("No documents provided")
        
        print(f"🔢 Creating embeddings for {len(documents)} chunks...")
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embeddings
        )
        print("✅ Vector store created")
        return self.vectorstore
    
    def save_vectorstore(self, path: str = None):
        """
        Save vector store to disk
        
        Args:
            path: Save location (uses default if None)
        """
        if self.vectorstore is None:
            raise ValueError("No vector store to save")
        
        save_path = path or DocumentConfig.VECTOR_DB_PATH
        self.vectorstore.save_local(save_path)
        print(f"💾 Vector store saved to: {save_path}")
    
    def load_vectorstore(self, path: str = None) -> FAISS:
        """
        Load vector store from disk
        
        Args:
            path: Load location (uses default if None)
        
        Returns:
            FAISS vectorstore
        """
        load_path = path or DocumentConfig.VECTOR_DB_PATH
        
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Vector store not found at: {load_path}")
        
        print(f"📂 Loading vector store from: {load_path}")
        self.vectorstore = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Vector store loaded")
        return self.vectorstore
    
    def get_retriever(
        self,
        search_type: str = None,
        k: int = None,
        fetch_k: int = None
    ):
        """
        Create retriever from vector store
        
        Args:
            search_type: 'similarity' or 'mmr'
            k: Number of results
            fetch_k: Candidates for MMR
        
        Returns:
            Configured retriever
        """
        if self.vectorstore is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        search_type = search_type or RetrieverConfig.SEARCH_TYPE
        k = k or RetrieverConfig.K
        fetch_k = fetch_k or RetrieverConfig.FETCH_K
        
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={'k': k, 'fetch_k': fetch_k}
        )
        
        print(f"🔍 Retriever configured: {search_type} (k={k}, fetch_k={fetch_k})")
        return retriever
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Direct similarity search
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("No vector store available")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def create_vectorstore(documents: List[Document], save: bool = True):
    """
    Quick vector store creation
    
    Args:
        documents: Chunked documents
        save: Whether to save to disk
    
    Returns:
        VectorStoreManager instance
    """
    manager = VectorStoreManager()
    manager.create_vectorstore(documents)
    
    if save:
        manager.save_vectorstore()
    
    return manager

def load_vectorstore(path: str = None):
    """
    Quick vector store loading
    
    Args:
        path: Custom path (optional)
    
    Returns:
        VectorStoreManager instance
    """
    manager = VectorStoreManager()
    manager.load_vectorstore(path)
    return manager