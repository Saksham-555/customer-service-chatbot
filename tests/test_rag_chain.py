"""
Unit tests for RAG chain
"""
import pytest
from unittest.mock import Mock, MagicMock
from src.rag_chain import RAGChainBuilder


class TestRAGChain:
    
    @pytest.fixture
    def mock_llm(self):
        return Mock()
    
    @pytest.fixture
    def mock_retriever(self):
        return Mock()
    
    @pytest.fixture
    def builder(self, mock_llm, mock_retriever):
        return RAGChainBuilder(mock_llm, mock_retriever)
    
    def test_initialization(self, builder):
        """Test RAG chain builder initialization"""
        assert builder.llm is not None
        assert builder.retriever is not None
        assert builder.system_prompt is not None
    
    def test_build_rag_chain(self, builder):
        """Test complete RAG chain building"""
        chain = builder.build_rag_chain()
        assert chain is not None