"""
Unit tests for document processor
"""
import pytest
from pathlib import Path
from src.document_processor import DocumentProcessor


class TestDocumentProcessor:
    
    @pytest.fixture
    def processor(self):
        return DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    def test_initialization(self, processor):
        """Test processor initialization"""
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
    
    def test_load_pdf(self, processor):
        """Test PDF loading"""
        # Create test PDF or use existing
        test_pdf = Path("data/pdfs/test.pdf")
        
        if test_pdf.exists():
            docs = processor.load_pdf(str(test_pdf))
            assert len(docs) > 0
            assert all(hasattr(doc, 'page_content') for doc in docs)
    
    def test_chunk_documents(self, processor):
        """Test document chunking"""
        from langchain.schema import Document
        
        # Create test document
        test_doc = Document(page_content="A" * 1000, metadata={"source": "test"})
        
        # Chunk it
        chunks = processor.chunk_documents([test_doc])
        
        # Verify chunks
        assert len(chunks) > 1  # Should split into multiple chunks
        assert all(len(chunk.page_content) <= 500 for chunk in chunks)