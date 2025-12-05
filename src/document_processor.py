"""
Document loading, processing, and chunking
"""
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.settings import DocumentConfig


class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Override default chunk size
            chunk_overlap: Override default overlap
        """
        self.chunk_size = chunk_size or DocumentConfig.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or DocumentConfig.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load single PDF file
        
        Args:
            file_path: Path to PDF
        
        Returns:
            List of Document objects (one per page)
        """
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            print(f"✅ Loaded: {file_path} ({len(documents)} pages)")
            return documents
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return []
    
    def load_folder(self, folder_path: str = None) -> List[Document]:
        """
        Load all PDFs from folder
        
        Args:
            folder_path: Path to folder (uses default if None)
        
        Returns:
            List of all documents
        """
        folder = Path(folder_path or DocumentConfig.PDF_FOLDER)
        
        if not folder.exists():
            print(f"❌ Folder not found: {folder}")
            return []
        
        pdf_files = list(folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in {folder}")
            return []
        
        all_documents = []
        for pdf_file in pdf_files:
            docs = self.load_pdf(str(pdf_file))
            all_documents.extend(docs)
        
        print(f"📚 Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"✂️ Created {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks
    
    def process_folder(self, folder_path: str = None) -> List[Document]:
        """
        Complete pipeline: load folder → chunk
        
        Args:
            folder_path: Path to folder
        
        Returns:
            List of chunked documents
        """
        documents = self.load_folder(folder_path)
        if not documents:
            return []
        
        chunks = self.chunk_documents(documents)
        return chunks


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def process_documents(folder_path: str = None) -> List[Document]:
    """
    Quick access to document processing
    
    Args:
        folder_path: Path to PDF folder
    
    Returns:
        Chunked documents ready for embedding
    """
    processor = DocumentProcessor()
    return processor.process_folder(folder_path)