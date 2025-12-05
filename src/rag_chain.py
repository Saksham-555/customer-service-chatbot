"""
RAG chain construction and management
"""
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.prompts import get_system_prompt, CONTEXTUALIZATION_PROMPT


class RAGChainBuilder:
    """Builds and manages RAG chains"""
    
    def __init__(self, llm, retriever, industry: str = "general"):
        """
        Initialize RAG chain builder
        
        Args:
            llm: Language model instance
            retriever: Vector store retriever
            industry: Industry for specialized prompts
        """
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = get_system_prompt(industry)
    
    def build_contextualization_chain(self):
        """
        Create history-aware retriever
        
        Returns:
            History-aware retriever chain
        """
        # Prompt for reformulating questions
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZATION_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "Question: {input}")
        ])
        
        # Create chain
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.retriever,
            prompt=context_prompt
        )
        
        return history_aware_retriever
    
    def build_qa_chain(self):
        """
        Create Q&A chain
        
        Returns:
            Q&A chain
        """
        # Q&A prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Question: {input}\n\nContext: {context}")
        ])
        
        # Create chain
        qa_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt
        )
        
        return qa_chain
    
    def build_rag_chain(self):
        """
        Build complete RAG chain
        
        Returns:
            Full RAG chain with history awareness
        """
        # Build sub-chains
        history_aware_retriever = self.build_contextualization_chain()
        qa_chain = self.build_qa_chain()
        
        # Combine into full chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            qa_chain
        )
        
        print("✅ RAG chain built successfully")
        return rag_chain


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════

def create_rag_chain(llm, retriever, industry: str = "general"):
    """
    Quick RAG chain creation
    
    Args:
        llm: Language model
        retriever: Vector retriever
        industry: Industry type for specialized prompts
    
    Returns:
        Complete RAG chain
    """
    builder = RAGChainBuilder(llm, retriever, industry)
    return builder.build_rag_chain()