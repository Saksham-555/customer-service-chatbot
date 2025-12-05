"""
Chat history and conversation management
"""
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class ChatManager:
    """Manages conversation history and interactions"""
    
    def __init__(self):
        """Initialize chat manager"""
        self.chat_history: List[BaseMessage] = []
    
    def add_user_message(self, message: str):
        """
        Add user message to history
        
        Args:
            message: User's message
        """
        self.chat_history.append(HumanMessage(content=message))
    
    def add_ai_message(self, message: str):
        """
        Add AI message to history
        
        Args:
            message: AI's response
        """
        self.chat_history.append(AIMessage(content=message))
    
    def get_history(self) -> List[BaseMessage]:
        """
        Get full chat history
        
        Returns:
            List of messages
        """
        return self.chat_history
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        print("🗑️ Chat history cleared")
    
    def get_last_n_messages(self, n: int) -> List[BaseMessage]:
        """
        Get last N messages (for context window management)
        
        Args:
            n: Number of recent messages
        
        Returns:
            Last N messages
        """
        return self.chat_history[-n:] if len(self.chat_history) > n else self.chat_history
    
    def chat(self, rag_chain, user_input: str) -> str:
        """
        Complete chat interaction
        
        Args:
            rag_chain: RAG chain instance
            user_input: User's question
        
        Returns:
            AI's response
        """
        # Add user message
        self.add_user_message(user_input)
        
        # Get response from RAG chain
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })
        
        # Extract answer
        answer = response["answer"]
        
        # Clean up <think> tags if present
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        else:
            answer = answer.strip()
        
        # Add AI message
        self.add_ai_message(answer)
        
        return answer
    
    def export_history(self) -> List[Dict[str, str]]:
        """
        Export history as JSON-serializable format
        
        Returns:
            List of message dictionaries
        """
        return [
            {
                "role": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content
            }
            for msg in self.chat_history
        ]
    
    def import_history(self, history: List[Dict[str, str]]):
        """
        Import history from JSON format
        
        Args:
            history: List of message dictionaries
        """
        self.chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "human"
            else AIMessage(content=msg["content"])
            for msg in history
        ]
        print(f"📥 Imported {len(self.chat_history)} messages")


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════

def create_chat_manager() -> ChatManager:
    """
    Quick chat manager creation
    
    Returns:
        ChatManager instance
    """
    return ChatManager()