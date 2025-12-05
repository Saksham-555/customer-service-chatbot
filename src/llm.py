"""
Language model initialization and management
"""
from langchain_groq import ChatGroq
from config.settings import ModelConfig, GROQ_API_KEY

class LLMManager:
    """Manages language model initialization and configuration"""

    def __init__(self, model_name: str = None, temperature: float = None):
        """
        Initialize LLM manager
        
        Args:
            model_name: Override default model
            temperature: Override default temperature
        """
        self.model_name = model_name or ModelConfig.MODEL_NAME
        self.temperature = temperature or ModelConfig.TEMPERATURE
        self._llm = None

    def get_llm(self) -> ChatGroq:
        """
        Get or create LLM instance (singleton pattern)
        
        Returns:
            Configured ChatGroq instance
        """
        if self._llm is None:
            self._llm = self._initialize_llm()
        return self._llm

    def _initialize_llm(self) -> ChatGroq:
        """
        Create new LLM instance
        
        Returns:
            ChatGroq instance
        """
        llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=ModelConfig.MAX_TOKENS,
            timeout=ModelConfig.TIMEOUT,
            max_retries=ModelConfig.MAX_RETRIES,
            groq_api_key=GROQ_API_KEY
        )
        
        print(f"✅ Loaded model: {self.model_name} (temp={self.temperature})")
        return llm

    def change_model(self, model_name: str):
        """
        Switch to different model
        
        Args:
            model_name: New model name
        """
        self.model_name = model_name
        self._llm = None  # Force re-initialization
        print(f"🔄 Switched to model: {model_name}")
    
    def change_temperature(self, temperature: float):
        """
        Adjust temperature
        
        Args:
            temperature: New temperature value
        """
        self.temperature = temperature
        self._llm = None  # Force re-initialization
        print(f"🔄 Changed temperature to: {temperature}")


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════

def get_llm(model_name: str = None, temperature: float = None) -> ChatGroq:
    """
    Quick access to LLM instance
    
    Args:
        model_name: Optional model override
        temperature: Optional temperature override
    
    Returns:
        ChatGroq instance
    """
    manager = LLMManager(model_name, temperature)
    return manager.get_llm()            
