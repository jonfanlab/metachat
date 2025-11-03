from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMResponse:
    def __init__(self, content: str, raw_response: Any, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None):
        self.content = content
        self.raw_response = raw_response
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

class BaseModel(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    @abstractmethod
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate a response from the model"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass