from typing import List, Dict, Optional
from together import Together
from .base import BaseModel, LLMResponse

class LlamaModel(BaseModel):
    """Llama API model implementation"""
    
    # Maximum number of tokens the model can generate in the response
    MODEL_MAX_TOKENS = {
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 2048,
        "meta-llama/Meta-Llama-3.1-70B-Instruct": 2048,
    }
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = Together()
        # self.max_tokens = self.MODEL_MAX_TOKENS.get(model_name, 2048)
    
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate a response using the Llama API"""
        
        # Create API request
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            # max_tokens=min(max_tokens, self.max_tokens) if max_tokens else self.max_tokens,
            stream=False
        )

        # api_request = {'model': 'llama3.3-70b', 'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'What is 2+2?'}], 'temperature': 0.0, 'stream': False}
        
        # Extract the response content
        content = response.choices[0].message.content
        
        return LLMResponse(
            content=content,
            raw_response=response,
            input_tokens=response.usage.prompt_tokens if hasattr(response, 'usage') else None,
            output_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else None
        )
    
    def count_tokens(self, text: str) -> int:
        # Rough token estimate
        return len(text.split()) * 1.3