from typing import List, Dict, Optional
import openai
from openai import AsyncOpenAI
from .base import BaseModel, LLMResponse

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            raw_response=response
        )
    
    def count_tokens(self, text: str) -> int:
        # #TODO: Implement token counting logic
        pass