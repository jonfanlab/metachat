from typing import List, Dict, Optional
from anthropic import AsyncAnthropic
from .base import BaseModel, LLMResponse

class AnthropicModel(BaseModel):
    # Maximum number of tokens the model can generate in the response
    MODEL_MAX_TOKENS = {
        "claude-3-opus": 4096,
        "claude-3-sonnet": 4096,
        "claude-3-haiku": 4096,
        "claude-3-5-sonnet": 8192,
        "claude-3-5-haiku": 8192
    }
    
    # Context window sizes (for reference)
    MODEL_CONTEXT_WINDOWS = {
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-3-5-sonnet": 200000,
        "claude-3-5-haiku": 200000
    }
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = AsyncAnthropic(api_key=api_key)
        
        # Find matching model prefix for output token limit
        matching_model = next(
            (model for model in self.MODEL_MAX_TOKENS.keys() 
             if model_name.startswith(model)), 
            None
        )
        
        # Get max output tokens for the model, default to 4096 if no match found
        self.max_tokens = self.MODEL_MAX_TOKENS.get(matching_model, 4096)
    
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        # Convert messages to Anthropic format
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Anthropic uses system prompt as part of the first user message
                system_prompt = msg["content"]
            else:
                formatted_messages.append({
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"]
                })
        
        # Add system prompt to first user message if exists
        if formatted_messages and "system_prompt" in locals():
            formatted_messages[0]["content"] = f"{system_prompt}\n\n{formatted_messages[0]['content']}"
        
        # Create API call parameters with required arguments
        # Use model's max tokens if none specified
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": min(max_tokens, self.max_tokens) if max_tokens else self.max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        response = await self.client.messages.create(**params)
        
        return LLMResponse(
            content=response.content[0].text,
            raw_response=response,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
    
    def count_tokens(self, text: str) -> int:
        # Keep this method for compatibility, can obtain token counts from the API
        return len(text.split()) * 1.3
