from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from core.models.base import BaseModel
from core.tools.base import BaseTool, ToolCall

class Agent(ABC):
    """Base agent class for problem-solving tasks."""
    
    def __init__(self,
                 model: BaseModel,
                 tools: Optional[List[BaseTool]] = None,
                 system_prompt: str = ""):
        self.model = model
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.system_prompt = system_prompt
        self.tool_calls: List[ToolCall] = []

    @abstractmethod
    async def solve(self, problem: str) -> Dict[str, Any]:
        """
        Solve a given problem and return the solution with metadata.
        
        Args:
            problem: The problem statement to solve
            
        Returns:
            Dict containing:
                - solution: str
                - metadata: Dict[str, Any] (implementation-specific data)
                - tool_calls: List[ToolCall]
        """
        pass

    async def _call_model(self, 
                         messages: List[Dict[str, str]], 
                         temperature: float = 1.0) -> str:
        """Helper method to call the underlying model."""
        response = await self.model.generate(messages, temperature=temperature)
        
        # Handle both string and LLMResponse returns
        if isinstance(response, str):
            return response
        return response.content

    async def _use_tool(self, 
                       tool_name: str, 
                       **kwargs) -> Any:
        """Helper method to use a tool and track its usage."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
            
        tool = self.tools[tool_name]
        result = await tool.run(**kwargs)
        self.tool_calls.append(result)
        return result

    def _format_messages(self, 
                        problem: str, 
                        additional_context: Optional[str] = None) -> List[Dict[str, str]]:
        """Helper method to format messages for the model."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem}
        ]
        
        if additional_context:
            messages.append({"role": "assistant", "content": additional_context})
            
        return messages