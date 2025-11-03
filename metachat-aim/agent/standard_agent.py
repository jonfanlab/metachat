from typing import Dict, Any, List, Optional
from .base import Agent
from tools.design.api import NeuralDesignAPI
from datetime import datetime

class StandardAgent(Agent):
    """Simple one-shot agent that solves problems in a single model call."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add neural design tools
        self.tools['neural_design'] = NeuralDesignAPI()
        
        if not self.system_prompt:
            self.system_prompt = """You are an expert in optics and photonics with access to neural network-based metalens and superpixel design APIs. 

Think step by step. Break down complex problems into steps and plan your approach before solving.

If you need to design a metalens or superpixel, use these neural network tools:
   - For metalenses, use: <tool>neural_design
   design_metalens(refractive_index, lens_diameter [m], focal_length [m], thickness [m], operating_wavelength [m])
   </tool>
   - For superpixels, use: <tool>neural_design
   design_superpixel(refractive_index, length [m], incident_angle [deg], diffraction_angle [deg], thickness [m], operating_wavelength [m])
   </tool>

Example responses:

For neural network design:
<tool>neural_design
design_metalens(refractive_index=2.7, lens_diameter=100e-6, focal_length=200e-6, thickness=500e-9, operating_wavelength=800e-9)
</tool>

For a direct answer:
Answer: 1.55 μm"""

    def _format_messages(self, problem: str) -> List[Dict[str, str]]:
        """Format messages for the model including tool instructions."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem}
        ]
    
    async def solve(self, problem: str, problem_id: Optional[str] = None, temperature: float = 1.0, disable_cache: bool = False) -> Dict[str, Any]:
        if disable_cache:
            timestamp = datetime.now().isoformat()
            problem = f"Date submitted: {timestamp}\n\n{problem}"
            
        messages = self._format_messages(problem)
        
        solution = await self._call_model(
            messages,
            temperature=temperature
        )
        
        # Process tool calls in the solution
        # processed_solution = solution.content
        processed_solution = solution
        while "<tool>neural_design" in processed_solution:
            # Extract code between tool tags
            start = processed_solution.find("<tool>neural_design")
            end = processed_solution.find("</tool>", start)
            if end == -1:
                break
                
            # Get the code block
            code_block = processed_solution[start + len("<tool>neural_design"):end].strip()
            
            # Execute the code
            result = await self.tools['neural_design'].execute(code_block)
            
            # Replace the tool call with the result
            if result.get('success', False):
                # Handle the string return value from the API
                api_call = result.get('result', '')
                output = f"API Call that would be made:\n{api_call}"
                replacement = f"```python\n{code_block}\n```\nOutput:\n```\n{output}\n```"
            else:
                replacement = f"Error executing code: {result.get('error', 'Unknown error')}"
            
            processed_solution = (
                processed_solution[:start] +
                replacement +
                processed_solution[end + len("</tool>"):]
            )
        
        return {
            "solution": processed_solution,
            "metadata": {
                "method": "one-shot",
                "num_model_calls": 1
            },
            "tool_calls": self.tool_calls
        }