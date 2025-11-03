from typing import Dict, Any, List, Optional
from .base import Agent
from tools.solvers.scientific_compute import ScientificCompute
from tools.solvers.symbolic_solver import SymbolicSolver
from tools.design.api import NeuralDesignAPI
from tools.material_db.query_materials import MaterialDatabaseCLI
from datetime import datetime

class StandardAgentToolsMaterials(Agent):
    """Simple one-shot agent that solves problems in a single model call."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add scientific computing tool
        self.tools['scientific_compute'] = ScientificCompute()
        # Add symbolic solving tool
        self.tools['symbolic_solve'] = SymbolicSolver()
        # Add neural design tools
        self.tools['neural_design'] = NeuralDesignAPI()
        # Initialize the materials CLI with the same model
        self.materials_cli = MaterialDatabaseCLI(
            db_path="tools/material_db/materials.db",
            model=self.model,  # Pass the model instance directly
            debug=False,
            log_dir="experiments/logs/eval_v1_corrected/cot_tools_materials_multi/materials_chat"
        )
        
        if not self.system_prompt:
            self.system_prompt = """You are an expert in optics and photonics with access to scientific computing, symbolic mathematics, neural network-based metalens and superpixel design APIs, and a materials database expert you can chat with. 

When solving problems:
0. Think step by step. Break down complex problems into steps and plan your approach before solving.
1. If calculations are needed, write code between <tool>scientific_compute</tool> tags
2. If symbolic manipulation is needed, write code between <tool>symbolic_solve</tool> tags
3. If you need to design a metalens or superpixel, use these neural network tools:
   - For metalenses, use: <tool>neural_design
   design_metalens(refractive_index, lens_diameter [m], focal_length [m], thickness [m], operating_wavelength [m])
   </tool>
   - For superpixels, use: <tool>neural_design
   design_superpixel(refractive_index, length [m], incident_angle [deg], diffraction_angle [deg], thickness [m], operating_wavelength [m])
   </tool>
4. If you need information about materials or their properties, you can chat with the materials expert:
   <tool>materials_chat
   Your question or message to the materials expert
   </tool>
5. Use numpy (as np) and scipy modules for numerical calculations
6. Use sympy (as sp) for symbolic mathematics
7. Make sure your code prints the final answer in the correct units
8. If no calculations are needed, simply state the answer directly

Example responses:

For materials chat:
<tool>materials_chat
What materials would you recommend for a high-power laser mirror operating at 1064 nm?
</tool>

<tool>materials_chat
What is the refractive index of TiO2 film at 800 nm wavelength?
</tool>

For neural network design:
<tool>neural_design
design_metalens(refractive_index=2.7, lens_diameter=100e-6, focal_length=200e-6, thickness=500e-9, operating_wavelength=800e-9)
</tool>

For a numerical calculation:
<tool>scientific_compute
import numpy as np
from scipy import constants

wavelength = 500e-9  # 500 nm
freq = constants.c / wavelength
print(f'Answer: {freq:.2e} Hz')
</tool>

For symbolic manipulation:
<tool>symbolic_solve
import sympy as sp

# Solve n1 * sin(theta1) = n2 * sin(theta2) for theta2
n1, n2, theta1, theta2 = sp.symbols('n1 n2 theta1 theta2')
eq = sp.Eq(n1 * sp.sin(theta1), n2 * sp.sin(theta2))
solution = sp.solve(eq, theta2)[0]
print(f'theta2 = {solution}')
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
        processed_solution = solution.content
        
        # Handle materials chat with conversation tracking
        while "<tool>materials_chat" in processed_solution:
            start = processed_solution.find("<tool>materials_chat")
            end = processed_solution.find("</tool>", start)
            if end == -1:
                break
                
            # Get the message to send to the materials expert
            message = processed_solution[start + len("<tool>materials_chat"):end].strip()
            
            # Send message to materials CLI and get response, passing problem_id
            results = await self.materials_cli.query_with_id(
                message,
                conversation_id=problem_id,
            )
            
            # Update conversation history
            if results.get("status") == "response":
                response = results.get("message", "No response provided")
                self.materials_cli.conversation_history.append({
                    "query_explanation": message,
                    "results": response
                })
                self.materials_cli.current_messages = results.get("messages")
                
                replacement = f"Materials Expert Chat:\nMe: {message}\nExpert: {response}"
            else:
                replacement = f"Error chatting with materials expert: {results.get('error', 'Unknown error')}"
            
            processed_solution = (
                processed_solution[:start] +
                replacement +
                processed_solution[end + len("</tool>"):]
            )
        
        while "<tool>scientific_compute" in processed_solution:
            # Extract code between tool tags
            start = processed_solution.find("<tool>scientific_compute")
            end = processed_solution.find("</tool>", start)
            if end == -1:
                break
                
            # Get the code block
            code_block = processed_solution[start + len("<tool>scientific_compute"):end].strip()
            
            # Execute the code
            result = await self.tools['scientific_compute'].execute(code_block)
            
            # Replace the tool call with the result
            if result.get('success', False):
                output = result.get('output', '').strip()
                if not output:  # If no printed output, try to use the result
                    output = f"Result: {result.get('result')}"
                replacement = f"```python\n{code_block}\n```\nOutput:\n```\n{output}\n```"
            else:
                replacement = f"Error executing code: {result.get('error', 'Unknown error')}"
            
            processed_solution = (
                processed_solution[:start] +
                replacement +
                processed_solution[end + len("</tool>"):]
            )
        
        while "<tool>symbolic_solve" in processed_solution:
            # Extract code between tool tags
            start = processed_solution.find("<tool>symbolic_solve")
            end = processed_solution.find("</tool>", start)
            if end == -1:
                break
                
            # Get the code block
            code_block = processed_solution[start + len("<tool>symbolic_solve"):end].strip()
            
            # Execute the code
            result = await self.tools['symbolic_solve'].execute(code_block)
            
            # Replace the tool call with the result
            if result.get('success', False):
                output = result.get('output', '').strip()
                if not output:  # If no printed output, try to use the result
                    output = f"Result: {result.get('result')}"
                replacement = f"```python\n{code_block}\n```\nOutput:\n```\n{output}\n```"
            else:
                replacement = f"Error executing code: {result.get('error', 'Unknown error')}"
            
            processed_solution = (
                processed_solution[:start] +
                replacement +
                processed_solution[end + len("</tool>"):]
            )
        
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