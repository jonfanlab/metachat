from typing import Dict, Any, List, Optional
from .base import Agent
from tools.solvers.scientific_compute import ScientificCompute
from tools.solvers.symbolic_solver import SymbolicSolver
from tools.design.api import NeuralDesignAPI
from tools.material_db.query_materials import MaterialDatabaseCLI
from pathlib import Path
import json
from datetime import datetime
import uuid

class IterativeAgentToolsMaterials(Agent):
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
            log_dir="experiments/logs/eval_v1_corrected_singlestepfunc/cot_iterative_tools_materials/materials_chat"
        )
        
        # Add logging configuration
        self.log_dir = Path(kwargs.get('log_dir', "experiments/logs/eval_v1_corrected_singlestepfunc/cot_iterative_tools_materials/self_chat")) / self.model.model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug = kwargs.get('debug', False)
        
        if not self.system_prompt:
            self.system_prompt = """You are an expert in optics and photonics engaging in a continuous conversation to help users with their optics and photonics questions. 
You have iterative access to numpy, sympy, neural network-based design APIs, and a materials database expert. 
You can talk to yourself and have an internal monologue. Plan out tool use to use information gathered from the tools at subsequent iterations.

Guidelines:
0. Think step by step. Break down complex problems into steps and plan your approach before solving.
1. If calculations are needed, write Python numpy code between <tool>scientific_compute</tool> tags
2. If symbolic manipulation is needed to e.g. solve for a variable or rearrange an equation because you are unsure, write Python sympy code between <tool>symbolic_solve</tool> tags
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
5. Return the final answer wrapped in <response> tags. Make sure your code prints the final answer in the correct units
6. If no calculations are needed, simply state the answer directly
7. You can only use ONE type of tag per message
8. Make sure to convert intermediate results to the correct units before using them to prevent multiplication or function unit mismatch errors
9. After using a tool, analyze its output before proceeding. If there is an error, think carefully why it occured and fix the code to try again.

IMPORTANT: Any text not wrapped in tags will be treated as your internal thoughts and planning. Only text within <response> tags will be shown to the user.

Examples:

1. Use available tools:
   - Scientific computing: <tool>scientific_compute</tool>
   - Symbolic solving: <tool>symbolic_solve</tool>
   - Neural design: <tool>neural_design</tool>
   - Materials expert chat: <tool>materials_chat</tool>

2. Respond to the user (Without wrapping in <response> tags the user will not be able to see your response!):
   <response>
   Your final answer or response to the user
   </response>

3. For neural network design (return the text you receive so the user can run the API call):
<tool>neural_design
design_metalens(refractive_index=2.7, lens_diameter=100e-6, focal_length=200e-6, thickness=500e-9, operating_wavelength=800e-9)
</tool>

Example workflow:
1. Think about approach:
    This problem requires calculating X, then checking material properties...

2. Use tools as needed:
   <tool>scientific_compute
   import numpy as np
   wavelength = 500e-9
   freq = constants.c / wavelength
   print(f'Frequency: {freq:.2e} Hz')
   </tool>

For symbolic manipulation:
<tool>symbolic_solve
import sympy as sp

# Rearrange thin lens equation 1/f = 1/u + 1/v to solve for image distance v
f, u, v = sp.symbols('f u v')
eq = sp.Eq(1/f, 1/u + 1/v)
solution = sp.solve(eq, v)[0]
print(f'v = {solution}')  # Should output: v = (f*u)/(u - f)
</tool>

3. Fix errors:
   -The solution was returned in a different format than expected. I have to first access the dictonary in the list.

4. Chat with materials expert:
   <tool>materials_chat
   What materials would work well for X application?
   </tool>

5. Provide final answer:
<response>
Based on the calculations and material properties, the solution is...
</response>"""

    def _format_messages(self, problem: str) -> List[Dict[str, str]]:
        """Format messages for the model including tool instructions."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem}
        ]
    
    async def _log_conversation(self, problem_id: str, messages: List[Dict[str, str]]):
        """Log conversation to a JSON file."""
        log_file = self.log_dir / f"{problem_id}.json"
        
        # Load existing log if it exists
        existing_log = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
            except json.JSONDecodeError:
                existing_log = []
                
        # Convert any SymPy objects to strings before serializing
        def convert_sympy_objects(obj):
            if isinstance(obj, dict):
                return {k: convert_sympy_objects(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sympy_objects(item) for item in obj]
            elif str(type(obj).__module__).startswith('sympy'):
                return str(obj)
            return obj
        
        # Convert SymPy objects in messages
        messages = convert_sympy_objects(messages)
        
        # Append new messages
        existing_log.extend(messages)
        
        # Write updated log
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_log, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to log file: {e}")

    async def solve(self, problem: str, problem_id: Optional[str] = None, temperature: float = 1.0, disable_cache: bool = False) -> Dict[str, Any]:
        # Generate problem_id if not provided
        if not problem_id:
            problem_id = str(uuid.uuid4())
            
        # Log initial problem
        await self._log_conversation(problem_id, [{
            "role": "user",
            "content": problem,
            "timestamp": datetime.now().isoformat()
        }])

        if disable_cache:
            timestamp = datetime.now().isoformat()
            problem = f"Date submitted: {timestamp}\n\n{problem}"

        messages = self._format_messages(problem)
        iteration_count = 0
        max_iterations = 20
        conversation = []

        while True:  # Let it run until natural completion or max iterations
            iteration_count += 1
            
            # Check if max iterations exceeded
            if iteration_count > max_iterations:
                # Log error if max iterations reached
                await self._log_conversation(problem_id, [{
                    "role": "error",
                    "content": "Maximum iterations reached",
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])

                return {
                    "status": "error",
                    "error": "Maximum iterations reached",
                    "conversation": conversation,
                    "problem_id": problem_id
                }
            
            if self.debug:
                print(f"\n=== Iteration {iteration_count} ===")

            current_response = await self._call_model(messages, temperature=temperature)

            # Log the model's response
            await self._log_conversation(problem_id, [{
                "role": "assistant",
                "content": current_response,
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration_count
            }])

            # Handle final response to user
            if "<response>" in current_response:
                response = current_response.split("<response>")[1].split("</response>")[0].strip()
                
                # Log final response
                await self._log_conversation(problem_id, [{
                    "role": "response",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                return {
                    "solution": response,
                    "metadata": {
                        "method": "iterative",
                        "num_iterations": iteration_count,
                        "conversation": conversation,
                        "problem_id": problem_id
                    },
                    "tool_calls": self.tool_calls
                }

            # Process tool calls and thoughts
            processed_solution = current_response
            
            # If no tags present, treat as thinking/planning
            if not any(tag in processed_solution for tag in ["<tool>", "<response>"]):
                # Log thinking
                await self._log_conversation(problem_id, [{
                    "role": "thinking",
                    "content": processed_solution,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                messages.append({"role": "assistant", "content": processed_solution})
                messages.append({"role": "user", "content": "Continue with your approach. If you need to reply to the user with an answer or need clarification, respond with <response> tags."})

                continue

            # Handle materials chat with logging
            if "<tool>materials_chat" in processed_solution:
                start = processed_solution.find("<tool>materials_chat")
                end = processed_solution.find("</tool>", start)
                if end != -1:
                    message = processed_solution[start + len("<tool>materials_chat"):end].strip()
                    
                    # Log materials chat query
                    await self._log_conversation(problem_id, [{
                        "role": "tool_call",
                        "tool": "materials_chat",
                        "query": message,
                        "timestamp": datetime.now().isoformat(),
                        "iteration": iteration_count
                    }])
                    
                    results = await self.materials_cli.query_with_id(message, conversation_id=problem_id)
                    
                    # Log materials chat response
                    await self._log_conversation(problem_id, [{
                        "role": "tool_response",
                        "tool": "materials_chat",
                        "content": results['message'],
                        "timestamp": datetime.now().isoformat(),
                        "iteration": iteration_count
                    }])
                    
                    messages.append({"role": "assistant", "content": current_response})
                    messages.append({"role": "user", "content": f"Materials expert response: {results['message']}"})

            # Similar logging for other tools
            for tool_name in ['scientific_compute', 'symbolic_solve', 'neural_design']:
                if f"<tool>{tool_name}" in processed_solution:
                    start = processed_solution.find(f"<tool>{tool_name}")
                    end = processed_solution.find("</tool>", start)
                    if end != -1:
                        code_block = processed_solution[start + len(f"<tool>{tool_name}"):end].strip()
                        
                        # Log tool call
                        await self._log_conversation(problem_id, [{
                            "role": "tool_call",
                            "tool": tool_name,
                            "code": code_block,
                            "timestamp": datetime.now().isoformat(),
                            "iteration": iteration_count
                        }])
                        
                        result = await self.tools[tool_name].execute(code_block)
                        
                        # Log tool response
                        await self._log_conversation(problem_id, [{
                            "role": "tool_response",
                            "tool": tool_name,
                            "result": result,
                            "timestamp": datetime.now().isoformat(),
                            "iteration": iteration_count
                        }])
                        
                        messages.append({"role": "assistant", "content": current_response})
                        messages.append({"role": "user", "content": f"Tool output: {result}"})

            # Update conversation history
            conversation.append({
                "iteration": iteration_count,
                "input": current_response,
                "output": processed_solution,
                "timestamp": datetime.now().isoformat()
            })