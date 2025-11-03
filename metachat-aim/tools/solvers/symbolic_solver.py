from typing import Dict, Any
import sympy as sp
from core.tools.base import BaseTool
from io import StringIO
import sys

class SymbolicSolver(BaseTool):
    """Tool for symbolic mathematics using SymPy."""
    
    def __init__(self):
        super().__init__(
            name="symbolic_solve",
            description="""Perform symbolic mathematics using SymPy.
            Available module: sympy (as sp)
            
            Input should be valid Python code as a string.
            Output will be the result of the symbolic manipulation.
            
            Example:
            Input: "x, y = sp.symbols('x y'); expr = x**2 + y; solved = sp.solve(expr - 10, x); print(solved)"
            """
        )
    
    async def execute(self, code: str) -> Dict[str, Any]:
        """Execute SymPy code and return results."""
        try:
            # Create a string buffer to capture printed output
            output_buffer = StringIO()
            old_stdout = sys.stdout
            sys.stdout = output_buffer
            
            # Create a dictionary for local variables
            locals_dict = {
                'sp': sp
            }
            
            try:
                # Execute the code
                exec(code, globals(), locals_dict)
                # Get the captured output
                output = output_buffer.getvalue()
            finally:
                # Restore stdout
                sys.stdout = old_stdout
            
            # Return the last assigned variable if it exists
            result = None
            for var_name, value in locals_dict.items():
                if not var_name.startswith('_') and var_name not in ['sp']:
                    result = value
            
            return {
                'success': True,
                'result': result,
                'output': output
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
