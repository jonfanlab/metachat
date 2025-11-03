from typing import Dict, Any
import numpy as np
from scipy import constants, optimize, integrate
from core.tools.base import BaseTool, ToolCall
from io import StringIO
import sys

class ScientificCompute(BaseTool):
    """Tool for executing scientific computing code using numpy and scipy."""
    
    def __init__(self):
        super().__init__(
            name="scientific_compute",
            description="""Execute scientific calculations using numpy and scipy. 
            Available modules: numpy (as np), scipy.constants, scipy.optimize, scipy.integrate
            
            Input should be valid Python code as a string.
            Output will be the result of the calculation.
            
            Example:
            Input: "import numpy as np; wavelength = 500e-9; freq = constants.c / wavelength; print(f'{freq:.2e} Hz')"
            """
        )
    
    async def execute(self, code: str) -> Dict[str, Any]:
        """Execute scientific Python code and return results."""
        try:
            # Create a string buffer to capture printed output
            output_buffer = StringIO()
            old_stdout = sys.stdout
            sys.stdout = output_buffer
            
            # Create a dictionary for local variables
            locals_dict = {
                'np': np,
                'constants': constants,
                'optimize': optimize,
                'integrate': integrate
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
                if not var_name.startswith('_') and var_name not in ['np', 'constants', 'optimize', 'integrate']:
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
