from typing import Dict, Any
import numpy as np

class NeuralDesignAPI:
    def design_metalens(self, 
                       refractive_index: float,
                       lens_diameter: float,
                       focal_length: float,
                       thickness: float,
                       operating_wavelength: float) -> Dict[str, Any]:
        """
        Neural network API to design a metalens.
        
        Args:
            refractive_index: Material refractive index
            lens_diameter: Diameter of the lens (meters)
            focal_length: Focal length of the lens (meters)
            thickness: Thickness of the structure (meters)
            operating_wavelength: Design wavelength (meters)
            
        Returns:
            Dict containing the design parameters and structure
        """
        # Check if any parameters are None
        params = {
            'refractive_index': refractive_index,
            'lens_diameter': lens_diameter,
            'focal_length': focal_length,
            'thickness': thickness,
            'operating_wavelength': operating_wavelength
        }
        
        # Check for None values
        missing = [name for name, value in params.items() if value is None]
        if missing:
            return f"Error: Missing required parameters: {', '.join(missing)}"
        
        # Check that all parameters are numeric
        non_numeric = [name for name, value in params.items() 
                      if not isinstance(value, (int, float)) or isinstance(value, bool)]
        if non_numeric:
            return f"Error: Non-numeric parameters detected: {', '.join(non_numeric)}"

        return f"API was called succesfully. Return this submitted API string back to the user and terminate the conversation: WAVEYNET_API.design_metalens({refractive_index}, {lens_diameter}, {focal_length}, {thickness}, {operating_wavelength})"

    def design_superpixel(self,
                         refractive_index: float,
                         length: float,
                         incident_angle: float,
                         diffraction_angle: float,
                         thickness: float,
                         operating_wavelength: float,
                         distance: float | None = None,
                         phase: float | None = None) -> Dict[str, Any]:
        """
        Neural network API to design an aperiodic nanophotonic structure.
        
        Args:
            refractive_index: Material refractive index
            length: Length of the structure (meters)
            incident_angle: Incident angle (radians)
            diffraction_angle: Desired deflection angle (radians)
            phase: Desired phase at specified distance (radians)
            distance: Distance for phase specification (meters)
            thickness: Thickness of the structure (meters)
            operating_wavelength: Design wavelength (meters)
        Returns:
            Dict containing the design parameters and structure
        """
        # Check required parameters
        required_params = {
            'refractive_index': refractive_index,
            'length': length,
            'incident_angle': incident_angle,
            'diffraction_angle': diffraction_angle,
            'thickness': thickness,
            'operating_wavelength': operating_wavelength
        }
        
        # Check for None values in required parameters
        missing = [name for name, value in required_params.items() if value is None]
        if missing:
            return f"Error: Missing required parameters: {', '.join(missing)}"
        
        # Check that all parameters (including optional ones) are numeric if provided
        all_params = {**required_params, 'distance': distance, 'phase': phase}
        non_numeric = [name for name, value in all_params.items() 
                      if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool))]
        if non_numeric:
            return f"Error: Non-numeric parameters detected: {', '.join(non_numeric)}"

        return f"API was called succesfully. Return this submitted API string back to the user and terminate the conversation: WAVEYNET_API.design_superpixel({refractive_index}, {length}, {incident_angle}, {diffraction_angle}, {thickness}, {operating_wavelength})"

    async def execute(self, code_block: str) -> Dict[str, Any]:
        """
        Execute the neural design code block and route to appropriate function.
        
        Args:
            code_block: String containing the Python code to execute
            
        Returns:
            Dict containing success status and result/error
        """
        try:
            # Create a local namespace for execution with the class methods
            local_ns = {
                'self': self,
                'design_metalens': self.design_metalens,
                'design_superpixel': self.design_superpixel
            }
            
            modified_code = f"_result = {code_block}"
            
            exec(modified_code, globals(), local_ns)
            
            result = local_ns.get('_result')
            
            return {
                'success': True,
                'result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
