import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
from datetime import datetime
import random
from .models import MaterialData, MaterialCategory, MaterialType
import uuid
import json
from pathlib import Path

class MaterialQueryInterface:
    def __init__(self, 
                 db_path: str = "material_database/materials.db",
                 model: Optional["BaseModel"] = None,
                 debug: bool = False,
                 log_dir: str = "experiments/logs/materials_chat"):
        self.db_path = db_path
        self.model = model
        self.debug = debug
        self.log_dir = Path(log_dir)# / (model.model_name if model else "default")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.schema = """
Database Schema:
1. materials
   - id (TEXT PRIMARY KEY)
   - name (TEXT)
   - category (TEXT: 'main', 'glass', 'organic', 'other')
   - type (TEXT: 'bulk', 'film', 'crystal', 'glass', 'liquid')
   - min_wavelength (REAL)
   - max_wavelength (REAL)
   - wavelength_unit (TEXT)
   - data_type (TEXT: 'tabulated nk', 'formula 1', 'formula 2', ..., 'formula 9')
   - file_path (TEXT)
   - created_at (TIMESTAMP)

2. material_references
   - id (TEXT PRIMARY KEY)
   - material_id (TEXT -> materials.id)
   - year (INTEGER)
   - title (TEXT)
   - journal (TEXT)
   - doi (TEXT)
   - citation_count (INTEGER)
   - last_citation_update (TIMESTAMP)

3. authors
   - id (TEXT PRIMARY KEY)
   - name (TEXT UNIQUE)

4. reference_authors
   - reference_id (TEXT -> material_references.id)
   - author_id (TEXT -> authors.id)
   - author_order (INTEGER)

5. specs
   - id (TEXT PRIMARY KEY)
   - material_id (TEXT -> materials.id)
   - thickness (REAL)
   - substrate (TEXT)
   - temperature (REAL)
   - additional_info (TEXT)

6. measurements
   - id (TEXT PRIMARY KEY)
   - material_id (TEXT -> materials.id)
   - wavelength (REAL)
   - n (REAL)
   - k (REAL)

7. material_formulas
   - id (TEXT PRIMARY KEY)
   - material_id (TEXT -> materials.id)
   - formula_type (TEXT NOT NULL)
   - wavelength_range_min (REAL)
   - wavelength_range_max (REAL)
   - coefficients (TEXT)  -- JSON array of coefficients
"""
        
        self.system_prompt = f"""You are an expert materials scientist with deep knowledge of optics and photonics, 
engaging in a continuous conversation to help users with their materials-related questions. You can query a 
materials database through an iterative process to find suitable materials and their properties for specific applications.

{self.schema}

You can perform multiple SQL queries to build up understanding and refine recommendations. You can talk to yourself to have an internal monologue.
Each query should focus on a specific aspect of the investigation.

IMPORTANT NOTE: The wavelengths are given in micrometers. Convert units before querying.

IMPORTANT NOTE 2: The material name is the chemical formula. So "Crystaline Silicon" is name: Si, type: crystal

Conversation Guidelines:

1. Maintain a natural conversation while using database queries to support your responses.

2. Response Format:
   - While gathering information: Use <query>, <interpolate>, or <calculate_n> to collect data
   - When you have a FINAL ANSWER: Always wrap it in <response> tags
   - You MUST eventually provide a response - don't get stuck in an infinite loop of queries
   - Example flow:
     1. Make queries to gather data
     2. Once you have enough information, provide final answer:
        <response>
        Based on the data gathered, I recommend [material] because [reasoning]...
        </response>

3. When you want to query the database:
   REASONING: Explain why you're making this query
   <query>
   Your SQL query here (if searching for a material, you can pre-sort by selection criteria: citation count, year, etc.)
   </query>
   ANALYSIS: Brief analysis of the results and next steps:
   - The material might not be stored as the exact string you search (e.g. BK7 has several variants, including "N-BK7" and "P-BK7" but there's no "BK7" in the database)
   - If you need more info: State what additional query/calculation is needed. If there are too many results, refine the query until you have fewer than twenty (20) results. Use citation counts to find the best answer. Do not limit the results to twenty, just refine the query.
   - If you have enough information: Provide final answer using <response> tags

IMPORTANT NOTE 3: If you use <query>, you cannot use <response> in the same message (you can only either query database or respond to user at once)

4. When you need to perform linear interpolation between two points:
   <interpolate>
   {{"x": target_value, "x1": first_x, "y1": first_y, "x2": second_x, "y2": second_y}}
   </interpolate>
   The system will return the interpolated value for further use.

5. When you need to calculate a refractive index using a dispersion formula:
   <calculate_n>
   {{"formula_type": X, "wavelength": wavelength_in_microns, "coefficients": [c1, c2, ...]}}
   </calculate_n>
   The system will return the calculated refractive index value.

IMPORTANT NOTE 3: If you use <query>, <interpolate>, or <calculate_n>, you cannot use <response> in the same message.

6. Consider throughout the conversation:
   - IF YOUR QUERY RETURNS MULTIPLE RESULTS, use citation counts and publication years to choose the data source. More/more comprehensive data or different formats do NOT necessarily make a source more reliable.
   - IMPORTANT: If there are multiple results, use citation counts and publication years to find the best answer BEFORE interpolating
   - Interpolate between measurements if necessary for finding properties at specific wavelengths between measured points
   - Application constraints

Remember to:
- Use queries only when needed to support the discussion
- Ask clarifying questions when necessary
- Provide clear and concise explanations
- Build upon previous conversation context
- Use interpolation when precise values are needed between measured points

You are engaging in an ongoing conversation, so don't try to solve everything at once. Break down complex 
questions into manageable parts and use queries strategically to support the discussion.
"""

    def _summarize_results(self, results: List[Dict[str, Any]], max_items: int = 20) -> Dict[str, Any]:
        """Summarize query results to reduce token usage."""
        if isinstance(results, list):
            total_items = len(results)
            if total_items > max_items:
                return {
                    "total_items": total_items,
                    "showing_first": max_items,
                    "sample": random.sample(results, min(max_items, len(results)))
                }
        return results

    async def _execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute a single SQL query and return results."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        try:
            c.execute(sql_query)
            results = [dict(row) for row in c.fetchall()]
            # Summarize results before returning
            return self._summarize_results(results)
        finally:
            conn.close()
            
        return results

    async def _interpolate(self, x: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """Linear interpolation between two points."""
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    async def _calculate_refractive_index(self, formula_type: int, wavelength: float, coefficients: List[float]) -> float:
        """Calculate refractive index using specified formula and coefficients."""
        try:
            # Pad coefficients with zeros based on formula type
            max_coeffs = {
                1: 17,  # Sellmeier: C1 + 8 pairs (C2/C3 ... C16/C17)
                2: 17,  # Sellmeier-2: Same structure as Sellmeier
                3: 17,  # Polynomial: C1 + 8 pairs
                4: 17,  # RefractiveIndex.INFO: C1 + various terms
                5: 11,  # Cauchy: C1 + 5 pairs
                6: 11,  # Gases: C1 + 5 pairs
                7: 6,   # Herzberger: C1 through C6
                8: 4,   # Retro: C1 through C4
                9: 6    # Exotic: C1 through C6
            }
            
            # Pad with zeros if needed
            if formula_type not in max_coeffs:
                raise ValueError(f"Unknown formula type: {formula_type}")
                
            padded_coeffs = coefficients + [0.0] * (max_coeffs[formula_type] - len(coefficients))

            if formula_type == 1:  # Sellmeier
                n_squared = 1 + padded_coeffs[0]  # C1
                for i in range(8):  # 8 terms
                    c_num = padded_coeffs[2*i + 1]  # C2, C4, C6, C8, C10, C12, C14, C16
                    c_den = padded_coeffs[2*i + 2]  # C3, C5, C7, C9, C11, C13, C15, C17
                    n_squared += (c_num * wavelength**2) / (wavelength**2 - c_den**2)
                return (n_squared)**0.5

            elif formula_type == 2:  # Sellmeier-2
                n_squared = 1 + padded_coeffs[0]  # C1
                for i in range(8):  # 8 terms
                    c_num = padded_coeffs[2*i + 1]  # C2, C4, C6, C8, C10, C12, C14, C16
                    c_den = padded_coeffs[2*i + 2]  # C3, C5, C7, C9, C11, C13, C15, C17
                    n_squared += (c_num * wavelength**2) / (wavelength**2 - c_den)
                return (n_squared)**0.5

            elif formula_type == 3:  # Polynomial
                n_squared = padded_coeffs[0]  # C1
                for i in range(8):  # 8 terms
                    c = padded_coeffs[2*i + 1]  # C2, C4, C6, C8, C10, C12, C14, C16
                    power = padded_coeffs[2*i + 2]  # C3, C5, C7, C9, C11, C13, C15, C17
                    n_squared += c * wavelength**power
                return (n_squared)**0.5

            elif formula_type == 4:  # RefractiveIndex.INFO
                n_squared = padded_coeffs[0]  # C1
                # First fraction term
                n_squared += (padded_coeffs[1] * wavelength**padded_coeffs[2]) / (wavelength**2 - padded_coeffs[3]**padded_coeffs[4])
                # Second fraction term
                n_squared += (padded_coeffs[5] * wavelength**padded_coeffs[6]) / (wavelength**2 - padded_coeffs[7]**padded_coeffs[8])
                # Remaining polynomial terms
                for i in range(9, len(padded_coeffs), 2):
                    c = padded_coeffs[i]  # C10, C12, C14, C16
                    power = padded_coeffs[i + 1]  # C11, C13, C15, C17
                    n_squared += c * wavelength**power
                return (n_squared)**0.5

            elif formula_type == 5:  # Cauchy
                n = padded_coeffs[0]  # C1
                for i in range(5):  # 5 terms
                    c = padded_coeffs[2*i + 1]  # C2, C4, C6, C8, C10
                    power = padded_coeffs[2*i + 2]  # C3, C5, C7, C9, C11
                    n += c * wavelength**power
                return n

            elif formula_type == 6:  # Gases
                n = 1 + padded_coeffs[0]  # C1
                for i in range(5):  # 5 terms
                    c_num = padded_coeffs[2*i + 1]  # C2, C4, C6, C8, C10
                    c_den = padded_coeffs[2*i + 2]  # C3, C5, C7, C9, C11
                    n += c_num / (c_den - wavelength**-2)
                return n

            elif formula_type == 7:  # Herzberger
                lambda_squared = wavelength**2
                L = 1 / (lambda_squared - 0.028)
                n = (padded_coeffs[0] +                    # C1
                     padded_coeffs[1] * L +                # C2/(λ² - 0.028)
                     padded_coeffs[2] * L**2 +             # C3(1/(λ² - 0.028))²
                     padded_coeffs[3] * lambda_squared +   # C4λ²
                     padded_coeffs[4] * lambda_squared**2 + # C5λ⁴
                     padded_coeffs[5] * lambda_squared**3)  # C6λ⁶
                return n

            elif formula_type == 8:  # Retro
                term = (padded_coeffs[1] * wavelength**2) / (wavelength**2 - padded_coeffs[2])
                term += padded_coeffs[3] * wavelength**2
                term += padded_coeffs[0]
                n_squared = (1 + 2*term) / (1 - term)
                return (n_squared)**0.5

            elif formula_type == 9:  # Exotic
                n_squared = (padded_coeffs[0] +                                      # C1
                           padded_coeffs[1] / (wavelength**2 - padded_coeffs[2]) +   # C2/(λ² - C3)
                           padded_coeffs[3] * (wavelength - padded_coeffs[4]) /      # C4(λ - C5)/
                           ((wavelength - padded_coeffs[4])**2 + padded_coeffs[5]))  # ((λ - C5)² + C6)
                return (n_squared)**0.5

        except Exception as e:
            raise ValueError(f"Error calculating refractive index with formula {formula_type}: {str(e)}")

    async def _iterative_query(self, 
                              question: str, 
                              conversation_history: List[Dict[str, Any]] = None,
                              max_history_tokens: int = 100000,
                              max_iterations: int = 20,
                              conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform iterative querying with conversation history."""
        if conversation_history is None:
            conversation_history = []
            
        # Start with just the system prompt
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Log each message in the conversation
        async def log_message(role: str, content: str):
            if conversation_id:
                await self._log_conversation(conversation_id, [{
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "problem_id": conversation_id  # Include the problem ID in each message
                }])
        
        # Log the initial question
        await log_message("user", question)
        
        if self.debug:
            print("\n=== Starting New Query ===")
            print(f"Initial question: {question}")
            print(f"History length: {len(conversation_history)}")
            
        # Function to estimate tokens (rough approximation)
        def estimate_tokens(text: str) -> int:
            # GPT models typically use ~4 chars per token
            return len(text) // 4
        
        # Add conversation history with token management
        current_tokens = estimate_tokens(self.system_prompt + question)
        filtered_history = []
        
        # Process history from most recent to oldest
        for entry in reversed(conversation_history):
            query_tokens = estimate_tokens(entry["query_explanation"])
            results_tokens = estimate_tokens(str(entry["results"]))
            
            # Check if adding this entry would exceed token limit
            if current_tokens + query_tokens + results_tokens > max_history_tokens:
                break
                
            current_tokens += query_tokens + results_tokens
            filtered_history.insert(0, entry)
        
        # Add filtered history to messages chronologically
        for entry in filtered_history:
            messages.extend([
                {"role": "user", "content": entry["query_explanation"]},
                {"role": "assistant", "content": entry["results"]}
            ])
        
        # Add the current question last
        messages.append({"role": "user", "content": question})
        
        
        full_conversation = []
        all_queries = []
        iteration_count = 0
        
        while True:
            # Check iteration limit
            iteration_count += 1
            if iteration_count > max_iterations:
                if self.debug:
                    print(f"\n=== Iteration limit reached ({max_iterations}) ===")
                return {
                    "status": "error",
                    "error": "Maximum number of self-thought iterations reached",
                    "conversation": full_conversation,
                    "queries": all_queries
                }
            
            if self.debug:
                print(f"\n=== Iteration {iteration_count} ===")
            
            # Get next query from LLM
            response = await self.model.generate(messages, temperature=0.3)
            current_response = response.content
            
            # Log the LLM's response
            await log_message("assistant", current_response)
            
            if self.debug:
                print("\nLLM Response:")
                print(current_response)
            
            # Handle interpolation request
            if "<interpolate>" in current_response:
                try:
                    # Extract interpolation parameters
                    interp_parts = current_response.split("<interpolate>")[1].split("</interpolate>")[0].strip()
                    params = json.loads(interp_parts)
                    
                    # Perform interpolation
                    result = await self._interpolate(
                        float(params["x"]),
                        float(params["x1"]),
                        float(params["y1"]),
                        float(params["x2"]),
                        float(params["y2"])
                    )
                    
                    # Log the interpolation
                    await log_message("interpolation", f"Parameters: {params}, Result: {result}")
                    
                    # Update messages with interpolation result
                    messages.extend([
                        {"role": "assistant", "content": current_response},
                        {"role": "user", "content": f"Interpolation result: {result}"}
                    ])
                    continue
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    if self.debug:
                        print(f"Interpolation Error: {str(e)}")
                    messages.extend([
                        {"role": "assistant", "content": current_response},
                        {"role": "user", "content": f"Interpolation error: {str(e)}"}
                    ])
                    continue

            # Handle formula calculation request
            if "<calculate_n>" in current_response:
                try:
                    # Extract calculation parameters
                    calc_parts = current_response.split("<calculate_n>")[1].split("</calculate_n>")[0].strip()
                    params = json.loads(calc_parts)
                    
                    # Perform calculation
                    result = await self._calculate_refractive_index(
                        int(params["formula_type"]),
                        float(params["wavelength"]),
                        params["coefficients"]
                    )
                    
                    # Log the calculation
                    await log_message("calculation", f"Parameters: {params}, Result: {result}")
                    
                    # Update messages with calculation result
                    messages.extend([
                        {"role": "assistant", "content": current_response},
                        {"role": "user", "content": f"Calculated n: {result}"}
                    ])
                    continue
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    if self.debug:
                        print(f"Calculation Error: {str(e)}")
                    messages.extend([
                        {"role": "assistant", "content": current_response},
                        {"role": "user", "content": f"Calculation error: {str(e)}"}
                    ])
                    continue

            # Handle user communication with unified response tag
            if "<response>" in current_response:
                response_text = current_response.split("<response>")[1].split("</response>")[0].strip()
                if self.debug:
                    print("\n=== Response to User ===")
                    print(f"Message: {response_text}")
                
                # Append to conversation history
                full_conversation.append({
                    "query_explanation": question,
                    "results": response_text
                })
                
                return {
                    "status": "response",
                    "message": response_text,
                    "conversation": full_conversation,
                    "queries": all_queries,
                    "messages": messages
                }
            
            # Extract and execute query
            if "<query>" in current_response:
                if self.debug:
                    print("\n=== Executing Query ===")
                
                # Set default reasoning if not found
                parts = current_response.split("REASONING: ")
                reasoning = parts[1].split("<query>")[0].strip() if len(parts) > 1 else "Query execution"
                
                if self.debug:
                    print(f"Reasoning: {reasoning}")
                
                query_parts = current_response.split("<query>")[1].split("</query>")
                sql_query = query_parts[0].strip()
                if self.debug:
                    print(f"SQL Query: {sql_query}")
                
                # Log reasoning
                await log_message("reasoning", reasoning)
                
                # Log query and results
                await log_message("query", sql_query)
                
                try:
                    results = await self._execute_query(sql_query)
                    await log_message("results", str(results))
                except sqlite3.Error as e:
                    error_msg = f"Query error: {str(e)}"
                    if self.debug:
                        print(error_msg)
                    
                    # Log the error
                    await log_message("error", f"Query error: {str(e)} | Query: {sql_query}")
                    
                    results = {"error": str(e)}
                    
                    # Add error to conversation history
                    messages.extend([
                        {"role": "assistant", "content": current_response},
                        {"role": "user", "content": error_msg}
                    ])
                    continue
                
                # Store query information
                query_info = {
                    "reasoning": reasoning,
                    "query": sql_query,
                    "results": results
                }
                all_queries.append(query_info)
                
                # Update conversation
                full_conversation.append({
                    "query_explanation": reasoning,
                    "results": results
                })
                messages.extend([
                    {"role": "assistant", "content": current_response},
                    {"role": "user", "content": f"Query results: {results}"}
                ])

    async def _log_conversation(self, conversation_id: str, messages: List[Dict[str, str]]):
        """Log conversation to a JSON file."""
        log_file = self.log_dir / f"{conversation_id}.json"
        
        # Load existing log if it exists
        existing_log = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
            except json.JSONDecodeError:
                # Handle case where file exists but is empty or invalid
                existing_log = []
                
        # Append new messages
        existing_log.extend(messages)
        
        # Write updated log
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_log, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to log file: {e}")

    async def query(self, 
                    question: str, 
                    conversation_id: Optional[str] = None,
                    additional_info: Optional[str] = None,
                    conversation_history: Optional[List[Dict[str, Any]]] = None,
                    max_history_tokens: int = 2000,
                    max_iterations: int = 20) -> Dict[str, Any]:
        """Process a natural language question with iterative querying."""
        if not self.model:
            raise ValueError("LLM model is required for natural language queries")
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        # Log initial question
        await self._log_conversation(conversation_id, [{
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat()
        }])
        
        if additional_info:
            question = f"{question}\nAdditional Information: {additional_info}"
            
        if self.debug:
            print("\n=== Starting Query Process ===")
            print(f"Question: {question}")
            if additional_info:
                print(f"Additional Info: {additional_info}")
            
        result = await self._iterative_query(
            question,
            conversation_history=conversation_history,
            max_history_tokens=max_history_tokens,
            max_iterations=max_iterations,
            conversation_id=conversation_id  # Pass the conversation_id to _iterative_query
        )
        
        if self.debug:
            print("\n=== Query Process Complete ===")
            print(f"Status: {result['status']}")
        
        return result

    async def recommend_material(self, 
                               application_description: str,
                               constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recommend materials for an application through iterative analysis.
        
        Args:
            application_description: Description of the intended use
            constraints: Optional dict of specific constraints like:
                - wavelength_range: Tuple[float, float]
                - min_citation_count: int
                - max_temperature: float
                - etc.
        """
        # Format question with constraints
        question = f"""Find the best material for this application: '{application_description}'"""
        if constraints:
            question += "\nWith these specific constraints:"
            for key, value in constraints.items():
                question += f"\n- {key}: {value}"
            
        return await self._iterative_query(question)

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format query results for display."""
        output = []
        
        for i, query in enumerate(results["queries"], 1):
            output.append(f"\nStep {i}:")
            output.append(f"Reasoning: {query['reasoning']}")
            output.append(f"Query: {query['query']}")
            output.append(f"Results: {query['results']}")
            output.append(f"Analysis: {query['analysis']}")
            
        output.append("\nFinal Recommendation:")
        output.append(results["final_recommendation"])
        
        return "\n".join(output)
