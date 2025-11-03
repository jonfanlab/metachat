from typing import Dict, Any, Optional, List
from core.models.base import BaseModel
import json
import re

class AnswerGrader:
    @staticmethod
    def check_numerical_tolerance(num1: float, num2: float) -> tuple[bool, str]:
        """
        Check if two numbers are within 2% tolerance of each other.
        Returns (within_tolerance, explanation)
        """
        try:
            percent_diff = abs(num1 - num2) / abs(num2) * 100
            within_tolerance = percent_diff <= 2.0
            explanation = f"Values differ by {percent_diff:.2f}%"
            return within_tolerance, explanation
        except ZeroDivisionError:
            return num1 == num2, "Expected value is 0"

    def __init__(self, model: BaseModel):
        self.model = model
        self.grading_prompt = """Solution to grade (IMPORTANT: LOOK ONLY HERE TO EXTRACT THE ANSWER):
{solution}

Expected answer (DO NOT EXTRACT THE ANSWER FROM THIS):
{expected}

Expected approach:
{approach}

If the answer contains numerical values that need to be compared, include them in number_checks.
Each number_check should contain the extracted and expected numbers without units.

Return ONLY a JSON object in this exact format:
{{
  "extracted_answer": "number with units, API call, database entries, explanation, etc. [Note: if an API call, extract only the API call and not also the explanation]", 
  "expected_answer": "number with units [Note: if extracted answer is a percentage, translate the expected answer to a percentage if it's a decimal to match or vice versa], API call, database entries, explanation, etc.",
  "extracted_answer_matches_expected": True/False, 
  "approach_matches": True/False, [IMPORTANT: Set to True if everything is correct but the numerical values are slightly different and need to be checked in number_checks.]
  "reason": "brief explanation of match/mismatch",
  "approach_feedback": "brief explanation of approach comparison",
  "number_checks": [
    {{ {{"extracted": number_1, "expected": number_1_expected, "description": "what this number represents"}}, {{"extracted": number_2, "expected": number_2_expected, "description": "what this number represents"}}, ...}} [Note: If numbers need to be checked from an API call, extract each of the numbers individualy to check them separately]
  ]
}}

The number_checks field is optional - only include it if numerical comparisons are needed. Remember to set extracted_answer_matches_expected to True if there are number_checks entries.

Here are chat logs to help with approach comparison:

Materials chat logs (if any):
{materials_chat}

Self chat logs (if any):
{self_chat}"""

    async def grade_solution(
        self, 
        solution: str, 
        expected_answer: str, 
        expected_approach: str,
        materials_chat_logs: Optional[List[Dict[str, Any]]] = None,
        self_chat_logs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        
        try:
            materials_chat_text = "None"
            if materials_chat_logs:
                chat_entries = []
                for entry in materials_chat_logs:
                    role = entry.get("role", "unknown")
                    content = entry.get("content", "")
                    chat_entries.append(f"{role}: {content}")
                materials_chat_text = "\n".join(chat_entries)

            self_chat_text = "None"
            if self_chat_logs:
                chat_entries = []
                for entry in self_chat_logs:
                    role = entry.get("role", "unknown")
                    content = entry.get("content", "")
                    chat_entries.append(f"{role}: {content}")
                self_chat_text = "\n".join(chat_entries)

            messages = [
                {"role": "system", "content": """You are a precise grading assistant. Evaluate both the numerical answer and the solution approach. 
If the solution is correct but the numerical values are slightly different and need to be checked in number_checks:
1. Include the numbers in the number_checks field of the specified JSON format for automated tolerance checking
2. IMPORTANT: Set extracted_answer_matches_expected to True.
Make sure to take units into account. The approach does not need to be exactly the same to be marked correct.
Ensure the whole answer is provided, not just part of it (e.g. make sure that WAVEYNET_API.design_metalens(A, B, C, D, E) is provided, not just A). Otherwise, matches_expected is false. 
DO NOT CONFUSE THE EXPECTED ANSWER AS PART OF THE EXTRACTED ANSWER WHEN GRADING. Respond only with valid JSON."""},
                {"role": "user", "content": self.grading_prompt.format(
                    solution=solution,
                    expected=expected_answer,
                    approach=expected_approach,
                    materials_chat=materials_chat_text,
                    self_chat=self_chat_text
                )}
            ]

            response = await self.model.generate(messages, temperature=0.0)
            
            result_text = response.content.strip()
            # Remove markdown code block markers if present
            if result_text.startswith('```'):
                # Remove everything before the first {
                result_text = result_text[result_text.find('{'):]
                # Remove everything after the last }
                result_text = result_text[:result_text.rfind('}')+1]

            # Clean the string: remove any hidden characters and normalize newlines
            result_text = ''.join(char for char in result_text if ord(char) >= 32 or char in '\n\r')
            result_text = result_text.replace('\r\n', '\n').replace('\r', '\n')
            result_text = result_text.strip()

            # Try to normalize the JSON string
            result_text = re.sub(r'\s+', ' ', result_text)  # Replace multiple spaces with single space
            result_text = result_text.replace('True', 'true').replace('False', 'false')  # Convert Python booleans to JSON booleans
            result_text = result_text.replace(' ,', ',').replace(', ', ',')  # Normalize spacing around commas
            result_text = result_text.replace(' :', ':').replace(': ', ':')  # Normalize spacing around colons

            try:
                grading_result = json.loads(result_text)
            except json.JSONDecodeError as e:
                print(f"Error position: {e.pos}")
                print(f"Error message: {e.msg}")
                print(f"Line number: {e.lineno}")
                print(f"Column number: {e.colno}")
                raise

            # Post-process numerical tolerances if present
            if "number_checks" in grading_result and grading_result["extracted_answer_matches_expected"]:
                all_within_tolerance = True
                tolerance_explanations = []
                
                for check in grading_result["number_checks"]:
                    within_tolerance, explanation = self.check_numerical_tolerance(
                        check["extracted"], 
                        check["expected"]
                    )
                    if not within_tolerance:
                        all_within_tolerance = False
                        tolerance_explanations.append(
                            f"{check['description']}: {explanation}"
                        )
                
                if not all_within_tolerance:
                    grading_result["extracted_answer_matches_expected"] = False
                    grading_result["reason"] += f" [Numerical tolerance check failed: {'; '.join(tolerance_explanations)}]"
                else:
                    grading_result["reason"] += f" [Numerical tolerance check passed: all values within 2% tolerance]"
            
            return {
                "matches_expected": grading_result["extracted_answer_matches_expected"],
                "approach_matches": grading_result["approach_matches"],
                "extracted_answer": grading_result["extracted_answer"],
                "reason": grading_result["reason"],
                "approach_feedback": grading_result["approach_feedback"]
            }
            
        except json.JSONDecodeError as e:
            return {
                "matches_expected": False,
                "approach_matches": False,
                "extracted_answer": None,
                "reason": f"Failed to parse grader response: {str(e)}. Raw response: {response.content}",
                "approach_feedback": "Grading failed"
            }
        except Exception as e:
            return {
                "matches_expected": False,
                "approach_matches": False,
                "extracted_answer": None,
                "reason": f"Grading error: {str(e)}",
                "approach_feedback": "Grading failed"
            }
