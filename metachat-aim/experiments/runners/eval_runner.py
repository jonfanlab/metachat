import os
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import dotenv
from tqdm import tqdm
import aiofiles
from asyncio import Lock

dotenv.load_dotenv()

from agent.cot_iterative_tools_materials import IterativeAgentToolsMaterials
from agent.cot_iterative_materials import IterativeAgentMaterials
from agent.cot_iterative_tools import IterativeAgentTools
from agent.cot_iterative import IterativeAgent
from agent.standard_agent_tools import StandardAgentToolsMaterials

from agent.standard_agent import StandardAgent
from core.models.openai import OpenAIModel
from core.models.anthropic import AnthropicModel
from core.models.llama import LlamaModel
from experiments.eval_framework.grader import AnswerGrader

async def run_evaluation(
    agent_name: str,
    model_name: str,
    eval_file: str,
    model: Any,
    output_dir: Optional[str] = None,
    max_concurrent: int = 10
) -> Dict[str, Any]:
    """Run evaluation for a specific agent/model combination."""
    
    start_time = datetime.now()
    
    # Create agent
    # agent = StandardAgentTools(model=model)
    # agent = StandardAgent(model=model)
    agent = IterativeAgentToolsMaterials(model=model)
    # Create dedicated grader with gpt-4o
    grader_model = OpenAIModel(
        model_name="gpt-4o", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    grader = AnswerGrader(grader_model)
    
    # Load problems
    with open(eval_file, 'r') as f:
        test_data = json.load(f)
    total_problems = len(test_data["problems"])
    
    print(f"\nStarting evaluation:")
    print(f"Agent: {agent_name}")
    print(f"Model: {model_name}")
    print(f"Problems: {total_problems}")
    print(f"Concurrent jobs: {max_concurrent}")
    print("-" * 50)
    
    # Initialize counters
    completed = 0
    successful = 0
    failed = 0
    
    # Initialize output file path
    output_file = None
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(output_dir) / agent_name / model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        output_file = result_dir / f"{Path(eval_file).stem}_{timestamp}.json"
        # Initialize empty results file
        with open(output_file, 'w') as f:
            json.dump([], f)

    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    
    # Add file lock
    file_lock = Lock()
    
    async def process_problem(problem: Dict[str, Any]):
        nonlocal completed, successful, failed
        async with semaphore:
            try:
                max_attempts = 3    # pass@3
                attempt = 1
                while attempt <= max_attempts:
                    # Get solution from agent
                    solution = await agent.solve(
                        problem["problem_statement"], 
                        problem["id"], 
                        temperature=0.0, 
                        disable_cache=True
                    )
                    
                    # Read materials chat logs if they exist
                    materials_chat_logs = []
                    log_file = Path(f"experiments/logs/eval_v1_corrected_singlestepfunc/cot_iterative_tools_materials/materials_chat") / f"{model_name}/{problem['id']}.json"
                    if log_file.exists():
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                materials_chat_logs = json.load(f)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse materials chat log for problem {problem['id']}")

                    self_chat_logs = []
                    log_file = Path(f"experiments/logs/eval_v1_corrected_singlestepfunc/cot_iterative_tools_materials/self_chat") / f"{model_name}/{problem['id']}.json"
                    if log_file.exists():
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                self_chat_logs = json.load(f)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse self chat log for problem {problem['id']}")

                    # Grade the solution
                    grade_result = await grader.grade_solution(
                        solution["solution"],
                        problem["answer"],
                        problem["expected_approach"],
                        materials_chat_logs=materials_chat_logs,
                        self_chat_logs=self_chat_logs
                    )
                    
                    # If answer is correct, break the loop
                    if grade_result["matches_expected"]:
                        break
                    
                    # If we still have attempts left, continue to next attempt
                    if attempt < max_attempts:
                        print(f"\nRetrying problem {problem['id']} (attempt {attempt + 1}/{max_attempts})")
                    
                    attempt += 1
                
                result = {
                    "problem_id": problem["id"],
                    "expected_answer": problem["answer"],
                    "agent_solution": solution["solution"],
                    "extracted_answer": grade_result["extracted_answer"],
                    "metadata": solution["metadata"],
                    "success": grade_result["matches_expected"],
                    "approach_matches": grade_result["approach_matches"],
                    "grading_explanation": grade_result["reason"],
                    "approach_feedback": grade_result["approach_feedback"],
                    "attempts_needed": attempt  # Add number of attempts to result
                }
                
                # Modified status to include approach check
                answer_status = "✓" if grade_result["matches_expected"] else "✗"
                approach_status = "A" if grade_result["approach_matches"] else "X"
                status = f"{answer_status}{approach_status}"
                if grade_result["matches_expected"]:
                    successful += 1
                else:
                    failed += 1
                
            except Exception as e:
                status = "✗"
                failed += 1
                result = {
                    "problem_id": problem["id"],
                    "success": False,
                    "error": str(e)
                }
            
            # Save intermediate results if output file specified
            if output_file:
                async with file_lock:  # Ensure only one write at a time
                    try:
                        async with aiofiles.open(output_file, 'r') as f:
                            content = await f.read()
                            current_results = json.loads(content) if content else []
                        current_results.append(result)
                        async with aiofiles.open(output_file, 'w') as f:
                            await f.write(json.dumps(current_results, indent=2))
                    except json.JSONDecodeError:
                        print(f"\nWarning: Could not read intermediate results for problem {problem['id']}, skipping save")
                    except Exception as e:
                        print(f"\nWarning: Error saving results for problem {problem['id']}: {str(e)}")
            
            completed += 1
            elapsed = datetime.now() - start_time
            remaining = total_problems - completed
            
            # Update progress information
            print(f"\rProgress: {completed}/{total_problems} "
                  f"[{successful} ✓, {failed} ✗] "
                  f"({completed/total_problems:.1%}) "
                  f"| Elapsed: {elapsed} "
                  f"| Remaining: ~{remaining} problems "
                  f"| {status} Problem {problem['id']} ", end="")
            
            return result, status

    # Create progress bar
    pbar = tqdm(total=total_problems, 
                desc="Processing problems",
                unit="problem")
    
    # Process all problems concurrently
    tasks = [process_problem(problem) for problem in test_data["problems"]]
    results_with_status = await asyncio.gather(*tasks)
    results = [r[0] for r in results_with_status]
    
    # Final statistics
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print("\n\nEvaluation Complete!")
    print("-" * 50)
    print(f"Total time: {total_time}")
    print(f"Problems processed: {completed}")
    print(f"Successful: {successful} ({successful/total_problems:.1%})")
    print(f"Failed: {failed} ({failed/total_problems:.1%})")
    print(f"Average time per problem: {total_time/total_problems}")
    print("-" * 50)
    
    return results

async def main():

    eval_files = [
        "experiments/benchmarks/metachat_eval_v1_corrected.json"
    ]
    
    # Initialize models
    models = {
        "gpt-4o": OpenAIModel(
            model_name="gpt-4o", 
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        "gpt-4o-mini": OpenAIModel(
            model_name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": LlamaModel(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            api_key=os.getenv("TOGETHER_API_KEY")
        ),
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": LlamaModel(
            model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_key=os.getenv("TOGETHER_API_KEY")
        ),
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": LlamaModel(
            model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            api_key=os.getenv("TOGETHER_API_KEY")
        ),
    }
    
    # Run evaluations
    for model_name, model in models.items():
        for eval_file in eval_files:
            print(f"\nEvaluating {model_name} on {eval_file}")
            results = await run_evaluation(
                agent_name="iterative_agent_tools_materials",
                model_name=model_name,
                eval_file=eval_file,
                model=model,
                output_dir="experiments/results_eval_v1_corrected_singlestepfunc/cot_iterative_tools_materials"
            )
            print(f"Completed: {len(results)} problems evaluated")

if __name__ == "__main__":
    asyncio.run(main())