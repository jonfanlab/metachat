# metachat-aim

Code for reproducing the evaluation of AIM MetaChat on the Stanford Nanophotonics Benchmark from [*A multi-agentic framework for real-time, autonomous freeform metasurface design*](https://www.science.org/doi/10.1126/sciadv.adx8006).

![AIM evaluation overview](../figs/fig2.png)

## Directory structure

- `agent/`: Agent implementations (e.g., `StandardAgent`, *AIM* `IterativeAgent`, tool-augmented variants)
- `core/models/`: LLM wrappers (`OpenAIModel`, `AnthropicModel`, `LlamaModel`)
- `tools/`: Domain tools (design utilities, materials DB, solvers)
- `experiments/`: Benchmarks, evaluation framework, and runner scripts
  - `benchmarks/`: Evaluation datasets (e.g., `metachat_eval_v1_corrected.json`)
  - `eval_framework/`: Grader and utilities for automated scoring
  - `runners/`: Scripts to run evaluations (see `eval_runner.py`)

> Note: The web-scraped Materials Database `tools/material_db/materials.db` can be downloaded from the [MetaChat Zenodo record](https://zenodo.org/records/15802727).

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set API keys (as needed by selected models):
   - `OPENAI_API_KEY` for OpenAI models
   - `TOGETHER_API_KEY` for Together-hosted Llama models
   - `ANTHROPIC_API_KEY` for Anthropic models

3. Run the benchmark evaluation:
   ```bash
   python experiments/runners/eval_runner.py
   ```

## Evaluating the benchmark with different LLMs and agentic architectures

The script `experiments/runners/eval_runner.py` orchestrates evaluation of the benchmark (`experiments/benchmarks/metachat_eval_v1_corrected.json`) across multiple LLMs and agent designs. It:

- Constructs one or more models (see the `models` dict)
- Instantiates an agent (default: *AIM* with tools and Materials Agent access `IterativeAgentToolsMaterials`)
- Runs problems concurrently and grades answers with `AnswerGrader`
- Writes incremental logs and aggregated results under `experiments/logs/` and `experiments/results_*`

To change the agent or models:
- Edit the agent instantiation inside `run_evaluation(...)` (e.g., switch to `StandardAgent`)
- Modify the `models` dictionary in `main()` to add/remove LLMs

## Notes

- Concurrency is controlled via `max_concurrent` in `run_evaluation`
- Results are timestamped and organized by agent and model names
- Ensure the required environment variables are set for the models you enable


