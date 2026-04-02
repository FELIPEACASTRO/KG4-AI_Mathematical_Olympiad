# KG4 — AI Mathematical Olympiad Progress Prize 3

Competition submission for [AIMO Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) on Kaggle.

## Strategy

- **Model**: DeepSeek-R1-Distill-Qwen-32B via vLLM on H100 GPU
- **Approach**: Multi-generation (8 solutions per problem) with majority voting
- **Code verification**: LLM-generated SymPy code is executed to cross-check answers
- **Consistency**: Same model + voting ensures consistent answers across two competition runs

## Project Structure

```
├── notebooks/
│   ├── submission.py       # Main submission notebook (2/2) — self-contained
│   └── utility_setup.py    # Dependency installer notebook (1/2)
├── src/
│   ├── answer_extraction.py  # Multi-strategy answer extraction from LLM output
│   ├── code_executor.py      # Sandboxed Python/SymPy code execution
│   ├── voting.py             # Majority voting and consensus selection
│   └── solver.py             # MathSolver orchestrator (dev reference)
├── configs/
│   ├── model_config.py       # Model paths, vLLM settings, time budgets
│   └── prompts.py            # System prompt and prompt templates
├── evaluation/
│   └── local_eval.py         # Unit tests + optional GPU model evaluation
├── data/
│   ├── reference.csv         # 10 reference problems with known answers
│   ├── test.csv              # Competition test problems
│   └── kaggle_evaluation/    # Kaggle evaluation API (gateway, relay, gRPC)
└── requirements.txt
```

## Submission Workflow (Kaggle)

1. **Notebook 1/2** (`utility_setup.py`): Run with internet enabled to install vLLM and dependencies. Save the notebook version.
2. **Notebook 2/2** (`submission.py`): Attach utility notebook output as data source. Attach the DeepSeek model from Kaggle Models. Select H100 GPU. Disable internet. Submit.

## Local Development

```bash
pip install -r requirements.txt

# Run unit tests (no GPU needed)
python evaluation/local_eval.py

# Run model evaluation against reference problems (GPU required)
python evaluation/local_eval.py --with-model --max-problems 3
```

## Key Design Decisions

- **Self-contained submission**: `submission.py` inlines all logic (answer extraction, code execution, voting) for Kaggle deployment simplicity.
- **8 generations + majority vote**: Competition penalizes inconsistency (both runs must agree). Voting favors consensus over exploration.
- **Code-assisted verification**: When the LLM generates Python code blocks, we execute them and compare output to the LLM's stated answer for additional signal.
- **340s per problem budget**: 5 hours / ~53 problems = ~340s each, with time for model loading.
