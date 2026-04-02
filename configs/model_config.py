"""
Model and inference configuration for AIMO3 competition.
All models must be open-weight, released before March 15, 2026.
"""

# ─── Primary Model Configuration ──────────────────────────────────────────────
# DeepSeek-R1-Distill-Qwen-32B: Strong math reasoning, fits on H100 80GB
PRIMARY_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
PRIMARY_MODEL_REVISION = "main"

# Fallback: smaller model if primary doesn't fit or for faster inference
FALLBACK_MODEL = "Qwen/Qwen2.5-32B-Instruct"
FALLBACK_MODEL_REVISION = "main"

# ─── vLLM Engine Configuration ────────────────────────────────────────────────
VLLM_CONFIG = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.92,
    "max_model_len": 32768,
    "dtype": "bfloat16",
    "seed": 42,
    "trust_remote_code": True,
    "enforce_eager": False,
    "disable_log_stats": True,
    "enable_prefix_caching": True,
}

# ─── Sampling Configuration ──────────────────────────────────────────────────
# For consistency (private LB runs 2x), we use deterministic sampling
# with temperature=0 for the primary attempt, and slightly higher for retries
SAMPLING_DETERMINISTIC = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 16384,
    "stop": ["</s>", "<|endoftext|>", "<|im_end|>"],
}

SAMPLING_DIVERSE = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": 16384,
    "stop": ["</s>", "<|endoftext|>", "<|im_end|>"],
}

# ─── Solver Configuration ────────────────────────────────────────────────────
# Number of solutions to generate per problem for majority voting
NUM_GENERATIONS = 8

# Maximum time budget per problem in seconds (5h = 18000s for 50 problems)
# Reserve 300s for setup, so ~354s per problem
MAX_TIME_PER_PROBLEM = 340

# Timeout for code execution (SymPy verification)
CODE_EXECUTION_TIMEOUT = 30

# ─── Time Budget Strategy ─────────────────────────────────────────────────────
# Total GPU time: 5 hours = 18000 seconds
TOTAL_TIME_BUDGET = 18000
SETUP_TIME = 300          # Model loading, warmup
SAFETY_MARGIN = 120       # Buffer to ensure completion
INFERENCE_BUDGET = TOTAL_TIME_BUDGET - SETUP_TIME - SAFETY_MARGIN
TIME_PER_PROBLEM = INFERENCE_BUDGET // 50  # ~351s per problem
