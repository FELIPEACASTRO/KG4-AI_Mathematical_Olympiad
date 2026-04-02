"""
AIMO3 Competition Submission Notebook — Notebook 2/2
====================================================
This is the main inference notebook for the AI Mathematical Olympiad Progress Prize 3.

Dependencies: Installed via the utility notebook (notebook 1/2).
Model: DeepSeek-R1-Distill-Qwen-32B loaded via vLLM on H100 GPU.
Strategy: Multi-generation with majority voting for consistency.

Usage on Kaggle:
  1. Attach the utility notebook (1/2) as a data source.
  2. Attach the competition dataset.
  3. Attach the model from Kaggle Models (deepseek-ai/DeepSeek-R1-Distill-Qwen-32B).
  4. Select GPU H100 accelerator.
  5. Disable internet.
  6. Submit.
"""

import sys
import os
import time
import logging
import re
import math
from pathlib import Path
from collections import Counter
from typing import Optional

try:
    import polars as pl
except ImportError:
    pl = None

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('aimo3')

GLOBAL_START = time.time()
logger.info("AIMO3 Submission starting...")

# ─── Install dependencies from utility notebook ──────────────────────────────
# On Kaggle, the utility notebook output contains pre-built wheels
UTILITY_DIR = "/kaggle/input/aimo3-utility-notebook-dependency-install-1-2"
if os.path.exists(UTILITY_DIR):
    logger.info(f"Installing deps from utility notebook: {UTILITY_DIR}")
    os.system(f"pip install --no-index --find-links {UTILITY_DIR} vllm 2>/dev/null")

# ─── Configuration ────────────────────────────────────────────────────────────
# Model path — adjust based on how the model is attached on Kaggle
MODEL_PATHS = [
    "/kaggle/input/deepseek-r1-distill-qwen-32b/transformers/default/1",
    "/kaggle/input/deepseek-r1-distill-qwen-32b",
    "/kaggle/input/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

# vLLM configuration optimized for H100
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
    "swap_space": 4,
    "enable_chunked_prefill": True,
}

# Inference settings
NUM_GENERATIONS = 16         # Solutions per problem (more samples = better consensus)
MAX_TOKENS_CODE = 16384      # Max tokens for code-based generations
MAX_TOKENS_SHORT = 8192      # Max tokens for pure reasoning (saves budget for more gens)
TOTAL_TIME_BUDGET = 17700    # 4h55m in seconds (5min safety margin)
SETUP_TIME = 300             # Model loading time
CODE_EXEC_TIMEOUT = 30       # Timeout for SymPy verification

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert mathematician solving competition-level olympiad problems.

Rules:
1. Think step by step with rigorous mathematical reasoning.
2. Consider multiple approaches before committing to one.
3. Verify your answer by substitution or alternative methods when possible.
4. The final answer is always a non-negative integer between 0 and 99999 inclusive.
5. Any modular arithmetic required is explicitly stated in the problem.
6. Present your final answer as \\boxed{ANSWER}

Important conventions:
- $\\log$ without a base means natural logarithm.
- $\\lfloor x \\rfloor$ is the floor function (greatest integer <= x).
- $\\lceil x \\rceil$ is the ceiling function (smallest integer >= x).
- $\\{x\\}$ for a single value means fractional part: $x - \\lfloor x \\rfloor$.
- $\\mathbb{N}$ = positive integers (>0), $\\mathbb{Z}$ = all integers.
- Taxonomy is Bourbakist: equilateral triangles are isosceles, squares are rectangles.
- A trapezium has at least one pair of parallel opposite sides.
- $\\binom{a}{b} = 0$ if $b > a$, and $\\binom{0}{0} = 1$.

Mathematical strategy tips:
- The modulus varies by problem (10^5, 5^7, 99991, etc.) — read the problem carefully.
- If the answer is naturally smaller than the modulus, return it directly without extra reduction.
- For very large exponents (e.g. 3^{n!}): use Fermat-Euler theorem — a^{phi(m)} ≡ 1 (mod m) when gcd(a,m)=1.
- For p-adic valuations of a^n ± b^n: use the Lifting the Exponent Lemma (LTE).
- For prime factorization of n!: use Legendre's formula — v_p(n!) = sum floor(n/p^k).
- For counting balanced parenthesizations or lattice paths: Catalan numbers C_n = (2n)! / (n!(n+1)!).
- For polynomial divisibility involving x^n ± 1: factor via cyclotomic polynomials.
- For combinatorial problems, consider generating functions, inclusion-exclusion, or recursion.
- For number theory: Chinese Remainder Theorem, Fermat's little theorem, quadratic residues.
- For geometry: Stewart's theorem, power of a point, radical axes, spiral similarities.
- For sequences: look for Fibonacci-like structure, Binet's formula, golden ratio.
- Always double-check: did the problem ask for the answer mod something? Make sure you applied it."""

# Few-shot examples (short, high-signal, diverse problem types)
FEW_SHOT_EXAMPLES = """Here are two solved examples to illustrate the expected format:

Example 1 (Geometry):
Problem: Let $ABC$ be an acute-angled triangle with integer side lengths and $AB<AC$. [...] Find the remainder when $abc$ is divided by $10^{5}$.
Solution: Through angle chasing and the constraint that Y lies on line AD, we derive that the triangle must satisfy specific Diophantine conditions. Testing minimal perimeters with integer sides, the unique minimal triangle gives $abc \\pmod{10^5} = \\\\boxed{336}$.

Example 2 (Combinatorics):
Problem: A fair coin is tossed within a given time period, and heads come up [...]. Find the expected number of tosses.
Solution: By linearity of expectation and geometric series, the answer is $\\boxed{50}$."""

# Prompt templates
SOLVE_TEMPLATE = """{system}

{few_shot}

Now solve the following problem:

Problem:
{problem}

Solve this problem step by step. Show your complete reasoning, then give your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer between 0 and 99999."""

CODE_TEMPLATE = """{system}

Problem:
{problem}

Approach this problem in two phases:

Phase 1 - Mathematical Analysis:
Analyze the problem mathematically. Identify the structure, key relationships, and a solution strategy.
Consider: Is this number theory (modular arithmetic, primes)? Combinatorics (counting, recursion)?
Algebra (polynomials, sequences)? Geometry (coordinates, trigonometry)?
For extremely large numbers (factorials, towers of exponents), find a theoretical reduction before computing.

Phase 2 - Computational Verification:
Write Python code to verify your answer. Wrap code in ```python ... ``` blocks.
Guidelines for the code:
- Use sympy for symbolic computation: factorint, totient, mod_inverse, divisor_sigma, cyclotomic_poly.
- For large numbers, use Python's arbitrary precision integers — do NOT use floating point.
- For modular arithmetic, use pow(base, exp, mod) for modular exponentiation.
- For p-adic valuations: sympy.multiplicity(p, n) or manual computation.
- For prime factorization of n!: Legendre's formula sum(n // p**k for k in range(1, ...)).
- For Euler's totient: sympy.totient(n). For Mobius: sympy.mobius(n).
- For Fibonacci: compute iteratively or use matrix exponentiation for large n.
- The code should explicitly print() the final integer answer.
- If the problem asks for a remainder mod m, compute mod m at the end.
- Keep the code self-contained (import everything it needs).

Present your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer between 0 and 99999."""

THEORY_TEMPLATE = """{system}

{few_shot}

Now solve the following problem:

Problem:
{problem}

This is a challenging olympiad problem. Before attempting any computation:

1. Classify the problem type: number theory, combinatorics, algebra, geometry, or hybrid.
2. Identify key mathematical structures (group theory, polynomial rings, metric spaces, etc.).
3. List relevant theorems: Fermat-Euler, LTE lemma, Legendre's formula, Vieta's, Burnside's lemma, etc.
4. Determine if the problem involves impossibly large numbers requiring theoretical reduction.
5. Solve the problem rigorously using the identified theory.
6. Write verification code if the answer can be checked computationally.

Present your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer between 0 and 99999."""

RETRY_TEMPLATE = """{system}

{few_shot}

Problem:
{problem}

Previous attempt yielded answer: {previous_answer}

Re-examine this problem carefully using a completely different approach.
If previous approach was purely algebraic, try computational/code verification.
If previous approach was computational, try a more theoretical/structural argument.
Present your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer between 0 and 99999."""

# Brute-force / computational search template
COMPUTE_TEMPLATE = """{system}

Problem:
{problem}

Solve this problem using a COMPUTATIONAL approach:

1. Identify what quantity needs to be computed.
2. Write a Python program to search for or compute the answer directly.
3. For large search spaces, use intelligent pruning or mathematical shortcuts.
4. Use sympy, itertools, and Python's arbitrary precision as needed.
5. The code should print() just the final integer answer.
6. Wrap all code in ```python ... ``` blocks.

If computation confirms a value, present your final answer as \\boxed{{ANSWER}}."""

# Verification template — asks model to confirm or disprove a candidate answer
VERIFY_TEMPLATE = """{system}

Problem:
{problem}

A candidate answer is: {candidate_answer}

Your task:
1. Verify whether {candidate_answer} is correct by checking ALL problem constraints.
2. Substitute the answer back into the problem and confirm consistency.
3. Write verification code in ```python ... ``` blocks to double-check computationally.
4. If the candidate is WRONG, solve the problem correctly from scratch.
5. Present your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer between 0 and 99999."""


# ═══════════════════════════════════════════════════════════════════════════════
# PROBLEM CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_problem(problem: str) -> str:
    """Classify problem type by keywords. Returns best-guess category."""
    text = problem.lower()
    scores = {
        'number_theory': 0,
        'combinatorics': 0,
        'algebra': 0,
        'geometry': 0,
    }
    # Number theory signals
    for kw in ['prime', 'divisor', 'gcd', 'lcm', 'modulo', 'mod ', 'remainder',
               'coprime', 'congruent', 'euler', 'fermat', 'digit', 'factorial',
               'divides', 'divisible', 'pmod', 'totient', 'residue']:
        if kw in text:
            scores['number_theory'] += 2
    # Combinatorics signals
    for kw in ['how many', 'count', 'number of ways', 'probability', 'permutation',
               'combination', 'choose', 'arrange', 'distribute', 'subset',
               'sequence', 'binary string', 'lattice path', 'catalan', 'recurrence',
               'expected', 'coin', 'dice', 'random']:
        if kw in text:
            scores['combinatorics'] += 2
    # Algebra signals
    for kw in ['polynomial', 'equation', 'root', 'coefficient', 'sum of',
               'product of', 'series', 'sequence', 'function', 'inequality',
               'maximize', 'minimize', 'minimum', 'maximum', 'real number']:
        if kw in text:
            scores['algebra'] += 2
    # Geometry signals
    for kw in ['triangle', 'circle', 'angle', 'area', 'perimeter', 'point',
               'line', 'segment', 'perpendicular', 'parallel', 'inscribed',
               'circumscribed', 'median', 'altitude', 'polygon', 'quadrilateral',
               'coordinate', 'distance', 'radius', 'diameter']:
        if kw in text:
            scores['geometry'] += 2
    
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'algebra'  # default to algebra


def get_template_order(problem_type: str) -> list:
    """Return optimized template rotation based on problem type."""
    if problem_type == 'number_theory':
        # NT benefits most from code verification (modular arithmetic)
        return [CODE_TEMPLATE, COMPUTE_TEMPLATE, SOLVE_TEMPLATE, CODE_TEMPLATE, THEORY_TEMPLATE]
    elif problem_type == 'combinatorics':
        # Combinatorics: brute-force search is often decisive
        return [CODE_TEMPLATE, COMPUTE_TEMPLATE, THEORY_TEMPLATE, SOLVE_TEMPLATE, CODE_TEMPLATE]
    elif problem_type == 'geometry':
        # Geometry: theory-first (synthetic reasoning), then verify with code
        return [THEORY_TEMPLATE, SOLVE_TEMPLATE, CODE_TEMPLATE, COMPUTE_TEMPLATE, SOLVE_TEMPLATE]
    else:  # algebra or unknown
        return [CODE_TEMPLATE, SOLVE_TEMPLATE, THEORY_TEMPLATE, COMPUTE_TEMPLATE, CODE_TEMPLATE]


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_answer_from_segment(text: str) -> Optional[int]:
    """Extract integer answer from a text segment."""
    if not text:
        return None
    # \\boxed{...}
    for pattern in [r'\\boxed\{(\s*\d+\s*)\}', r'\\boxed\{\s*(\d[\d,\s]*)\s*\}']:
        matches = re.findall(pattern, text)
        if matches:
            raw = matches[-1].strip().replace(',', '').replace(' ', '')
            try:
                v = int(raw)
                if 0 <= v <= 99999:
                    return v
            except ValueError:
                pass
    # "answer is X"
    for pattern in [r'(?:final\s+)?answer\s+is\s*[:\s]*(\d+)', r'answer\s*[:=]\s*(\d+)']:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                v = int(matches[-1])
                if 0 <= v <= 99999:
                    return v
            except ValueError:
                pass
    # Last integer in tail
    tail = text[-500:]
    matches = re.findall(r'\b(\d{1,5})\b', tail)
    if matches:
        try:
            v = int(matches[-1])
            if 0 <= v <= 99999:
                return v
        except ValueError:
            pass
    return None


def extract_answer(text: str) -> Optional[int]:
    """Extract integer answer from LLM output, aware of DeepSeek <think> tags.
    
    DeepSeek-R1 outputs <think>...</think> followed by the final response.
    We prioritize the answer from AFTER </think> (the polished response),
    falling back to the full text if no answer is found there.
    """
    if not text:
        return None
    # Try extracting from after </think> first (DeepSeek-R1 format)
    think_end = text.rfind('</think>')
    if think_end != -1:
        after_think = text[think_end + len('</think>'):]
        ans = _extract_answer_from_segment(after_think)
        if ans is not None:
            return ans
    # Fall back to full text
    return _extract_answer_from_segment(text)


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from LLM output."""
    pattern = r'```(?:[Pp]ython)?\s*\n(.*?)```'
    blocks = re.findall(pattern, text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# CODE EXECUTION (Simple thread-based sandbox)
# ═══════════════════════════════════════════════════════════════════════════════

def execute_code(code: str, timeout: int = CODE_EXEC_TIMEOUT) -> tuple[bool, str]:
    """Execute Python code in a subprocess with timeout for reliable isolation."""
    import subprocess
    import tempfile

    if not code or not code.strip():
        return False, ""

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                          encoding='utf-8') as f:
            wrapper = (
                "import math, itertools, functools, collections, fractions, decimal\n"
                "try:\n    import sympy\nexcept ImportError:\n    pass\n"
            )
            f.write(wrapper + code)
            tmp_path = f.name

        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        os.unlink(tmp_path)

        if proc.returncode == 0:
            return True, proc.stdout
        else:
            return False, proc.stderr[-500:] if proc.stderr else "error"

    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False, "timeout"
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def _parse_answer_from_output(output: str) -> Optional[int]:
    """Parse an integer answer from code execution output."""
    lines = output.strip().split('\n')
    # Pass 1: look for explicit answer patterns in any line (last match wins)
    for line in reversed(lines):
        stripped = line.strip()
        for pat in [r'(?:final\s+)?answer\s*(?:is|=|:)\s*([\d.]+)',
                    r'result\s*(?:is|=|:)\s*([\d.]+)']:
            m = re.search(pat, stripped, re.IGNORECASE)
            if m:
                try:
                    v = int(float(m.group(1)))
                    if 0 <= v <= 99999:
                        return v
                except (ValueError, OverflowError):
                    pass
    # Pass 2: last line with a number (handles bare print(42) or print(42.0))
    for line in reversed(lines):
        stripped = line.strip()
        # Try the whole line as a number first (most common: just "336")
        try:
            v = int(float(stripped))
            if 0 <= v <= 99999:
                return v
        except (ValueError, OverflowError):
            pass
        # Fall back to extracting last integer token
        m = re.search(r'(\d+)', stripped)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 99999:
                    return v
            except ValueError:
                pass
    return None


def _try_repair_code(code: str, error_msg: str) -> Optional[str]:
    """Attempt to repair common code errors."""
    repaired = code
    # Missing imports — detect NameError for common math modules
    import_fixes = {
        'sympy': 'import sympy\nfrom sympy import *',
        'numpy': 'import numpy as np',
        'Fraction': 'from fractions import Fraction',
        'combinations': 'from itertools import combinations, permutations, product',
        'permutations': 'from itertools import combinations, permutations, product',
        'product': 'from itertools import product',
        'Counter': 'from collections import Counter',
        'defaultdict': 'from collections import defaultdict',
        'gcd': 'from math import gcd',
        'comb': 'from math import comb',
        'factorial': 'from math import factorial',
        'Decimal': 'from decimal import Decimal',
    }
    for name, imp in import_fixes.items():
        if f"name '{name}' is not defined" in error_msg or f"name '{name.lower()}' is not defined" in error_msg:
            repaired = imp + '\n' + repaired
    if repaired == code:
        return None  # No fix applied
    return repaired


def verify_with_code(llm_output: str) -> Optional[int]:
    """Extract and run code from LLM output, return computed answer.
    
    Strategy:
    1. Try each code block individually (last first — usually the final solution).
    2. If individual blocks fail, try concatenating ALL blocks as one script
       (handles progressive code where later blocks depend on earlier ones).
    3. If a block fails with a fixable error, attempt auto-repair and retry.
    """
    blocks = extract_code_blocks(llm_output)
    if not blocks:
        return None
    
    # Strategy 1: Try each block individually (last first)
    for code in reversed(blocks):
        ok, output = execute_code(code)
        if ok and output.strip():
            ans = _parse_answer_from_output(output)
            if ans is not None:
                return ans
        # Strategy 3: Auto-repair on failure
        if not ok and output:
            repaired = _try_repair_code(code, output)
            if repaired:
                ok2, output2 = execute_code(repaired)
                if ok2 and output2.strip():
                    ans = _parse_answer_from_output(output2)
                    if ans is not None:
                        return ans
    
    # Strategy 2: Concatenate all blocks as one script
    if len(blocks) > 1:
        combined = '\n\n'.join(blocks)
        ok, output = execute_code(combined)
        if ok and output.strip():
            ans = _parse_answer_from_output(output)
            if ans is not None:
                return ans
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# VOTING
# ═══════════════════════════════════════════════════════════════════════════════

def majority_vote(answers: list[Optional[int]]) -> Optional[int]:
    """Simple majority vote over valid answers."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def confidence_of(answers: list[Optional[int]]) -> tuple[Optional[int], float]:
    """Return (best_answer, confidence_ratio)."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0
    counter = Counter(valid)
    best, count = counter.most_common(1)[0]
    return best, count / len(valid)


def weighted_vote(text_answers: list[Optional[int]],
                 code_answers: list[Optional[int]]) -> tuple[Optional[int], float]:
    """Weighted voting: code-verified answers get 3x weight.
    
    When text and code from the same generation agree, that sample
    gets an extra bonus (4x total). When they disagree, both are
    discarded (cross-validation).
    """
    counter: Counter = Counter()
    total_weight = 0
    
    for i in range(len(text_answers)):
        ta = text_answers[i]
        ca = code_answers[i] if i < len(code_answers) else None
        
        if ta is not None and ca is not None:
            if ta == ca:
                # Text and code agree → highest confidence (4x)
                counter[ta] += 4
                total_weight += 4
            else:
                # Text and code disagree → discard both (cross-validation)
                pass
        elif ca is not None:
            # Code-only answer → 3x weight
            counter[ca] += 3
            total_weight += 3
        elif ta is not None:
            # Text-only answer → 1x weight
            counter[ta] += 1
            total_weight += 1
    
    if not counter:
        return None, 0.0
    
    best, weight = counter.most_common(1)[0]
    confidence = weight / total_weight if total_weight > 0 else 0.0
    return best, confidence


def select_answer(text_answers: list[Optional[int]],
                  code_answers: list[Optional[int]]) -> int:
    """Multi-stage answer selection with weighted voting."""
    # Stage 1: Weighted vote (code-verified 3x, concordant 4x, cross-validated)
    best, conf = weighted_vote(text_answers, code_answers)
    if best is not None and conf >= 0.3:
        return best
    
    # Stage 2: Code-only consensus (any agreement among code answers)
    valid_code = [a for a in code_answers if a is not None]
    if valid_code:
        ca, cc = confidence_of(code_answers)
        if ca is not None:
            return ca
    
    # Stage 3: Plain majority vote on all text answers
    ans = majority_vote(text_answers)
    return ans if ans is not None else 0


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class AIMOSolver:
    """Main solver: LLM generation + extraction + voting pipeline."""

    def __init__(self, llm, sp_det, sp_div_factory, sp_explore_factory=None,
                 sp_refine_factory=None, num_gens=NUM_GENERATIONS):
        self.llm = llm
        self.sp_det = sp_det
        self.sp_div_factory = sp_div_factory      # callable(seed) -> SamplingParams (temp=0.6)
        self.sp_explore_factory = sp_explore_factory  # callable(seed) -> SamplingParams (temp=0.9)
        self.sp_refine_factory = sp_refine_factory    # callable(seed) -> SamplingParams (temp=0.3)
        self.num_gens = num_gens
        self.problems_solved = 0
        self.total_problems = 50

    def _problem_seed(self, problem: str) -> int:
        """Deterministic seed from problem text for consistent results across runs."""
        import hashlib
        return int(hashlib.sha256(problem.encode()).hexdigest()[:8], 16) % (2**31)

    def solve(self, problem: str) -> int:
        start = time.time()
        self.problems_solved += 1
        remaining_problems = self.total_problems - self.problems_solved + 1
        elapsed_global = time.time() - GLOBAL_START
        remaining_global = max(0, TOTAL_TIME_BUDGET - elapsed_global)
        time_budget = min(remaining_global / remaining_problems, 340)
        
        # Classify problem for template routing
        prob_type = classify_problem(problem)
        templates = get_template_order(prob_type)
        
        logger.info(
            f"[{self.problems_solved}/{self.total_problems}] "
            f"Type: {prob_type} | Budget: {time_budget:.0f}s | Global remaining: {remaining_global:.0f}s"
        )

        all_answers = []
        code_answers = []
        prob_seed = self._problem_seed(problem)

        # Phase 1: Deterministic generation (temperature=0)
        if time.time() - start < time_budget - 20:
            fmt_kwargs = dict(system=SYSTEM_PROMPT, problem=problem)
            prompt = CODE_TEMPLATE.format(**fmt_kwargs)
            texts = self._gen([prompt], self.sp_det)
            for t in texts:
                all_answers.append(extract_answer(t))
                code_answers.append(verify_with_code(t))

        # Phase 2: First batch — diverse exploration (temp=0.6)
        num_diverse = self.num_gens - 1
        batch1_size = min(8, num_diverse)
        if batch1_size > 0 and time.time() - start < time_budget - 20:
            prompts = []
            for i in range(batch1_size):
                tmpl = templates[i % len(templates)]
                fmt_kwargs = dict(system=SYSTEM_PROMPT, problem=problem)
                if '{few_shot}' in tmpl:
                    fmt_kwargs['few_shot'] = FEW_SHOT_EXAMPLES
                prompts.append(tmpl.format(**fmt_kwargs))
            
            sp_div = self.sp_div_factory(prob_seed)
            texts = self._gen(prompts, sp_div)
            for t in texts:
                all_answers.append(extract_answer(t))
                code_answers.append(verify_with_code(t))

        # Phase 2b: Second batch — adaptive temperature
        batch2_size = num_diverse - batch1_size
        if batch2_size > 0 and time.time() - start < time_budget - 20:
            # Check consensus from first batch to pick temperature
            _, early_conf = weighted_vote(all_answers, code_answers)
            if early_conf >= 0.6 and self.sp_refine_factory:
                sp_batch2 = self.sp_refine_factory(prob_seed + 50)
                logger.info(f"  Batch2: REFINE mode (conf={early_conf:.2f}, temp=0.3)")
            elif early_conf < 0.3 and self.sp_explore_factory:
                sp_batch2 = self.sp_explore_factory(prob_seed + 50)
                logger.info(f"  Batch2: EXPLORE mode (conf={early_conf:.2f}, temp=0.9)")
            else:
                sp_batch2 = self.sp_div_factory(prob_seed + 50)
                logger.info(f"  Batch2: STANDARD mode (conf={early_conf:.2f}, temp=0.6)")
            
            prompts = []
            for i in range(batch2_size):
                tmpl = templates[(batch1_size + i) % len(templates)]
                fmt_kwargs = dict(system=SYSTEM_PROMPT, problem=problem)
                if '{few_shot}' in tmpl:
                    fmt_kwargs['few_shot'] = FEW_SHOT_EXAMPLES
                prompts.append(tmpl.format(**fmt_kwargs))
            
            texts = self._gen(prompts, sp_batch2)
            for t in texts:
                all_answers.append(extract_answer(t))
                code_answers.append(verify_with_code(t))

        # Phase 3: Aggressive retry on low confidence (up to 3 rounds)
        best, conf = weighted_vote(all_answers, code_answers)
        retry_round = 0
        while conf < 0.4 and retry_round < 3 and time.time() - start < time_budget - 30:
            valid = [a for a in all_answers if a is not None]
            candidate_info = ""
            if valid:
                dist = Counter(valid).most_common(3)
                candidate_info = ", ".join(f"{v} ({c} votes)" for v, c in dist)
            
            prompt = RETRY_TEMPLATE.format(
                system=SYSTEM_PROMPT, problem=problem,
                previous_answer=candidate_info if candidate_info else "unknown",
                few_shot=FEW_SHOT_EXAMPLES
            )
            sp_retry = self.sp_div_factory(prob_seed + 100 + retry_round)
            texts = self._gen([prompt], sp_retry)
            for t in texts:
                all_answers.append(extract_answer(t))
                code_answers.append(verify_with_code(t))
            
            best, conf = weighted_vote(all_answers, code_answers)
            retry_round += 1
            logger.info(f"  Retry {retry_round}: best={best}, conf={conf:.2f}")

        # Phase 4: Verification — ask model to confirm best answer
        best_candidate = select_answer(all_answers, code_answers)
        _, pre_verify_conf = weighted_vote(all_answers, code_answers)
        if (pre_verify_conf < 0.8
            and best_candidate != 0
            and time.time() - start < time_budget - 40):
            verify_prompt = VERIFY_TEMPLATE.format(
                system=SYSTEM_PROMPT, problem=problem,
                candidate_answer=best_candidate
            )
            sp_verify = self.sp_div_factory(prob_seed + 200)
            texts = self._gen([verify_prompt], sp_verify)
            for t in texts:
                v_text = extract_answer(t)
                v_code = verify_with_code(t)
                if v_text is not None or v_code is not None:
                    all_answers.append(v_text)
                    code_answers.append(v_code)
            logger.info(f"  Verification: pre_conf={pre_verify_conf:.2f}")

        final = select_answer(all_answers, code_answers)
        elapsed = time.time() - start
        _, final_conf = weighted_vote(all_answers, code_answers)
        logger.info(
            f"  Result: {final} | Confidence: {final_conf:.2f} | "
            f"Answers: {all_answers} | Code: {code_answers} | Time: {elapsed:.1f}s"
        )
        return final

    def _gen(self, prompts, sp):
        """Generate from a list of prompts (batched for GPU efficiency)."""
        try:
            outputs = self.llm.generate(prompts, sp)
            return [c.text for o in outputs for c in o.outputs]
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Kaggle Inference Server
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Initialize model and start the Kaggle inference server."""
    
    # ─── Find Model Path ──────────────────────────────────────────────────
    model_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            logger.info(f"Found model at: {model_path}")
            break
    
    if model_path is None:
        # Try to find any model directory
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for d in kaggle_input.iterdir():
                if 'deepseek' in d.name.lower() or 'qwen' in d.name.lower():
                    # Walk to find the deepest directory with model files
                    for root, dirs, files in os.walk(d):
                        if any(f.endswith(('.safetensors', '.bin', 'config.json')) for f in files):
                            model_path = root
                            break
                    if model_path:
                        break
        
        if model_path is None:
            logger.error("No model found! Available inputs:")
            if Path("/kaggle/input").exists():
                for item in Path("/kaggle/input").iterdir():
                    logger.error(f"  {item}")
            raise FileNotFoundError("Model not found in any expected path")
    
    logger.info(f"Using model: {model_path}")

    # ─── Initialize vLLM ──────────────────────────────────────────────────
    from vllm import LLM, SamplingParams

    logger.info("Initializing vLLM engine...")
    t0 = time.time()
    
    llm = LLM(
        model=model_path,
        **VLLM_CONFIG,
    )
    
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Sampling params
    sp_det = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS_CODE,
        stop=["</s>", "<|endoftext|>", "<|im_end|>"],
    )

    def make_sp_div(seed):
        """Create diverse SamplingParams with a deterministic seed.
        
        Using a per-problem seed ensures identical results across both
        competition runs, maximizing consistency score.
        Uses MAX_TOKENS_SHORT to allow more generations within budget.
        """
        return SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            max_tokens=MAX_TOKENS_SHORT,
            seed=seed,
            stop=["</s>", "<|endoftext|>", "<|im_end|>"],
        )

    # ─── Warmup ───────────────────────────────────────────────────────────
    logger.info("Warming up model...")
    _ = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=32, temperature=0))
    logger.info(f"Setup completed in {time.time() - GLOBAL_START:.1f}s")

    def make_sp_explore(seed):
        """High-diversity SamplingParams (temp=0.9) for low-consensus problems."""
        return SamplingParams(
            temperature=0.9,
            top_p=0.98,
            top_k=80,
            max_tokens=MAX_TOKENS_SHORT,
            seed=seed,
            stop=["</s>", "<|endoftext|>", "<|im_end|>"],
        )

    def make_sp_refine(seed):
        """Low-diversity SamplingParams (temp=0.3) for high-consensus refinement."""
        return SamplingParams(
            temperature=0.3,
            top_p=0.90,
            top_k=30,
            max_tokens=MAX_TOKENS_SHORT,
            seed=seed,
            stop=["</s>", "<|endoftext|>", "<|im_end|>"],
        )

    # ─── Create Solver ────────────────────────────────────────────────────
    solver = AIMOSolver(llm, sp_det, make_sp_div, make_sp_explore, make_sp_refine)

    # ─── Prediction Function ──────────────────────────────────────────────
    def predict(*args):
        """Called by the Kaggle evaluation API for each problem.
        
        The gateway calls predict(*data_batch) where data_batch is a polars
        DataFrame with columns ['id', 'problem']. Polars DataFrame unpacking
        yields Series objects, so we receive (id_series, problem_series).
        """
        try:
            # Extract id and problem from polars Series args
            if len(args) == 2:
                id_col, prob_col = args
                if pl is not None and isinstance(id_col, pl.Series):
                    id_ = str(id_col[0])
                    problem = str(prob_col[0])
                else:
                    id_ = str(id_col)
                    problem = str(prob_col)
            elif len(args) == 1 and pl is not None and isinstance(args[0], pl.DataFrame):
                id_ = str(args[0]['id'][0])
                problem = str(args[0]['problem'][0])
            else:
                id_ = 'unknown'
                problem = str(args[0]) if args else ''

            logger.info(f"Problem {id_}: {problem[:100]}...")
            answer = solver.solve(problem)
            logger.info(f"Problem {id_}: answer = {answer}")
            return answer
        except Exception as e:
            logger.error(f"Problem failed: {e}")
            return 0

    # ─── Start Kaggle Inference Server ────────────────────────────────────
    import kaggle_evaluation.aimo_3_inference_server
    
    inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            (os.path.join('/kaggle/input/ai-mathematical-olympiad-progress-prize-3', 'test.csv'),)
        )


if __name__ == '__main__':
    main()
