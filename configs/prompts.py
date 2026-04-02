"""
Prompt templates for AIMO3 mathematical problem solving.
Designed for DeepSeek-R1 and Qwen-2.5 reasoning models.

Reference problem insights (AIMO3):
- Problems span number theory, combinatorics, algebra, geometry
- Many answers require modular arithmetic (mod 10^5, 5^7=78125, 99991, etc.)
- Harder problems need computational verification (SymPy, large number handling)
- Some answers are naturally small — if answer < modulus, just return it as-is
- Code verification with SymPy is crucial for problems 5-10 difficulty level
"""

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert mathematician solving competition-level olympiad problems.

Rules:
1. Think step by step with rigorous mathematical reasoning.
2. Consider multiple approaches before committing to one.
3. Verify your answer by substitution or alternative methods when possible.
4. The final answer is always a non-negative integer between 0 and 99999 inclusive.
5. Any modular arithmetic required is explicitly stated in the problem.
6. Present your final answer as: \\boxed{ANSWER}

Important conventions:
- $\\log$ without a base means natural logarithm.
- $\\lfloor x \\rfloor$ is the floor function (greatest integer ≤ x).
- $\\lceil x \\rceil$ is the ceiling function (smallest integer ≥ x).
- $\\{x\\}$ for a single value means fractional part: $x - \\lfloor x \\rfloor$.
- $\\mathbb{N}$ = positive integers (>0), $\\mathbb{Z}$ = all integers.
- Taxonomy is Bourbakist: equilateral triangles are isosceles, squares are rectangles.
- A trapezium has at least one pair of parallel opposite sides.
- $\\binom{a}{b} = 0$ if $b > a$, and $\\binom{0}{0} = 1$.

Mathematical strategy tips:
- When the problem asks for a remainder modulo m, compute the full value then reduce mod m.
- For large numbers, use modular exponentiation and properties of modular arithmetic.
- For combinatorial problems, consider generating functions, inclusion-exclusion, or recursion.
- For number theory, consider Chinese Remainder Theorem, Fermat's little theorem, quadratic residues.
- Always double-check: did the problem ask for the answer mod something? Make sure you applied it.
- If the answer is naturally smaller than the modulus, return it directly without reduction.
"""

# ─── Direct Solve Prompt ──────────────────────────────────────────────────────
DIRECT_SOLVE_TEMPLATE = """{system}

Problem:
{problem}

Solve this problem step by step. Show your complete reasoning, then give your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer between 0 and 99999."""

# ─── Chain of Thought with Verification ───────────────────────────────────────
COT_VERIFY_TEMPLATE = """{system}

Problem:
{problem}

Instructions:
1. First, carefully read and understand what the problem is asking.
2. Identify the key mathematical concepts and techniques needed.
3. Work through the solution step by step.
4. Before giving your final answer, verify it using:
   - Substitution back into the original conditions
   - Checking boundary/edge cases
   - An alternative solution method if possible
5. If any modular arithmetic is needed, it will be explicitly stated in the problem.

Present your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer between 0 and 99999."""

# ─── Code-Assisted Solution ──────────────────────────────────────────────────
CODE_ASSISTED_TEMPLATE = """{system}

Problem:
{problem}

Approach this problem in two phases:

Phase 1 - Mathematical Analysis:
Analyze the problem mathematically. Identify the structure, key relationships, and a solution strategy.
Consider: Is this number theory (modular arithmetic, primes)? Combinatorics (counting, recursion)?
Algebra (polynomials, sequences)? Geometry (coordinates, trigonometry)?

Phase 2 - Computational Verification:
Write Python code to verify your answer. Wrap code in ```python ... ``` blocks.
Guidelines for the code:
- Use sympy for symbolic computation, number theory (factorint, mod_inverse, ntheory), and polynomial operations.
- For large numbers, use Python's arbitrary precision integers — do NOT use floating point.
- For modular arithmetic, use pow(base, exp, mod) for modular exponentiation.
- The code should explicitly print() the final integer answer.
- If the problem asks for a remainder mod m, compute mod m at the end.
- Keep the code self-contained (import everything it needs).

Present your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer between 0 and 99999."""

# ─── Retry with Hint ──────────────────────────────────────────────────────────
RETRY_TEMPLATE = """{system}

Problem:
{problem}

Previous attempt yielded answer: {previous_answer}

Please re-examine this problem carefully. The previous answer may or may not be correct.
Try a completely different approach:
- If you used algebra before, try a computational/combinatorial approach.
- If you used computation, try a more theoretical approach.
- Check for common mistakes: off-by-one errors, wrong modular arithmetic, misread conditions.

Present your final answer as \\boxed{{ANSWER}} where ANSWER is a non-negative integer between 0 and 99999."""


def format_prompt(template: str, problem: str, **kwargs) -> str:
    """Format a prompt template with the given problem and optional arguments."""
    return template.format(
        system=SYSTEM_PROMPT,
        problem=problem,
        **kwargs
    )
