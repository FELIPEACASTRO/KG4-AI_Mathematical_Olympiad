"""
Core solver pipeline for AIMO3 competition.
Orchestrates: LLM inference → answer extraction → code verification → voting.
"""

import time
import logging
from typing import Optional

from src.answer_extraction import extract_answer, extract_code_blocks, extract_answer_from_code_output
from src.code_executor import execute_code_simple
from src.voting import select_final_answer, confidence_score
from configs.prompts import (
    format_prompt,
    DIRECT_SOLVE_TEMPLATE,
    COT_VERIFY_TEMPLATE,
    CODE_ASSISTED_TEMPLATE,
    RETRY_TEMPLATE,
)

logger = logging.getLogger(__name__)


class MathSolver:
    """
    Solves olympiad math problems using an LLM with multi-strategy 
    generation and consistency-focused voting.
    """

    def __init__(self, llm_engine, sampling_params_deterministic, sampling_params_diverse, 
                 num_generations: int = 8, max_time_per_problem: int = 340,
                 code_timeout: int = 30):
        """
        Args:
            llm_engine: Initialized vLLM engine (or compatible interface).
            sampling_params_deterministic: SamplingParams for deterministic generation.
            sampling_params_diverse: SamplingParams for diverse generation.
            num_generations: Number of solutions to generate per problem.
            max_time_per_problem: Time budget in seconds per problem.
            code_timeout: Timeout for code execution verification.
        """
        self.llm = llm_engine
        self.sp_deterministic = sampling_params_deterministic
        self.sp_diverse = sampling_params_diverse
        self.num_generations = num_generations
        self.max_time = max_time_per_problem
        self.code_timeout = code_timeout

    def solve(self, problem: str) -> int:
        """
        Solve a single math problem.
        
        Strategy:
        1. Generate 1 deterministic solution (temperature=0) for consistency.
        2. Generate N-1 diverse solutions (temperature=0.6) for coverage.
        3. Extract answers from all solutions.
        4. Run code verification on solutions containing Python code.
        5. Apply weighted majority voting.
        
        Returns:
            Integer answer in [0, 99999].
        """
        start_time = time.time()
        all_answers = []
        code_verified_answers = []
        generation_texts = []

        # Phase 1: Deterministic generation (for consistency across runs)
        remaining = self._time_remaining(start_time)
        if remaining > 30:
            det_prompt = format_prompt(COT_VERIFY_TEMPLATE, problem)
            det_outputs = self._generate(det_prompt, self.sp_deterministic, n=1)
            for text in det_outputs:
                generation_texts.append(text)
                answer = extract_answer(text)
                all_answers.append(answer)
                # Try code verification
                code_answer = self._verify_with_code(text)
                code_verified_answers.append(code_answer)

        # Phase 2: Diverse generations with different prompts
        diverse_count = self.num_generations - len(generation_texts)
        templates = [
            DIRECT_SOLVE_TEMPLATE,
            CODE_ASSISTED_TEMPLATE,
            COT_VERIFY_TEMPLATE,
        ]

        remaining = self._time_remaining(start_time)
        if remaining > 30 and diverse_count > 0:
            # Split diverse generations across different prompt templates
            for i in range(diverse_count):
                remaining = self._time_remaining(start_time)
                if remaining < 20:
                    break

                template = templates[i % len(templates)]
                prompt = format_prompt(template, problem)
                outputs = self._generate(prompt, self.sp_diverse, n=1)
                
                for text in outputs:
                    generation_texts.append(text)
                    answer = extract_answer(text)
                    all_answers.append(answer)
                    code_answer = self._verify_with_code(text)
                    code_verified_answers.append(code_answer)

        # Phase 3: Check confidence and optionally retry
        best_answer, conf = confidence_score(all_answers)
        remaining = self._time_remaining(start_time)
        
        if conf < 0.5 and remaining > 60:
            # Low confidence: try one more with retry template
            retry_prompt = format_prompt(
                RETRY_TEMPLATE, problem,
                previous_answer=str(best_answer) if best_answer is not None else "unknown"
            )
            outputs = self._generate(retry_prompt, self.sp_diverse, n=1)
            for text in outputs:
                generation_texts.append(text)
                answer = extract_answer(text)
                all_answers.append(answer)
                code_answer = self._verify_with_code(text)
                code_verified_answers.append(code_answer)

        # Phase 4: Final answer selection via voting
        final_answer = select_final_answer(
            answers=all_answers,
            code_verified_answers=code_verified_answers if any(a is not None for a in code_verified_answers) else None,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Problem solved in {elapsed:.1f}s | "
            f"Generations: {len(generation_texts)} | "
            f"Answers: {all_answers} | "
            f"Code verified: {code_verified_answers} | "
            f"Final: {final_answer} (confidence: {conf:.2f})"
        )

        return final_answer

    def _generate(self, prompt: str, sampling_params, n: int = 1) -> list[str]:
        """Generate text using the LLM engine."""
        try:
            outputs = self.llm.generate([prompt], sampling_params)
            texts = []
            for output in outputs:
                for completion in output.outputs:
                    texts.append(completion.text)
            return texts
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return []

    def _verify_with_code(self, llm_output: str) -> Optional[int]:
        """Extract and execute code from LLM output for answer verification."""
        code_blocks = extract_code_blocks(llm_output)
        if not code_blocks:
            return None

        # Try executing the last code block (most likely the final solution)
        for code in reversed(code_blocks):
            success, output = execute_code_simple(code, timeout=self.code_timeout)
            if success:
                answer = extract_answer_from_code_output(output)
                if answer is not None:
                    return answer

        return None

    def _time_remaining(self, start_time: float) -> float:
        """Calculate remaining time for this problem."""
        return max(0, self.max_time - (time.time() - start_time))
