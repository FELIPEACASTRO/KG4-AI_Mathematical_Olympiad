"""
Answer extraction from LLM outputs.
Extracts integer answers from \\boxed{}, plain text, and code output patterns.
"""

import re
from typing import Optional


def extract_answer(text: str) -> Optional[int]:
    """
    Extract the final integer answer from an LLM response.
    Tries multiple strategies in order of reliability.
    
    Returns an integer in [0, 99999] or None if no valid answer found.
    """
    if not text:
        return None

    # Strategy 1: Last \\boxed{...} pattern (most reliable)
    answer = _extract_from_boxed(text)
    if answer is not None:
        return answer

    # Strategy 2: "final answer is X" or "answer is X" patterns
    answer = _extract_from_answer_phrase(text)
    if answer is not None:
        return answer

    # Strategy 3: "= X" at end of mathematical expression
    answer = _extract_from_equals(text)
    if answer is not None:
        return answer

    # Strategy 4: Last standalone integer in the text
    answer = _extract_last_integer(text)
    if answer is not None:
        return answer

    return None


def _extract_from_boxed(text: str) -> Optional[int]:
    """Extract answer from \\boxed{...} LaTeX notation."""
    # Find all \boxed{...} patterns, take the last one
    patterns = [
        r'\\boxed\{(\s*\d+\s*)\}',
        r'\\boxed\{(\s*-?\d+\s*)\}',
        r'\\boxed\{\s*(\d[\d,\s]*)\s*\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            raw = matches[-1].strip().replace(',', '').replace(' ', '')
            return _validate_answer(raw)
    return None


def _extract_from_answer_phrase(text: str) -> Optional[int]:
    """Extract from 'the answer is X' type phrases."""
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(\d+)',
        r'(?:the\s+)?(?:final\s+)?answer\s*[:=]\s*(\d+)',
        r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?answer\s+is\s*(\d+)',
        r'answer\s*:\s*(\d+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return _validate_answer(matches[-1])
    return None


def _extract_from_equals(text: str) -> Optional[int]:
    """Extract from '= X' at end of computation."""
    patterns = [
        r'=\s*(\d+)\s*$',
        r'=\s*(\d+)\s*[.\n]',
        r'≡\s*(\d+)\s*(?:\(?\s*mod\s)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            return _validate_answer(matches[-1])
    return None


def _extract_last_integer(text: str) -> Optional[int]:
    """Extract the last standalone integer from text (fallback)."""
    # Look for integers at word boundaries near end of text
    # Use the last 500 chars to avoid grabbing random numbers from reasoning
    tail = text[-500:]
    matches = re.findall(r'\b(\d{1,5})\b', tail)
    if matches:
        return _validate_answer(matches[-1])
    return None


def _validate_answer(raw: str) -> Optional[int]:
    """Validate that an extracted string is a valid AIMO3 answer (0-99999)."""
    try:
        value = int(raw)
        if 0 <= value <= 99999:
            return value
    except (ValueError, OverflowError):
        pass
    return None


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from LLM output."""
    # Match ```python ... ``` or ```Python ... ``` or ``` ... ```
    pattern = r'```(?:[Pp]ython)?\s*\n(.*?)```'
    blocks = re.findall(pattern, text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def extract_answer_from_code_output(output: str) -> Optional[int]:
    """Extract an integer answer from code execution output."""
    if not output:
        return None
    # Take the last line that contains a number
    lines = output.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        # Direct integer
        match = re.match(r'^(\d+)$', line)
        if match:
            return _validate_answer(match.group(1))
        # "Answer: X" or "Result: X"
        match = re.search(r'(?:answer|result|output)\s*[:=]\s*(\d+)', line, re.IGNORECASE)
        if match:
            return _validate_answer(match.group(1))
        # Any integer in the line
        match = re.search(r'\b(\d{1,5})\b', line)
        if match:
            return _validate_answer(match.group(1))
    return None
