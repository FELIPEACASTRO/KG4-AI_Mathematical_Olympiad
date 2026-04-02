"""
Majority voting and answer consensus system.
Critical for AIMO3: private LB runs 2x with consistency penalty.
Models that produce consistent answers score higher.
"""

import math
from collections import Counter
from typing import Optional


def majority_vote(answers: list[Optional[int]], weights: Optional[list[float]] = None) -> Optional[int]:
    """
    Select the most common answer using weighted majority voting.
    
    Args:
        answers: List of extracted integer answers (may contain None).
        weights: Optional weights for each answer (e.g., based on confidence).
    
    Returns:
        The answer with the highest weighted vote count, or None.
    """
    valid = [(a, w) for a, w in zip(answers, weights or [1.0] * len(answers)) if a is not None]
    if not valid:
        return None

    # Weighted vote counting
    vote_counts: dict[int, float] = {}
    for answer, weight in valid:
        vote_counts[answer] = vote_counts.get(answer, 0.0) + weight

    if not vote_counts:
        return None

    # Return the answer with the highest weighted count
    best_answer = max(vote_counts, key=vote_counts.get)
    return best_answer


def confidence_score(answers: list[Optional[int]]) -> tuple[Optional[int], float]:
    """
    Compute the best answer and its confidence score.
    
    Confidence = (votes for winner) / (total valid votes)
    Higher confidence means more consistency across generations.
    
    Returns:
        (best_answer, confidence) where confidence is in [0.0, 1.0]
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0

    counter = Counter(valid)
    best_answer, count = counter.most_common(1)[0]
    confidence = count / len(valid)

    return best_answer, confidence


def entropy_weighted_vote(
    answers: list[Optional[int]],
    log_probs: Optional[list[float]] = None,
) -> Optional[int]:
    """
    Entropy-weighted voting: weight each answer by the inverse entropy
    (or log probability) of the generation that produced it.
    
    Lower entropy = more confident generation = higher weight.
    
    Args:
        answers: List of extracted answers.
        log_probs: Optional list of average log probabilities per generation.
    
    Returns:
        The winning answer.
    """
    if log_probs is None:
        return majority_vote(answers)

    # Convert log probs to weights (higher log prob = more confident = higher weight)
    weights = []
    for lp in log_probs:
        if lp is not None:
            weights.append(math.exp(lp))  # Convert log prob to probability
        else:
            weights.append(1.0)

    return majority_vote(answers, weights)


def select_final_answer(
    answers: list[Optional[int]],
    code_verified_answers: Optional[list[Optional[int]]] = None,
    log_probs: Optional[list[float]] = None,
) -> int:
    """
    Select the final answer using a multi-stage strategy:
    
    1. If code-verified answers agree, use that (highest confidence).
    2. Otherwise, use entropy-weighted majority voting.
    3. If no valid answer, return 0 (safe default).
    
    Args:
        answers: All extracted answers from LLM generations.
        code_verified_answers: Answers verified by code execution.
        log_probs: Log probabilities for entropy weighting.
    
    Returns:
        A validated integer answer in [0, 99999].
    """
    # Stage 1: Check code-verified answers
    if code_verified_answers:
        valid_code = [a for a in code_verified_answers if a is not None]
        if valid_code:
            code_answer, code_confidence = confidence_score(code_verified_answers)
            if code_confidence >= 0.5:
                return code_answer

    # Stage 2: Entropy-weighted voting on all answers
    if log_probs:
        answer = entropy_weighted_vote(answers, log_probs)
        if answer is not None:
            return answer

    # Stage 3: Simple majority voting
    answer = majority_vote(answers)
    if answer is not None:
        return answer

    # Stage 4: Fallback — return most common among all valid, or 0
    valid = [a for a in answers if a is not None]
    if valid:
        return Counter(valid).most_common(1)[0][0]

    return 0
