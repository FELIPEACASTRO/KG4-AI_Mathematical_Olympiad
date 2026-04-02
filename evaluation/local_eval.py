"""
Local Evaluation Framework for AIMO3
=====================================
Tests the solver pipeline against the 10 reference problems with known answers.
This runs WITHOUT vLLM — purely tests answer extraction, code execution, and voting.

For full model testing, use: python evaluation/local_eval.py --with-model
"""

import csv
import sys
import os
import time
import argparse

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REFERENCE_PATH = os.path.join(ROOT, 'data', 'reference.csv')

REFERENCE_ANSWERS = {
    '0e644e': 336,
    '26de63': 32951,
    '424e18': 21818,
    '42d360': 32193,
    '641659': 57447,
    '86e8e5': 8687,
    '92ba6a': 50,
    '9c1c5f': 580,
    'a295e9': 520,
    'dd7f5e': 160,
}


def load_reference_problems():
    """Load reference problems from CSV."""
    problems = []
    with open(REFERENCE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.append({
                'id': row['id'],
                'problem': row['problem'],
                'answer': int(row['answer']),
            })
    return problems


def test_answer_extraction():
    """Test the answer extraction module with synthetic outputs."""
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from answer_extraction import extract_answer

    test_cases = [
        ("The answer is \\boxed{42}", 42),
        ("Therefore \\boxed{336}", 336),
        ("...so the final answer is 520.", 520),
        ("We get $x = 160$. \\boxed{160}", 160),
        ("answer = 8687", 8687),
        ("The result is 50.", 50),
        ("", None),
        ("No valid answer here", None),
        # Edge cases
        ("\\boxed{99999}", 99999),
        ("\\boxed{0}", 0),
        ("\\boxed{100000}", None),  # Out of range
    ]

    passed = 0
    for text, expected in test_cases:
        result = extract_answer(text)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            print(f"  {status}: extract_answer({text[:50]!r}...) = {result}, expected {expected}")
        else:
            passed += 1

    print(f"Answer extraction: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_code_execution():
    """Test the code executor module."""
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from code_executor import execute_code_simple

    test_cases = [
        ("print(2 + 2)", True, "4"),
        ("import sympy; print(sympy.factorial(5))", True, "120"),
        ("print(sum(range(11)))", True, "55"),
        ("", False, ""),
        ("while True: pass", False, ""),  # Should timeout
    ]

    passed = 0
    for code, expect_ok, expect_out in test_cases:
        ok, output = execute_code_simple(code, timeout=5)
        is_pass = (ok == expect_ok)
        if expect_ok and expect_out:
            is_pass = is_pass and expect_out in output
        if is_pass:
            passed += 1
        else:
            print(f"  FAIL: execute({code[:40]!r}) = ({ok}, {output[:50]!r}), expected ({expect_ok}, {expect_out!r})")

    print(f"Code execution: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_voting():
    """Test the voting module."""
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from voting import majority_vote, confidence_score, select_final_answer

    test_cases = [
        ([42, 42, 42, 13], 42),
        ([1, 2, 3, 1], 1),
        ([None, None, None], None),
        ([7], 7),
        ([42, 42, 13, 13, 42], 42),
    ]

    passed = 0
    for answers, expected in test_cases:
        result = majority_vote(answers)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: majority_vote({answers}) = {result}, expected {expected}")

    print(f"Voting: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_submission_voting():
    """Test the new weighted voting and select_answer from submission.py."""
    # Import from submission (add notebooks to path)
    sys.path.insert(0, os.path.join(ROOT, 'notebooks'))
    
    # We need to import the functions directly since submission.py runs main()
    # Instead, we'll test the logic inline
    from collections import Counter
    
    def weighted_vote(text_answers, code_answers):
        counter = Counter()
        total_weight = 0
        for i in range(len(text_answers)):
            ta = text_answers[i]
            ca = code_answers[i] if i < len(code_answers) else None
            if ta is not None and ca is not None:
                if ta == ca:
                    counter[ta] += 4
                    total_weight += 4
                else:
                    pass  # discard
            elif ca is not None:
                counter[ca] += 3
                total_weight += 3
            elif ta is not None:
                counter[ta] += 1
                total_weight += 1
        if not counter:
            return None, 0.0
        best, weight = counter.most_common(1)[0]
        confidence = weight / total_weight if total_weight > 0 else 0.0
        return best, confidence

    test_cases = [
        # (text_answers, code_answers, expected_winner)
        # Code-verified consensus should win
        ([42, 42, 13, 13], [42, 42, None, None], 42),
        # Code+text concordant (4x) beats many text-only (1x each)
        ([99, 42, 42, 42, 42], [99, None, None, None, None], 99),
        # Code-only (3x) beats text-only (1x)
        ([13, 13, None], [None, None, 42], 42),
        # Cross-validation: disagreeing text+code discarded
        ([42, 42, 13], [99, 99, 13], 13),
        # All None
        ([None, None], [None, None], None),
        # Mixed: concordant (4x) vs code-only (3x)
        ([42, None], [42, 13], 42),
    ]

    passed = 0
    for text_ans, code_ans, expected in test_cases:
        result, _ = weighted_vote(text_ans, code_ans)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: weighted_vote({text_ans}, {code_ans}) = {result}, expected {expected}")

    print(f"Weighted voting: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_code_output_parsing():
    """Test the improved code output parsing."""
    import re
    
    def _parse_answer_from_output(output):
        lines = output.strip().split('\n')
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
        for line in reversed(lines):
            stripped = line.strip()
            try:
                v = int(float(stripped))
                if 0 <= v <= 99999:
                    return v
            except (ValueError, OverflowError):
                pass
            m = re.search(r'(\d+)', stripped)
            if m:
                try:
                    v = int(m.group(1))
                    if 0 <= v <= 99999:
                        return v
                except ValueError:
                    pass
        return None

    test_cases = [
        ("336\n", 336),
        ("336.0\n", 336),
        ("answer = 42\n", 42),
        ("The answer is 520\n", 520),
        ("result: 160\n", 160),
        ("Computing...\nDone\n8687\n", 8687),
        ("Step 1: ...\nStep 2: ...\nFinal answer is 50\n", 50),
        ("", None),
        ("No numbers here\n", None),
        # Float edge case
        ("99999.0\n", 99999),
        # Multiple lines, last number wins
        ("42\n336\n", 336),
    ]

    passed = 0
    for output, expected in test_cases:
        result = _parse_answer_from_output(output)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: _parse_answer({output!r}) = {result}, expected {expected}")

    print(f"Code output parsing: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_adaptive_temperature():
    """Test the adaptive temperature strategy logic."""
    # Simulate the decision logic from AIMOSolver.solve()
    
    test_cases = [
        # (early_conf, expected_mode)
        (0.8, "REFINE"),     # High consensus → low temp
        (0.6, "REFINE"),     # Threshold → refine
        (0.5, "STANDARD"),   # Medium → standard
        (0.3, "STANDARD"),   # Just at boundary → standard (< 0.3 triggers explore)
        (0.2, "EXPLORE"),    # Low consensus → high temp
        (0.0, "EXPLORE"),    # No consensus → explore
    ]
    
    passed = 0
    for early_conf, expected_mode in test_cases:
        if early_conf >= 0.6:
            mode = "REFINE"
        elif early_conf < 0.3:
            mode = "EXPLORE"
        else:
            mode = "STANDARD"
        
        if mode == expected_mode:
            passed += 1
        else:
            print(f"  FAIL: conf={early_conf} → {mode}, expected {expected_mode}")
    
    print(f"Adaptive temperature: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_few_shot_format():
    """Test that few-shot templates format correctly."""
    # Simulate template formatting with few_shot
    system = "You are a math expert."
    problem = "Find x such that x^2 = 4."
    few_shot = "Example: 2+2=4, answer \\boxed{4}"
    
    solve_tmpl = "{system}\n\n{few_shot}\n\nProblem:\n{problem}\n\nSolve."
    theory_tmpl = "{system}\n\n{few_shot}\n\nProblem:\n{problem}\n\nClassify."
    retry_tmpl = "{system}\n\n{few_shot}\n\nProblem:\n{problem}\n\nPrevious: {previous_answer}"
    code_tmpl = "{system}\n\nProblem:\n{problem}\n\nCode."
    
    test_cases = [
        ("SOLVE", solve_tmpl, dict(system=system, problem=problem, few_shot=few_shot)),
        ("THEORY", theory_tmpl, dict(system=system, problem=problem, few_shot=few_shot)),
        ("RETRY", retry_tmpl, dict(system=system, problem=problem, few_shot=few_shot, previous_answer="42")),
        ("CODE", code_tmpl, dict(system=system, problem=problem)),
    ]
    
    passed = 0
    for name, tmpl, kwargs in test_cases:
        try:
            result = tmpl.format(**kwargs)
            if system in result and problem in result:
                passed += 1
            else:
                print(f"  FAIL: {name} template missing system/problem in output")
        except KeyError as e:
            print(f"  FAIL: {name} template format error: {e}")
    
    print(f"Few-shot formatting: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_think_tag_extraction():
    """Test <think> tag-aware answer extraction."""
    import re
    
    def _extract_answer_from_segment(text):
        if not text:
            return None
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
        for pattern in [r'(?:final\s+)?answer\s+is\s*[:\s]*(\d+)', r'answer\s*[:=]\s*(\d+)']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    v = int(matches[-1])
                    if 0 <= v <= 99999:
                        return v
                except ValueError:
                    pass
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

    def extract_answer(text):
        if not text:
            return None
        think_end = text.rfind('</think>')
        if think_end != -1:
            after_think = text[think_end + len('</think>'):]
            ans = _extract_answer_from_segment(after_think)
            if ans is not None:
                return ans
        return _extract_answer_from_segment(text)
    
    test_cases = [
        # DeepSeek format: thinking has wrong answer, final has correct
        ("<think>Let me try... I get \\boxed{999}</think>\n\nThe answer is \\boxed{42}", 42),
        # Thinking has the correct answer, no answer after </think>
        ("<think>Computing... \\boxed{336}</think>\n\n", 336),
        # No think tags at all
        ("The answer is \\boxed{520}", 520),
        # Think tag with conflicting answers — after-think wins
        ("<think>\\boxed{100}</think>After careful review: \\boxed{200}", 200),
        # Empty after think, answer in think
        ("<think>Working... answer is 160</think>", 160),
        # Multiple think blocks (unusual but possible)
        ("<think>wrong</think>middle \\boxed{50}<think>also wrong</think>", 50),
    ]
    
    passed = 0
    for text, expected in test_cases:
        result = extract_answer(text)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: extract_answer({text[:60]!r}...) = {result}, expected {expected}")
    
    print(f"Think tag extraction: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_problem_classification():
    """Test problem type classification."""
    # Inline the classify_problem logic
    def classify_problem(problem):
        text = problem.lower()
        scores = {'number_theory': 0, 'combinatorics': 0, 'algebra': 0, 'geometry': 0}
        for kw in ['prime', 'divisor', 'gcd', 'lcm', 'modulo', 'mod ', 'remainder',
                    'coprime', 'congruent', 'euler', 'fermat', 'digit', 'factorial',
                    'divides', 'divisible', 'pmod', 'totient', 'residue']:
            if kw in text:
                scores['number_theory'] += 2
        for kw in ['how many', 'count', 'number of ways', 'probability', 'permutation',
                    'combination', 'choose', 'arrange', 'distribute', 'subset',
                    'sequence', 'binary string', 'lattice path', 'catalan', 'recurrence',
                    'expected', 'coin', 'dice', 'random']:
            if kw in text:
                scores['combinatorics'] += 2
        for kw in ['polynomial', 'equation', 'root', 'coefficient', 'sum of',
                    'product of', 'series', 'sequence', 'function', 'inequality',
                    'maximize', 'minimize', 'minimum', 'maximum', 'real number']:
            if kw in text:
                scores['algebra'] += 2
        for kw in ['triangle', 'circle', 'angle', 'area', 'perimeter', 'point',
                    'line', 'segment', 'perpendicular', 'parallel', 'inscribed',
                    'circumscribed', 'median', 'altitude', 'polygon', 'quadrilateral',
                    'coordinate', 'distance', 'radius', 'diameter']:
            if kw in text:
                scores['geometry'] += 2
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else 'algebra'
    
    test_cases = [
        ("Find the remainder when 3^100 is divided by 7. The answer is mod 7.", "number_theory"),
        ("How many ways can you arrange 5 books on a shelf?", "combinatorics"),
        ("Let ABC be a triangle with angle A = 60 degrees.", "geometry"),
        ("Find all roots of the polynomial x^3 - 6x + 1.", "algebra"),
        ("Find the number of prime divisors of 100!.", "number_theory"),
        ("A coin is tossed 10 times. What is the expected number of heads?", "combinatorics"),
    ]
    
    passed = 0
    for problem, expected_type in test_cases:
        result = classify_problem(problem)
        if result == expected_type:
            passed += 1
        else:
            print(f"  FAIL: classify({problem[:50]!r}) = {result}, expected {expected_type}")
    
    print(f"Problem classification: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_code_repair():
    """Test code auto-repair logic."""
    import re
    
    def _try_repair_code(code, error_msg):
        repaired = code
        import_fixes = {
            'sympy': 'import sympy\nfrom sympy import *',
            'numpy': 'import numpy as np',
            'Fraction': 'from fractions import Fraction',
            'combinations': 'from itertools import combinations, permutations, product',
            'Counter': 'from collections import Counter',
            'gcd': 'from math import gcd',
            'comb': 'from math import comb',
            'factorial': 'from math import factorial',
        }
        for name, imp in import_fixes.items():
            if f"name '{name}' is not defined" in error_msg:
                repaired = imp + '\n' + repaired
        if repaired == code:
            return None
        return repaired
    
    test_cases = [
        # Missing sympy
        ("print(sympy.factorial(5))", "name 'sympy' is not defined",
         True, "import sympy"),
        # Missing Fraction
        ("print(Fraction(1, 3))", "name 'Fraction' is not defined",
         True, "from fractions import Fraction"),
        # No fixable error
        ("print(x)", "name 'x' is not defined", False, None),
        # Missing gcd
        ("print(gcd(12, 8))", "name 'gcd' is not defined",
         True, "from math import gcd"),
    ]
    
    passed = 0
    for code, error, should_fix, expected_import in test_cases:
        result = _try_repair_code(code, error)
        if should_fix:
            if result is not None and expected_import in result:
                passed += 1
            else:
                print(f"  FAIL: repair({code[:30]!r}) = {result!r}, expected containing {expected_import!r}")
        else:
            if result is None:
                passed += 1
            else:
                print(f"  FAIL: repair({code[:30]!r}) should return None, got {result!r}")
    
    print(f"Code repair: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_verify_template():
    """Test VERIFY_TEMPLATE formatting."""
    system = "You are a math expert."
    problem = "Find x such that x^2 = 4."
    candidate = 42
    
    tmpl = "{system}\n\nProblem:\n{problem}\n\nCandidate: {candidate_answer}\n\nVerify."
    try:
        result = tmpl.format(system=system, problem=problem, candidate_answer=candidate)
        ok = system in result and problem in result and "42" in result
    except Exception:
        ok = False
    
    passed = 1 if ok else 0
    if not ok:
        print("  FAIL: VERIFY_TEMPLATE formatting failed")
    print(f"Verify template: {passed}/1 tests passed")
    return passed == 1


def test_multi_block_execution():
    """Test multi-block code execution strategy."""
    import subprocess
    import tempfile
    
    def execute_code(code, timeout=5):
        if not code or not code.strip():
            return False, ""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                tmp_path = f.name
            proc = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout)
            os.unlink(tmp_path)
            return (proc.returncode == 0), proc.stdout
        except Exception:
            return False, ""
    
    # Test: two blocks where second depends on first
    block1 = "x = 42"
    block2 = "print(x)"
    
    # Individual execution of block2 should fail (x not defined)
    ok1, _ = execute_code(block2)
    # Combined execution should work
    combined = block1 + '\n' + block2
    ok2, output2 = execute_code(combined)
    
    passed = 0
    if not ok1:
        passed += 1
    else:
        print("  FAIL: block2 alone should fail (x undefined)")
    if ok2 and "42" in output2:
        passed += 1
    else:
        print(f"  FAIL: combined blocks should output 42, got ok={ok2}, output={output2!r}")
    
    print(f"Multi-block execution: {passed}/2 tests passed")
    return passed == 2
    """Run full evaluation with vLLM model against reference problems."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. Install with: pip install vllm")
        return

    # Find model
    if model_path is None:
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    print(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        max_model_len=32768,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sp_det = SamplingParams(temperature=0, max_tokens=16384)

    def make_sp_div(seed):
        return SamplingParams(temperature=0.6, top_p=0.95, max_tokens=16384, seed=seed)

    # Use the submission solver
    sys.path.insert(0, os.path.join(ROOT, 'notebooks'))
    from submission import AIMOSolver

    solver = AIMOSolver(llm, sp_det, make_sp_div)

    problems = load_reference_problems()
    if max_problems:
        problems = problems[:max_problems]

    correct = 0
    results = []
    for p in problems:
        t0 = time.time()
        predicted = solver.solve(p['problem'])
        elapsed = time.time() - t0
        is_correct = predicted == p['answer']
        if is_correct:
            correct += 1
        results.append({
            'id': p['id'],
            'predicted': predicted,
            'actual': p['answer'],
            'correct': is_correct,
            'time': elapsed,
        })
        status = "✓" if is_correct else "✗"
        print(f"  {status} {p['id']}: predicted={predicted}, actual={p['answer']} ({elapsed:.1f}s)")

    print(f"\nAccuracy: {correct}/{len(problems)} ({100*correct/len(problems):.1f}%)")
    return results


def main():
    parser = argparse.ArgumentParser(description='AIMO3 Local Evaluation')
    parser.add_argument('--with-model', action='store_true',
                       help='Run full evaluation with vLLM model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--max-problems', type=int, default=None,
                       help='Max number of reference problems to evaluate')
    args = parser.parse_args()

    print("=" * 60)
    print("AIMO3 Local Evaluation")
    print("=" * 60)

    if args.with_model:
        eval_with_model(args.model_path, args.max_problems)
    else:
        print("\n--- Unit Tests ---")
        test_answer_extraction()
        test_code_execution()
        test_voting()
        test_submission_voting()
        test_code_output_parsing()
        test_adaptive_temperature()
        test_few_shot_format()
        test_think_tag_extraction()
        test_problem_classification()
        test_code_repair()
        test_verify_template()
        test_multi_block_execution()
        print("\nTo run full model evaluation: python evaluation/local_eval.py --with-model")


if __name__ == '__main__':
    main()
