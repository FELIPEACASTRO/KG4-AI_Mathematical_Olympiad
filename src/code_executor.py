"""
Safe code execution sandbox for verifying mathematical computations.
Executes Python/SymPy code extracted from LLM outputs in a restricted environment.
"""

import multiprocessing
import os
import signal
import sys
import traceback
from typing import Optional


# Allowed modules for safe execution
ALLOWED_MODULES = {
    'math', 'cmath', 'fractions', 'decimal', 'itertools',
    'functools', 'collections', 'operator', 'sympy', 'numpy',
    're', 'random',
}

# Blocked builtins for security
BLOCKED_BUILTINS = {
    'exec', 'eval', 'compile', '__import__', 'open', 'input',
    'breakpoint', 'exit', 'quit',
}


def _execute_in_process(code: str, result_queue: multiprocessing.Queue, timeout: int):
    """Execute code in a child process and put result in queue."""
    import io
    import contextlib

    safe_builtins = {k: v for k, v in __builtins__.items() if k not in BLOCKED_BUILTINS} if isinstance(__builtins__, dict) else {k: getattr(__builtins__, k) for k in dir(__builtins__) if k not in BLOCKED_BUILTINS}

    # Build restricted globals
    restricted_globals = {
        '__builtins__': safe_builtins,
        '__name__': '__main__',
    }

    # Pre-import allowed modules
    for mod_name in ['math', 'sympy', 'itertools', 'functools', 'collections', 'fractions', 'decimal']:
        try:
            restricted_globals[mod_name] = __import__(mod_name)
        except ImportError:
            pass

    # Capture stdout
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, restricted_globals)  # noqa: S102
        result_queue.put(('success', output_buffer.getvalue()))
    except Exception as e:
        result_queue.put(('error', f'{type(e).__name__}: {e}'))


def execute_code(code: str, timeout: int = 30) -> tuple[bool, str]:
    """
    Execute Python code in a sandboxed subprocess with timeout.
    
    Returns:
        (success: bool, output: str) - The output or error message.
    """
    if not code or not code.strip():
        return False, "Empty code"

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execute_in_process,
        args=(code, result_queue, timeout),
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        return False, f"Code execution timed out after {timeout}s"

    if not result_queue.empty():
        status, output = result_queue.get_nowait()
        return status == 'success', output

    return False, "No output from code execution"


def execute_code_simple(code: str, timeout: int = 30) -> tuple[bool, str]:
    """
    Code execution using subprocess for reliable timeout handling.
    Works on both Windows and Linux, and properly kills runaway code.
    """
    import subprocess
    import tempfile

    if not code or not code.strip():
        return False, "Empty code"

    # Write code to a temp file and run in subprocess for isolation
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                          encoding='utf-8') as f:
            # Pre-import common math libraries
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
            return False, proc.stderr[-500:] if proc.stderr else "Non-zero exit"

    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False, f"Code execution timed out after {timeout}s"
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'
