"""
AIMO3 Utility Setup Notebook — Notebook 1/2
============================================
Pre-installs vLLM and dependencies for the competition submission notebook.

Usage on Kaggle:
  1. Create a new notebook with GPU H100 accelerator.
  2. Enable internet access.
  3. Paste/upload this script as the notebook code.
  4. Run it. The output directory will contain pre-built wheels.
  5. Save the notebook version.
  6. Use the saved version as a data source for the submission notebook (2/2).

The submission notebook (2/2) will install packages from this notebook's
output directory in offline mode (internet disabled during competition).
"""

import os
import subprocess
import sys

OUTPUT_DIR = "/kaggle/working"


def run(cmd: str) -> int:
    """Run a shell command, printing output."""
    print(f"\n>>> {cmd}")
    return subprocess.call(cmd, shell=True)


def main():
    print("=" * 60)
    print("AIMO3 Utility Setup — Installing Dependencies")
    print("=" * 60)

    # Upgrade pip
    run(f"{sys.executable} -m pip install --upgrade pip")

    # Install vLLM and core dependencies
    # vLLM will pull in torch, transformers, etc.
    packages = [
        "vllm",
        "sympy",
        "polars",
    ]

    for pkg in packages:
        run(f"{sys.executable} -m pip install {pkg}")

    # Download wheels to output directory for offline use by submission notebook
    wheels_dir = os.path.join(OUTPUT_DIR, "wheels")
    os.makedirs(wheels_dir, exist_ok=True)

    run(f"{sys.executable} -m pip download vllm sympy -d {wheels_dir}")

    # Verify installations
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    checks = [
        ("vllm", "import vllm; print(f'vLLM {vllm.__version__}')"),
        ("sympy", "import sympy; print(f'SymPy {sympy.__version__}')"),
        ("torch", "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"),
        ("polars", "import polars; print(f'Polars {polars.__version__}')"),
    ]

    for name, check in checks:
        try:
            exec(check)
        except Exception as e:
            print(f"WARNING: {name} check failed: {e}")

    # List what's in the output
    print(f"\nOutput directory contents ({OUTPUT_DIR}):")
    for item in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(path):
            count = len(os.listdir(path))
            print(f"  {item}/ ({count} files)")
        else:
            size = os.path.getsize(path)
            print(f"  {item} ({size:,} bytes)")

    print("\n✓ Utility setup complete. Save this notebook and use it as a data source.")


if __name__ == "__main__":
    main()
