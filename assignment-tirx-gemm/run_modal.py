"""
TIRX GEMM Assignment — Modal Cloud B200 Runner

Setup (one-time):
    pip install modal
    modal setup

Usage:
    # Run all tests
    modal run run_modal.py
    # Run a specific step
    modal run run_modal.py --step 1
    # Run multiple specific steps
    modal run run_modal.py --step 1,3,5

    # Inspect generated CUDA source
    modal run run_modal.py --inspect 7
    modal run run_modal.py --inspect 10 --size 2048
"""

import sys

import modal

app = modal.App("tirx-gemm-assignment")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends libx11-6 && rm -rf /var/lib/apt/lists/*",
    )
    .run_commands(
        "python -m pip install --pre -U -f https://mlc.ai/wheels 'mlc-ai-tirx-cu130==0.0.1b2'",
    )
    .run_commands(
        "python -m pip install --force-reinstall 'apache-tvm-ffi==0.1.9'",
    )
    .pip_install(
        "torch",
        "pytest",
        "numpy",
    )
    .add_local_dir(".", remote_path="/workspace", ignore=[".git"])
)


@app.function(image=image, gpu="B200:1", timeout=3600)
def run_tests(test_pattern: str) -> int:
    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", test_pattern, "-vs"],
        cwd="/workspace",
    )
    return result.returncode


@app.function(image=image, gpu="B200:1", timeout=600)
def run_inspect(step: int, size: int) -> str:
    import subprocess

    result = subprocess.run(
        ["python", "inspect_cuda.py", str(step), str(size)],
        cwd="/workspace",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"Error:\n{result.stderr}"
    return result.stdout


@app.local_entrypoint()
def main(step: str = "", inspect: int = 0, size: int = 1024):
    if inspect > 0:
        print(f"Inspecting step {inspect}, size {size}x{size}x{size}...", file=sys.stderr)
        src = run_inspect.remote(inspect, size)
        print(src)
        return

    if not step:
        print("Running all tests...")
        returncode = run_tests.remote("tests/")
        if returncode != 0:
            print(f"\nTests failed (exit code {returncode})")
            sys.exit(returncode)
        print("\nAll tests passed!")
        return

    try:
        steps = [int(s.strip()) for s in step.split(",")]
    except ValueError:
        print(f"Error: --step must be a number or list of numbers (got {step!r})")
        sys.exit(1)

    failed = []
    for s in steps:
        test_pattern = f"tests/test_step{s:02d}.py"
        print(f"Running step {s}: {test_pattern}")
        returncode = run_tests.remote(test_pattern)
        if returncode != 0:
            print(f"  Step {s} FAILED (exit code {returncode})")
            failed.append(s)
        else:
            print(f"  Step {s} passed.")

    if failed:
        print(f"\nFailed steps: {failed}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
