import os
import pytest


def pytest_configure(config):
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpu_id = min(lines, key=lambda l: int(l.split(",")[1])).split(",")[0].strip()
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
