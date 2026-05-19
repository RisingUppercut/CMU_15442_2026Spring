import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import prepare_data, compile_and_run, verify, check_timing
from gemm_kernels import hgemm_v3


@pytest.mark.parametrize("size", [256, 512, 1024, 2048])
def test_spatial_tiling(size):
    M, N, K = size, size, size
    kernel = hgemm_v3(M, N, K)
    A, B, C = prepare_data(M, N, K)
    C_tir = compile_and_run(kernel, A, B, C)
    verify(C_tir, A, B)
    check_timing(kernel, step=3, M=M, N=N, K=K)
