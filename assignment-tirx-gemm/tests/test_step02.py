import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import prepare_data, compile_and_run, verify, check_timing
from gemm_kernels import hgemm_v2


@pytest.mark.parametrize("K", [64, 512, 1024, 4096])
def test_k_loop(K):
    M, N = 128, 128
    kernel = hgemm_v2(M, N, K)
    A, B, C = prepare_data(M, N, K)
    C_tir = compile_and_run(kernel, A, B, C)
    verify(C_tir, A, B)
    check_timing(kernel, step=2, M=M, N=N, K=K)
