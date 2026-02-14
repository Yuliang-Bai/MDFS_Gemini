# src/utils/parallel.py
import os

def configure_for_multiprocessing(n_cores: int, inner_threads: int = 1) -> None:
    """
    外层用多进程时：限制每个进程内部的 OpenMP/BLAS 线程数，避免并行嵌套导致巨慢。
    注意：必须在 import numpy/scipy/torch/pandas 之前调用才最稳定（spawn 会在子进程重新 import）。
    """
    if n_cores and n_cores > 1:
        t = str(max(1, int(inner_threads)))
        os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
        os.environ.setdefault("OMP_NESTED", "FALSE")
        os.environ["OMP_NUM_THREADS"] = t
        os.environ["OPENBLAS_NUM_THREADS"] = t
        os.environ["MKL_NUM_THREADS"] = t
        os.environ["VECLIB_MAXIMUM_THREADS"] = t
        os.environ["NUMEXPR_NUM_THREADS"] = t
        os.environ.setdefault("KMP_WARNINGS", "off")

def worker_init() -> None:
    """
    每个子进程启动时执行：进一步锁住 PyTorch 线程池（mac 上非常关键）。
    """
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
