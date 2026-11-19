"""Classical Discrete Fourier Transform baseline."""
from __future__ import annotations

import numpy as np


def dft_matrix(N: int) -> np.ndarray:
    """Unitary DFT matrix with the same sign convention as the QFT:
    F[k, x] = exp(+2*pi*i*x*k/N) / sqrt(N)."""
    k = np.arange(N).reshape(-1, 1)
    x = np.arange(N).reshape(1, -1)
    return np.exp(2j * np.pi * k * x / N) / np.sqrt(N)


def classical_dft(signal: np.ndarray) -> np.ndarray:
    """Fast classical DFT (NumPy FFT).  ``ifft`` with norm='ortho' is exactly
    (1/sqrt(N)) * sum_x x[n] exp(+2*pi*i*n*k/N), i.e. the QFT convention."""
    return np.fft.ifft(signal, norm="ortho")


def reference_distribution(signal: np.ndarray) -> np.ndarray:
    """|DFT(x)|^2 - the ideal measurement distribution after a QFT."""
    p = np.abs(classical_dft(signal)) ** 2
    return p / p.sum()


def classical_cost(N: int) -> dict:
    """Arithmetic cost of the classical baselines (complex multiplications)."""
    n = int(np.log2(N))
    return {"direct_dft_mults": N * N, "radix2_fft_mults": (N // 2) * n}
