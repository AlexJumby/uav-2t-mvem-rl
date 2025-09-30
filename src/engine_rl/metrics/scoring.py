import numpy as np


def iae(y: np.ndarray, ref: float = 1.0) -> float:
    return float(np.sum(np.abs(y - ref)))


def overshoot(y: np.ndarray, ref: float = 1.0) -> float:
    return float(max(0.0, np.max(y) - ref))


def violations(y: np.ndarray, lo: float = 0.9, hi: float = 1.1) -> int:
    return int(np.sum((y < lo) | (y > hi)))
