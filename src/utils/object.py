import numpy as np
import pandas as pd


def root_mean_sqrt_error(y: np.ndarray | pd.Series, y_pred: np.ndarray) -> np.float16:
    y = np.array(y)
    return np.sqrt(np.mean((y - y_pred) ** 2))


def root_mean_sqrt_log_error(
    y: np.ndarray | pd.Series, y_pred: np.ndarray
) -> np.float16:
    y = np.array(y)
    return np.sqrt(np.mean((np.log((1 + y_pred) / (1 + y))) ** 2))


def mean_sqrt_error(a: np.ndarray, b: np.ndarray | np.float16) -> np.float16:
    return np.float16(np.mean(a - b) ** 2)
