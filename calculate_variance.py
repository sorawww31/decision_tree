import numpy as np

def calculate_variance(y:np.array) -> np.float16:
    """
    回帰木の各ノードの不純度に関わる分散を計算する
    np.var(y)で解決しますが。
    配列　→　配列の分散
    Args:
        y: np.array()
            y.shape = (y_dim, )
    Return:
        variance: np.float32

    """
    variance = np.mean(
        np.mean(
            (y - y.mean())**2
        )
    )
    return variance
