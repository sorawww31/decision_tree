import numpy as np
def split_data(X: np.array, y:np.array, feature_index:np.int8, threshold:np.float16) -> Tuple(Tuple(np.array, np.float16), Tuple(np.array, np.float16)):
    """
    行列Xのfeature_indexによる特定の列において、thresholdでXを左右に分けるコード
    Args:
        X:(データ数, 特徴数), 
    Returns
        X_left: Xのfeature_index列にて、thresholdよりも小さい値の行列
        X_right:Xのfeature_index列にて、thresholdよりも大きい値の行列

    X = [[ 0  1  2  3  4]
        [ 5  6  7  8  9]
        [10 11 12 13 14]
        [15 16 17 18 19]]
    このとき、X.shape = (4行, 5列)
    """
    mask = X[:, feature_index] < threshold
    X_left = X[mask]
    X_right = X[~mask]
    y_left = y[mask]
    y_right = y[~mask]
    return (X_left, y_left), (X_right, y_right)