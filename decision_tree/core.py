import random
from typing import Tuple

import numpy as np


def calculate_cost(y_left: np.ndarray, y_right: np.ndarray) -> np.float16:
    """
    左右の目的変数の分散の重み付き平均スコア
    データ数が多いということは、それだけ多くのデータポイントがその分散の影響を受けるということなので、重みを大きくして評価する必要があります。
    Args:
        y_left: 左側の目的変数
        y_right: 右側の目的変数
            y.shape = (データ数, )


    """
    # 左の分散
    val_left, N_left = calculate_variance(y_left), len(y_left)
    # 右の分散
    val_right, N_right = calculate_variance(y_right), len(y_right)

    return (val_left * N_left + val_right * N_right) / (N_left + N_right)


def find_best_split(X, y, seed=42):
    """
    Args:
        X: 特徴行列 (データ数, 特徴の数)
        y: 目的変数 (データの数, )
    Returns:
        returns:
            val: 最小の分散
            feat: 最小の分散をつくるための特徴インデックス
            thr: 最小の分散をつくるための閾値
    すべてのXの列(feat)に対し、X[:, feat]のユニークな値を全部探索し、分割の分散を最も小さくする分け方を探索する。

    feat, thrの分け方を工夫することで、XGBoost, LightGBMと進化していく。
    """
    n_samples, n_feats = X.shape

    best_split = {"val": float("inf"), "feat": None, "thr": None}
    bests = []
    for feat in range(n_feats):
        # np.unique() を使用
        thresholds = np.unique(X[:, feat])

        for thr in thresholds:
            # 前に作った関数を利用
            (X_left, y_left), (X_right, y_right) = split_data(X, y, feat, thr)

            # 片方が空っぽならスキップ（分割になっていないため）
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            val = calculate_cost(y_left, y_right)

            if val <= best_split["val"]:
                best_split["val"] = val
                best_split["feat"] = feat
                best_split["thr"] = thr
                bests.append(best_split)
    if seed is not None:
        random.seed(seed)
    return random.choice(bests)


def calculate_variance(y: np.ndarray) -> np.float16:
    """
    回帰木の各ノードの不純度に関わる分散を計算する
    np.var(y)で解決しますが。
    配列　→　配列の分散
    Args:
        y: np.array()
            y.shape = (y_dim, )
    Return:
        variance: np.float16

    """
    variance = np.mean(np.mean((y - y.mean()) ** 2))
    return variance


def split_data(
    X: np.ndarray, y: np.ndarray, feature_index: np.int8, threshold: np.float16
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
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
