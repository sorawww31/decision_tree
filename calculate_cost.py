import numpy as np
from calculate_variance import  calculate_variance
def calculate_cost(y_left:np.float16, y_right:np.float16) -> np.float16:
    """
    左右の目的変数の分散の重み付き平均スコア
    データ数が多いということは、それだけ多くのデータポイントがその分散の影響を受けるということなので、重みを大きくして評価する必要があります。
    Args:
        y_left: 左側の目的変数
        y_right: 右側の目的変数
            y.shape = (データ数, )


    """
    # 左の分散
    val_left, N_left = calculate_variance(y_left) , len(y_left)
    # 右の分散
    val_right, N_right = calculate_variance(y_right), len(y_right)

    return  (val_left*N_left + val_right*N_right) / (N_left + N_right)