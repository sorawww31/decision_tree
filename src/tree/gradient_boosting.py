import numpy as np
import pandas as pd

from .decision_tree import SimpleDecisionTree
from .random_forest import RandomForestRegressor


class GradientBoostingRegressor(RandomForestRegressor):
    """
    勾配ブースティング決定木は、木を直列につなぎ、残差を予測するモデルの総称
    現在の予測値Fm(x)とし、ターゲットの差をr = y - Fm(x)とする。
    次の木hm+1は、このrを予測値とする。

    次の予測値は、Fm+1 = Fm + \iter * hm+1(x)となる。

    予測値をどんどん正解にちかづけていくイメージ
    GBDTは、最初はバイアスが高い状態（大雑把な予測）からスタートして、徐々にバイアスを下げていく手法

    バイアスが高い: モデルが単純すぎて、データの傾向を捉えられない（学習不足）。
    バリアンスが高い: モデルが複雑すぎて、データごとの細かいノイズにまで過剰に反応してしまう（過学習）。
    """

    def __init__(
        self,
        max_depth: int,
        learning_rate: float,
        min_samples_split: int,
        min_samples_leaf: int,
        n_estimators: int,
        max_samples: int = 1000,
        seed: int = 42,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            n_estimators=n_estimators,
        )

        self.learning_rate = learning_rate
        self.f0 = None
        self.trees: list = []

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series):
        X = np.array(X)
        y = np.array(y)
        self.f0 = np.mean(y)
        f = self.f0
        r = y - self.f0
        for m in range(1, self.n_estimators + 1):
            tree = SimpleDecisionTree(
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                max_features=1.0,
                seed=self.seed,
            )
            tree.fit(X, r)
            h = tree.predict(X)

            f = f + self.learning_rate * h

            r = y - f

            self.trees.append(tree)
        self.calculate_gain()

    def predict(self, X: np.ndarray | pd.DataFrame):
        X = np.array(X)
        return self.f0 + self.learning_rate * np.sum(
            [tree.predict(X) for tree in self.trees], axis=0
        )
