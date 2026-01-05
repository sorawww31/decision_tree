import numpy as np

from .decision_tree import SimpleDecisionTree


class GradientBoostingRegressor:
    """
    勾配ブースティング決定木は、木を直列につなぎ、残差を予測するモデルの総称
    現在の予測値Fm(x)とし、ターゲットの差をr = y - Fm(x)とする。
    次の木hm+1は、このrを予測値とする。

    次の予測値は、Fm+1 = Fm + \iter * hm+1(x)となる。

    予測値をどんどん正解にちかづけていくイメージ
    """

    def __init__(
        self,
        max_depth: int,
        learning_rate: float,
        min_samples_split: int,
        min_samples_leaf: int,
        n_estimators: int,
        max_features: float,
        max_samples: int = 1000,
        seed: int = 42,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.seed = seed

        self.f0 = None
        self.trees: list = []

    def fit(self, X: np.ndarray, y: np.ndarray):
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

    def predict(self, X: np.ndarray):
        return self.f0 + self.learning_rate * np.sum(
            [tree.predict(X) for tree in self.trees], axis=0
        )
