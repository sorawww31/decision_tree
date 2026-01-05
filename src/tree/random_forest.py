from collections import defaultdict

import numpy as np
import pandas as pd

from .decision_tree import SimpleDecisionTree


class RandomForestRegressor:
    """
    RandomForestの凄いところ
    1.アンサンブル
    2.データ、特徴の多様性
    3.特徴の重要度を調べることができる

    """

    def __init__(
        self,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        n_estimators: int,
        max_features: float = 1.0,
        max_samples: int = 1000,
        seed: int = 42,
    ):
        """
        ランダムフォレストの初期化
            ツリーのn_estimators個のツリーを作る
            すべてのツリーのハイパラ (max_depth, min_samples_split)は同じ
        ブートストラップサンプリング
            元のデータセットから、一部を取り除き、新たなデータセットとして扱う方法

            選ばれない: 運悪く（あるいは確率的に）、一度も選ばれないボールが出てくるため、データの多様性を生むことができる。

        Args:
            n_estimators: ランダムフォレスト木の数
            max_features: 全特徴のうち、どれくらいの割合をつかうのか
            max_samples: ブートストラップサンプリング
            seed: 乱数シード（再現性のため）

        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples if max_samples is not None else 1000
        self.seed = seed
        # 各ツリーに異なるシードを渡す（再現性と多様性を両立）
        self.trees = [
            SimpleDecisionTree(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                seed=seed + i,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            )
            for i in range(n_estimators)
        ]
        self.feature_importances_: dict = defaultdict(lambda: 0)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series):
        """
        Args: X(data_dim, features), y(data_dim, ) 特徴行列, 目的変数

        復元抽出（Bootstrapping)と非復元抽出（Subsampling)について
        Boost:
            箱から取り出して、また戻して、取り出して、戻して...
            選ばれる確率が互いに独立になる
        Sub:
            箱から取り出して、取り出して ... (戻さない)
            選ばれる確率は独立にならない

        Boostは、「もしもう一度、現実世界からデータを集め直したら、どんなデータのバラつきになるか？」
        という状況を数学的にうまく近似（シミュレーション）できる
        つまり、1回目のとりだし、2回目の取り出し...i回目...すべての取り出しが同じ確率分布からのサンプリングになる

        次元には気をつけて
        [array([1, 2, 3, 4]), array([2, 3, 4, 5]), array([3, 4, 5, 6])]
        >>> np.array(list)
        array([[1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6]])
        >>> np.array(list).shape
        (3, 4)
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_feat = X.shape
        rng = np.random.default_rng(self.seed)
        for i in range(self.n_estimators):
            # 復元抽出（シード付きrngで再現性を確保）
            idx_samples = rng.choice(np.array(range(n_samples)), size=self.max_samples)
            #
            # 各ツリーを各自訓練。
            self.trees[i].fit(X[idx_samples, :], y[idx_samples])

        self.calculate_gain()

    def predict(self, X: np.ndarray | pd.DataFrame):
        X = np.array(X)
        preds = [
            self.trees[i].predict(X) for i in range(len(self.trees))
        ]  # (n_estimators, n_samples)

        return np.mean(preds, axis=0)

    def calculate_gain(self):
        for i in range(self.n_estimators):
            for k, v in self.trees[i].gain.items():
                self.feature_importances_[k] += v
        self.feature_importances_ = dict(
            sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        )
