import random
from collections import defaultdict

import numpy as np
import pandas as pd

from .core import calculate_cost, calculate_variance, split_data


class SimpleDecisionTree:
    """
    シンプルな決定木
    決定木はDecisionノードとLeafノードで構成される

    Decision Node:
        決定木の分岐を表すノード。どの列のどの閾値で左右に分割したらいいのかを表す
    Leaf Node:
        最終的なスコアを表すノード。分類タスクなら各クラスの確率が出るし、回帰なら予測値を表す
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 5,
        max_features: float = 0.8,
        seed: int = 42,
    ):
        """
        Args:
            max_depth:
                木の深さを制限する。GBGTでは浅い木を沢山くっつけて（アンサンブル）することでスコアを出す
            min_samples_split:
                分けれる最小限のデータの数。1vs99のように、片方に偏りすぎないようにできる。
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree: dict | np.float16 | None = None
        self.max_features = max_features
        self.seed: int = seed
        self.rng = random.Random(seed)
        self.gain: dict = defaultdict(lambda: 0)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series):
        X = np.array(X)
        y = np.array(y)
        self.tree = self._grow_tree(X, y, depth=0)

    def predict(self, X: np.ndarray | pd.DataFrame):
        X = np.array(X)
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _grow_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> dict | np.float16:
        """
        再帰的に木を成長させる関数
        Q: なぜ再帰なのか？？
        A:  左右に分けた木を左右に分けたとき、新たに左右にできたのもまた、木構造であるから。
            具体的には左右に分けたX_left, X_rightに対して更に木を成長させる必要があるから。

        Args:
            X, y: 行列データXと目的変数y
        Returns:
            tree: dict
                feature_index: np.int8, 分割に使う列
                threshold, np.float16 分割の閾値
                left: 左の子ツリー
                right: 右の子ツリー
        """
        # 深さが上限を超えた OR データ数が少なすぎる 場合...
        # 一番最初にやることで工数削減
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            # ｙの平均値を返す
            # 例えばRossがMAEならmedianだし、今回はMSEなのでmean。もちろん他にもあるよ
            return np.float16(np.mean(y))

        best_split = self.find_best_split(X, y)
        if not best_split:
            return np.float16(np.mean(y))
        (X_left, y_left), (X_right, y_right) = split_data(
            X, y, best_split["feat"], best_split["thr"]
        )
        # 両方の子が、ちゃんとmin_samples_leaf個以上持ってるかどうかを判定する
        if (
            len(y_left) >= self.min_samples_leaf
            and len(y_right) >= self.min_samples_leaf
        ):
            left_subtree = self._grow_tree(X_left, y_left, depth + 1)
            right_subtree = self._grow_tree(X_right, y_right, depth + 1)
        else:
            return np.float16(np.mean(y))
        # これがノード, left, rightは子のノード
        return {
            "feature_index": best_split["feat"],
            "threshold": best_split["thr"],
            "left": left_subtree,  # 再帰の結果
            "right": right_subtree,  # 再帰の結果
        }

    def _predict_tree(self, x: np.ndarray, tree_node: dict | np.float16 | None):
        if not isinstance(tree_node, dict):
            return tree_node
        elif x[tree_node["feature_index"]] < tree_node["threshold"]:
            return self._predict_tree(x, tree_node["left"])
        else:
            return self._predict_tree(x, tree_node["right"])

    def find_best_split(self, X, y):
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

        best_val = float("inf")
        bests = []  # 同率の最高スコアの分け方を格納するリスト
        selected_feats = self.rng.sample(
            range(n_feats), k=int(n_feats * self.max_features)
        )
        base_val = calculate_variance(y)

        for feat in selected_feats:
            # np.unique() を使用
            thresholds = np.unique(X[:, feat])

            for thr in thresholds:
                # 前に作った関数を利用
                (X_left, y_left), (X_right, y_right) = split_data(X, y, feat, thr)

                # 片方が空っぽならスキップ（分割になっていないため）
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                val = calculate_cost(y_left, y_right)

                if val < best_val:
                    # より良いスコアが見つかった場合、リストをリセット
                    best_val = val
                    bests = [{"val": val, "feat": feat, "thr": thr}]
                elif val == best_val:
                    # 同率の場合、リストに追加
                    bests.append({"val": val, "feat": feat, "thr": thr})

        # 同率の候補からランダムに1つを選択（最初に作成したrngを再利用）
        if bests:
            best_split = self.rng.choice(bests)
            self.gain[best_split["feat"]] += base_val - best_split["val"]
            return best_split
        else:
            # 分割が見つからなかった場合（通常は起こらない）
            return None
