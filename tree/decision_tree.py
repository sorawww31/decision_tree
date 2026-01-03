import numpy as np
import pandas as pd

from .core import find_best_split, split_data


class SimpleDecisionTree:
    """
    シンプルな決定木
    決定木はDecisionノードとLeafノードで構成される

    Decision Node:
        決定木の分岐を表すノード。どの列のどの閾値で左右に分割したらいいのかを表す
    Leaf Node:
        最終的なスコアを表すノード。分類タスクなら各クラスの確率が出るし、回帰なら予測値を表す
    """

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2, seed: int = 42):
        """
        Args:
            max_depth:
                木の深さを制限する。GBGTでは浅い木を沢山くっつけて（アンサンブル）することでスコアを出す
            min_samples_split:
                分けれる最小限のデータの数。1vs99のように、片方に偏りすぎないようにできる。
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree: dict | np.float16 | None = None
        self.seed: int = seed

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

        best_split = find_best_split(X, y, self.seed)
        (X_left, y_left), (X_right, y_right) = split_data(
            X, y, best_split["feat"], best_split["thr"]
        )
        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)

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
