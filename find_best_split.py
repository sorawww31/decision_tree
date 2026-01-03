import numpy as np
from split_data import split_data

def find_best_split(X, y):
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
    
    best_split = {
        'val': float('inf'),
        'feat': None,
        'thr': None
    }

    for feat in range(n_feats):
            # 修正1: np.unique() を使用
            thresholds = np.unique(X[:, feat])
            
            for thr in thresholds:
                # 前に作った関数を利用
                X_left, y_left, X_right, y_right = split_data(X, y, feat, thr)
                
                # 片方が空っぽならスキップ（分割になっていないため）
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                    
                val = calculate_cost(y_left, y_right)
                
                if val < best_split['val']:
                    best_split['val'] = val
                    best_split['feat'] = feat
                    best_split['thr'] = thr
                
    return best_split
