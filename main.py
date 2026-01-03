import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

from decision_tree.tree import SimpleDecisionTree
from utils import object


def parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", "-m", type=str, default="decision_tree")
    parser.add_argument("--max_depth", "-d", type=int, default=3)
    parser.add_argument("--min_samples_split", "--s", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.8)
    parser.add_argument("--objective", "-o", type=str, default="RSME")
    args = parser.parse_args()
    return args


def _load_datasets(
    val_split: float = 0.8, debug=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    sklearnのデータセットを読み込み、訓練データと検証データに分ける関数
    Args:
        val_split: trainとvalをどの割合で分けるか。0.8の場合、全体の80%がtrainとなり、20%がvalとなる
    Returns:
        tr_data, val_data: 分けられたtrainとval
    """
    dataset = load_diabetes()
    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    target = pd.DataFrame(dataset.target, columns=["target"])
    df = pd.concat([data, target], axis=1)
    if debug:
        df = df[:100]

    train_size = int(len(df) * 0.8)
    return df[:train_size], df[train_size:]


def main():
    args = parsers()
    tr_df, val_df = _load_datasets(val_split=args.val_split, debug=args.debug)

    print(f"Use {args.mode.upper()} algorithm ")
    if args.mode.lower() == "decision_tree":
        model = SimpleDecisionTree(
            max_depth=args.max_depth, min_samples_split=args.min_samples_split
        )
        # sklearn DecisionTreeRegressor comparison
        sklearn_model = DecisionTreeRegressor(
            max_depth=args.max_depth, min_samples_split=args.min_samples_split
        )
    else:
        raise ValueError(f"{args.mode}は対応しておりません")

    X_tr, y_tr = tr_df.drop(columns=["target"]), tr_df["target"]
    X_val, y_val = val_df.drop(columns=["target"]), val_df["target"]

    # 推論
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    y_pred_train = model.predict(X_tr)
    means = np.array([np.mean(y_val)] * (len(y_val)))

    sklearn_model.fit(X_tr, y_tr)
    sklearn_pred = sklearn_model.predict(X_val)
    sklearn_pred_train = sklearn_model.predict(X_tr)

    if args.objective.lower() == "rsme":
        print("===RSME===")
        score = object.root_mean_sqrt_error(y=y_val, y_pred=y_pred)
        train_score = object.root_mean_sqrt_error(y=y_tr, y_pred=y_pred_train)
        baseline = object.root_mean_sqrt_error(y=y_val, y_pred=means)

        sklearn_score = object.root_mean_sqrt_error(y=y_val, y_pred=sklearn_pred)
        sklearn_train_score = object.root_mean_sqrt_error(
            y=y_tr, y_pred=sklearn_pred_train
        )
        sklearn_baseline = object.root_mean_sqrt_error(y=y_val, y_pred=means)

    elif args.objective.lower() == "rsmle":
        print("===RSMLE===")
        score = object.root_mean_sqrt_log_error(y=y_val, y_pred=y_pred)
        train_score = object.root_mean_sqrt_log_error(y=y_tr, y_pred=y_pred_train)
        baseline = object.root_mean_sqrt_log_error(y=y_val, y_pred=means)

        sklearn_score = object.root_mean_sqrt_log_error(y=y_val, y_pred=sklearn_pred)
        sklearn_train_score = object.root_mean_sqrt_log_error(
            y=y_tr, y_pred=sklearn_pred_train
        )
        sklearn_baseline = object.root_mean_sqrt_log_error(y=y_val, y_pred=means)

    else:
        raise ValueError(f"{args.objective}は対応していません")

    print(f"[Simple] score: {score}")
    print(f"[Simple] train_score: {train_score}, means_score: {baseline}")
    print(f"[sklearn] score: {sklearn_score}")
    print(
        f"[sklearn] train_score: {sklearn_train_score}, means_score: {sklearn_baseline}"
    )


if __name__ == "__main__":
    main()
