import argparse
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from tree.decision_tree import SimpleDecisionTree
from tree.random_forest import RandomForestRegressor
from utils import object


def parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="decision_tree",
        choices=["decision_tree", "random_forest"],
    )
    parser.add_argument("--max_depth", "-d", type=int, default=3)
    # 分割に必要な最小の数
    parser.add_argument("--min_samples_split", "-s", type=int, default=2)
    # 葉を構成するのに必要な最小の数
    parser.add_argument("--min_samples_leaf", "-l", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.8)
    parser.add_argument("--objective", "-o", type=str, default="RSME")
    parser.add_argument("--seed", type=int, default=42)
    # RandomForest用の引数
    parser.add_argument(
        "--n_estimators",
        "-n",
        type=int,
        default=10,
        help="ランダムフォレストのツリー数",
    )
    parser.add_argument(
        "--max_features", type=float, default=0.8, help="使用する特徴量の割合"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="ブートストラップサンプリングのサンプル数",
    )
    args = parser.parse_args()
    return args


def _load_datasets(
    val_split: float = 0.8, debug: bool = False
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

    train_size = int(len(df) * val_split)
    return df[:train_size], df[train_size:]


def _create_model(args):
    """
    引数に基づいてモデルを作成する
    Returns:
        model: 自作モデル
        sklearn_model: sklearn比較用モデル
    """
    mode = args.mode.lower()

    if mode == "decision_tree":
        model = SimpleDecisionTree(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            seed=args.seed,
        )
        sklearn_model = DecisionTreeRegressor(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
        )
    elif mode == "random_forest":
        # max_samplesがNoneの場合、デフォルト値を設定
        max_samples = args.max_samples
        model = RandomForestRegressor(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            n_estimators=args.n_estimators,
            max_features=args.max_features,
            max_samples=max_samples,
            min_samples_leaf=args.min_samples_leaf,
            seed=args.seed,
        )
        sklearn_model = SklearnRandomForestRegressor(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            n_estimators=args.n_estimators,
            max_features=args.max_features,
            max_samples=max_samples,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
        )
    else:
        raise ValueError(f"{args.mode}は対応しておりません")

    return model, sklearn_model


def _get_objective_func(objective: str) -> Tuple[Callable, str]:
    """
    目的関数を取得する
    Returns:
        objective_func: 評価関数
        objective_name: 評価関数名
    """
    objective = objective.lower()
    if objective == "rsme":
        return object.root_mean_sqrt_error, "RSME"
    elif objective == "rsmle":
        return object.root_mean_sqrt_log_error, "RSMLE"
    else:
        raise ValueError(f"{objective}は対応していません")


def _evaluate_and_print(
    model_name: str,
    model,
    objective_func: Callable,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_tr: np.ndarray,
    y_pred_train: np.ndarray,
    means: np.ndarray,
):
    """
    モデルの評価結果を出力する
    """
    score = objective_func(y=y_val, y_pred=y_pred)
    train_score = objective_func(y=y_tr, y_pred=y_pred_train)
    baseline = objective_func(y=y_val, y_pred=means)

    print(f"[{model_name}] score: {score}")
    print(f"[{model_name}] train_score: {train_score}, means_score: {baseline}")
    print(model.feature_importances_)


def main():
    args = parsers()
    tr_df, val_df = _load_datasets(val_split=args.val_split, debug=args.debug)

    # max_samplesのデフォルト値を設定（DataFrameのサイズに基づく）
    if args.max_samples is None:
        args.max_samples = len(tr_df)

    print(f"Use {args.mode.upper()} algorithm")

    # モデル作成
    model, sklearn_model = _create_model(args)

    # データ準備
    X_tr, y_tr = tr_df.drop(columns=["target"]), tr_df["target"]
    X_val, y_val = val_df.drop(columns=["target"]), val_df["target"]
    means = np.array([np.mean(y_val)] * len(y_val))

    # 自作モデルの学習と推論
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    y_pred_train = model.predict(X_tr)

    # sklearnモデルの学習と推論
    sklearn_model.fit(X_tr, y_tr)
    sklearn_pred = sklearn_model.predict(X_val)
    sklearn_pred_train = sklearn_model.predict(X_tr)

    # 評価関数の取得
    objective_func, objective_name = _get_objective_func(args.objective)
    print(f"==={objective_name}===")

    # 評価結果の出力
    _evaluate_and_print(
        model_name=args.mode.upper(),
        model=model,
        objective_func=objective_func,
        y_val=y_val,
        y_pred=y_pred,
        y_tr=y_tr,
        y_pred_train=y_pred_train,
        means=means,
    )
    _evaluate_and_print(
        model_name="sklearn",
        model=sklearn_model,
        objective_func=objective_func,
        y_val=y_val,
        y_pred=sklearn_pred,
        y_tr=y_tr,
        y_pred_train=sklearn_pred_train,
        means=means,
    )


if __name__ == "__main__":
    main()
