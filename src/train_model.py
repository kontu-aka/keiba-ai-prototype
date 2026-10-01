"""架空データを日付順に分割し、LightGBMモデルを学習・評価する。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# `python src/train_model.py` のような直接実行と、
# `from src.train_model import ...` によるパッケージ経由の読み込みの
# 両方でsrc.evaluationを解決できるよう、プロジェクトルートをsys.pathへ加える。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import summarize_betting_strategy, summarize_calibration

DEFAULT_DATA = PROJECT_ROOT / "data" / "sample_races.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RANDOM_SEED = 20260928

# 予想時点で利用可能な列だけを、意図的に列挙する。
# horse_numberは架空データ上で結果と意味のある関係を持たないため除外している。
# 将来の実データ版で馬番を恒久的に使わないという意味ではない。
NUMERIC_FEATURES = [
    "horse_age",
    "carried_weight",
    "distance",
    "recent_win_rate",
    "recent_top3_rate",
    "avg_finish_last5",
    "career_starts",
    "days_since_last_race",
    "surface_win_rate",
    "distance_top3_rate",
    "jockey_win_rate",
    "pre_race_odds",
]
CATEGORICAL_FEATURES = ["surface", "track_condition"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RESULT_COLUMNS = {"finish_position", "target_win"}

# v0.1で使用していたLightGBM設定（比較のために保持）。
LEGACY_LGBM_PARAMS: dict[str, object] = {
    "objective": "binary",
    "n_estimators": 220,
    "learning_rate": 0.035,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

# Phase Aでvalidデータのみを見て選んだ設定。データ量が10倍化したことで
# より深い正則化と早期終了（valid log lossで判定）が可能になったための変更。
# testの結果を見て調整したものではない。
NEW_LGBM_PARAMS: dict[str, object] = {
    "objective": "binary",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 60,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}
NEW_CONFIG_EARLY_STOPPING_ROUNDS = 50


def validate_data(data: pd.DataFrame) -> None:
    required = {
        "race_id",
        "race_date",
        "horse_id",
        "horse_name",
        "finish_position",
        "target_win",
        *FEATURE_COLUMNS,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"必須列がありません: {missing}")
    if RESULT_COLUMNS.intersection(FEATURE_COLUMNS):
        raise RuntimeError("結果列が特徴量に混入しています。")
    if not data.groupby("race_id")["target_win"].sum().eq(1).all():
        raise ValueError("各レースのtarget_win合計が1ではありません。")
    if data["race_date"].nunique() < 3:
        raise ValueError("train/valid/testの時系列3分割には3日以上のデータが必要です。")


def temporal_split_three(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ユニーク日付を2箇所で区切り、train < valid < test の時系列3分割にする。

    testはモデル調整に使用しない。設定選択（正則化など）の判断はvalidのみで行う。
    """
    unique_dates = np.sort(data["race_date"].unique())
    n_dates = len(unique_dates)
    train_end_index = min(max(int(n_dates * train_ratio), 1), n_dates - 2)
    valid_span = max(int(n_dates * valid_ratio), 1)
    valid_end_index = min(train_end_index + valid_span, n_dates - 1)
    valid_end_index = max(valid_end_index, train_end_index + 1)

    first_valid_date = unique_dates[train_end_index]
    first_test_date = unique_dates[valid_end_index]

    train = data.loc[data["race_date"] < first_valid_date].copy()
    valid = data.loc[
        (data["race_date"] >= first_valid_date) & (data["race_date"] < first_test_date)
    ].copy()
    test = data.loc[data["race_date"] >= first_test_date].copy()

    if train.empty or valid.empty or test.empty:
        raise RuntimeError("train/valid/testのいずれかが空です。")
    if not (train["race_date"].max() < valid["race_date"].min()):
        raise RuntimeError("trainとvalidの日付順序が不正です。")
    if not (valid["race_date"].max() < test["race_date"].min()):
        raise RuntimeError("validとtestの日付順序が不正です。")
    return train, valid, test


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def fit_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    model_config: str,
) -> tuple[ColumnTransformer, LGBMClassifier, int | None]:
    """指定設定でモデルを学習する。

    model_config="new" のときだけ、validを早期終了の判定に使う
    （testはこの時点で一切参照しない）。
    """
    if model_config not in ("legacy", "new"):
        raise ValueError(f"未知のmodel_configです: {model_config}")

    params = dict(LEGACY_LGBM_PARAMS if model_config == "legacy" else NEW_LGBM_PARAMS)

    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(train[FEATURE_COLUMNS])
    y_train = train["target_win"]

    model = LGBMClassifier(**params)
    fit_kwargs: dict[str, object] = {}
    if model_config == "new":
        x_valid = preprocessor.transform(valid[FEATURE_COLUMNS])
        y_valid = valid["target_win"]
        fit_kwargs = {
            "eval_set": [(x_valid, y_valid)],
            "eval_metric": "binary_logloss",
            "callbacks": [
                lgb.early_stopping(
                    stopping_rounds=NEW_CONFIG_EARLY_STOPPING_ROUNDS, verbose=False
                ),
                lgb.log_evaluation(0),
            ],
        }

    model.fit(x_train, y_train, **fit_kwargs)
    best_iteration = getattr(model, "best_iteration_", None)
    best_iteration = int(best_iteration) if best_iteration else None
    return preprocessor, model, best_iteration


def predict_win_probability(
    preprocessor: ColumnTransformer, model: LGBMClassifier, data: pd.DataFrame
) -> np.ndarray:
    return model.predict_proba(preprocessor.transform(data[FEATURE_COLUMNS]))[:, 1]


def _odds_baseline_metrics(test: pd.DataFrame) -> dict[str, float]:
    """1番人気（市場オッズ）基準線の指標。モデル設定には依存しない。"""
    market_implied_probability = 1.0 / test["pre_race_odds"]
    race_sum = market_implied_probability.groupby(test["race_id"]).transform("sum")
    normalized_probability = market_implied_probability / race_sum

    favorite_indices = test.groupby("race_id")["pre_race_odds"].idxmin()
    favorite_pick_accuracy = float(test.loc[favorite_indices, "target_win"].mean())

    return {
        "roc_auc": float(roc_auc_score(test["target_win"], market_implied_probability)),
        "log_loss": float(log_loss(test["target_win"], normalized_probability, labels=[0, 1])),
        "top_pick_accuracy": favorite_pick_accuracy,
    }


def compare_model_configs(data_path: Path, artifact_dir: Path) -> dict[str, object]:
    """同一の新データ・同一のtrain/valid/test分割で legacy / new 設定を比較する。

    legacy・newの両方をvalid/testで評価してmodel_config_comparison.jsonへ保存する。
    new設定の採否は「validのlog lossが小さい方」という機械的な基準のみで決定し、
    ここで保存するtestの数値は記録用であって、採否判定には使用していない。
    """
    data = pd.read_csv(data_path, parse_dates=["race_date"])
    validate_data(data)
    train, valid, test = temporal_split_three(data)

    baseline = _odds_baseline_metrics(test)

    config_results: dict[str, dict[str, object]] = {}
    for model_config in ("legacy", "new"):
        preprocessor, model, best_iteration = fit_model(train, valid, model_config)

        valid_probability = predict_win_probability(preprocessor, model, valid)
        test_probability = predict_win_probability(preprocessor, model, test)

        valid_roc_auc = float(roc_auc_score(valid["target_win"], valid_probability))
        valid_log_loss = float(log_loss(valid["target_win"], valid_probability, labels=[0, 1]))
        test_roc_auc = float(roc_auc_score(test["target_win"], test_probability))
        test_log_loss = float(log_loss(test["target_win"], test_probability, labels=[0, 1]))

        test_predictions = test[["race_id", "target_win"]].copy()
        test_predictions["probability"] = test_probability
        top_pick_indices = test_predictions.groupby("race_id")["probability"].idxmax()
        test_top_pick_accuracy = float(
            test_predictions.loc[top_pick_indices, "target_win"].mean()
        )

        config_results[model_config] = {
            "valid_roc_auc": valid_roc_auc,
            "valid_log_loss": valid_log_loss,
            "test_roc_auc": test_roc_auc,
            "test_log_loss": test_log_loss,
            "test_top_pick_accuracy": test_top_pick_accuracy,
            "best_iteration": best_iteration,
            "diff_vs_favorite_baseline": {
                "roc_auc": test_roc_auc - baseline["roc_auc"],
                "log_loss": test_log_loss - baseline["log_loss"],
                "top_pick_accuracy": test_top_pick_accuracy - baseline["top_pick_accuracy"],
            },
        }

    # new設定の採否は、validのlog lossのみで機械的に決める（testは判定に使わない）。
    chosen_config = (
        "new"
        if config_results["new"]["valid_log_loss"] < config_results["legacy"]["valid_log_loss"]
        else "legacy"
    )

    comparison: dict[str, object] = {
        "data_path": str(data_path),
        "train_start": train["race_date"].min().strftime("%Y-%m-%d"),
        "train_end": train["race_date"].max().strftime("%Y-%m-%d"),
        "valid_start": valid["race_date"].min().strftime("%Y-%m-%d"),
        "valid_end": valid["race_date"].max().strftime("%Y-%m-%d"),
        "test_start": test["race_date"].min().strftime("%Y-%m-%d"),
        "test_end": test["race_date"].max().strftime("%Y-%m-%d"),
        "train_races": int(train["race_id"].nunique()),
        "valid_races": int(valid["race_id"].nunique()),
        "test_races": int(test["race_id"].nunique()),
        "favorite_baseline": baseline,
        "configs": config_results,
        "selection": {
            "chosen_config": chosen_config,
            "decision_metric": "valid_log_loss",
            "decision_rule": (
                "legacyとnewのvalid log lossを比較し、小さい方を機械的に採用する。"
                "testの数値は記録のみで、この判定には使用していない。"
            ),
            "legacy_valid_log_loss": config_results["legacy"]["valid_log_loss"],
            "new_valid_log_loss": config_results["new"]["valid_log_loss"],
        },
        "note": (
            "回収率・購入シミュレーションはここでは計算していない"
            "（train_and_evaluateのmetrics.jsonに保存される）。"
        ),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "model_config_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)
    return comparison


def train_and_evaluate(
    data_path: Path, artifact_dir: Path, model_config: str = "new"
) -> dict[str, object]:
    """train < valid < test の時系列3分割で学習・評価する。

    testはモデル調整（設定選択・早期終了）に一切使用しない。
    回収率・信頼区間・較正指標はここで1回だけ計算し、metrics.jsonへ保存する。
    Streamlit側はこの保存済み値を読むだけで再計算しない。
    """
    data = pd.read_csv(data_path, parse_dates=["race_date"])
    validate_data(data)
    train, valid, test = temporal_split_three(data)

    preprocessor, model, best_iteration = fit_model(train, valid, model_config)
    raw_probability = predict_win_probability(preprocessor, model, test)

    predictions = test[
        [
            "race_id",
            "race_date",
            "horse_id",
            "horse_name",
            "horse_number",
            "pre_race_odds",
            "finish_position",
            "target_win",
        ]
    ].copy()
    predictions["raw_probability"] = raw_probability

    # 二値分類のraw probabilityをレース内で正規化した表示専用の暫定値。
    probability_sum = predictions.groupby("race_id")["raw_probability"].transform("sum")
    race_size = predictions.groupby("race_id")["raw_probability"].transform("size")
    predictions["ai_estimated_win_probability"] = np.where(
        probability_sum > 0,
        predictions["raw_probability"] / probability_sum,
        1.0 / race_size,
    )
    predictions["market_implied_probability"] = 1.0 / predictions["pre_race_odds"]
    predictions["value_gap"] = (
        predictions["ai_estimated_win_probability"] - predictions["market_implied_probability"]
    )

    # AI予想1位＝race_id内でai_estimated_win_probabilityが最大の1頭。
    ai_top_pick_indices = predictions.groupby("race_id")["ai_estimated_win_probability"].idxmax()
    ai_top_picks = predictions.loc[ai_top_pick_indices].copy()
    # 1番人気＝race_id内でpre_race_oddsが最小（＝市場暗黙確率が最大）の1頭。
    favorite_indices = predictions.groupby("race_id")["pre_race_odds"].idxmin()
    favorite_picks = predictions.loc[favorite_indices].copy()

    top_pick_accuracy = float(ai_top_picks["target_win"].mean())

    # ⑤ 回収率シミュレーション（架空データ。実競馬での収益性は示さない）。
    betting = {
        "ai_top_pick": summarize_betting_strategy(ai_top_picks),
        "favorite": summarize_betting_strategy(favorite_picks),
    }
    # ⑥ 確率較正状態の確認用指標（AI予想1位について）。
    calibration = summarize_calibration(ai_top_picks, "ai_estimated_win_probability")

    metrics: dict[str, object] = {
        "model_config": model_config,
        "roc_auc": float(roc_auc_score(test["target_win"], raw_probability)),
        "log_loss": float(log_loss(test["target_win"], raw_probability, labels=[0, 1])),
        "top_pick_accuracy": top_pick_accuracy,
        "favorite_pick_accuracy": float(favorite_picks["target_win"].mean()),
        "best_iteration": best_iteration,
        "train_start": train["race_date"].min().strftime("%Y-%m-%d"),
        "train_end": train["race_date"].max().strftime("%Y-%m-%d"),
        "valid_start": valid["race_date"].min().strftime("%Y-%m-%d"),
        "valid_end": valid["race_date"].max().strftime("%Y-%m-%d"),
        "test_start": test["race_date"].min().strftime("%Y-%m-%d"),
        "test_end": test["race_date"].max().strftime("%Y-%m-%d"),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_races": int(train["race_id"].nunique()),
        "valid_races": int(valid["race_id"].nunique()),
        "test_races": int(test["race_id"].nunique()),
        "feature_columns": FEATURE_COLUMNS,
        "result_columns_excluded": sorted(RESULT_COLUMNS),
        "betting": betting,
        "calibration": calibration,
        "probability_note": (
            "AI暫定推定勝率はLightGBMのraw probabilityをrace_id単位で正規化した表示用の暫定値"
        ),
        "simulation_note": (
            "回収率・的中率は架空データ上の購入シミュレーションであり、"
            "実競馬での収益性や的中を示すものではない。"
        ),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "result_columns_excluded": sorted(RESULT_COLUMNS),
            "model_config": model_config,
        },
        artifact_dir / "model.joblib",
    )
    predictions.sort_values(
        ["race_date", "race_id", "ai_estimated_win_probability"],
        ascending=[True, True, False],
    ).to_csv(artifact_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBMモデルを時系列3分割で学習します。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--model-config",
        choices=["legacy", "new"],
        default="new",
        help="legacy=v0.1のLightGBM設定 / new=validのみで選んだPhase A設定",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "legacy/new設定を同一のtrain/valid/testで評価し、"
            "model_config_comparison.jsonへ保存して終了する"
            "（--model-configは無視され、artifacts/配下の学習成果物は更新しない）"
        ),
    )
    args = parser.parse_args()

    if args.compare:
        started_at = time.perf_counter()
        comparison = compare_model_configs(args.data, args.artifacts)
        elapsed_seconds = time.perf_counter() - started_at

        print(f"train_period={comparison['train_start']}..{comparison['train_end']}")
        print(f"valid_period={comparison['valid_start']}..{comparison['valid_end']}")
        print(f"test_period={comparison['test_start']}..{comparison['test_end']}")
        print(
            f"train_races={comparison['train_races']} "
            f"valid_races={comparison['valid_races']} "
            f"test_races={comparison['test_races']}"
        )
        baseline = comparison["favorite_baseline"]
        print(
            "favorite_baseline "
            f"roc_auc={baseline['roc_auc']:.6f} "
            f"log_loss={baseline['log_loss']:.6f} "
            f"top_pick_accuracy={baseline['top_pick_accuracy']:.6f}"
        )
        for config_name, result in comparison["configs"].items():
            print(
                f"[{config_name}] "
                f"valid_roc_auc={result['valid_roc_auc']:.6f} "
                f"valid_log_loss={result['valid_log_loss']:.6f} "
                f"test_roc_auc={result['test_roc_auc']:.6f} "
                f"test_log_loss={result['test_log_loss']:.6f} "
                f"test_top_pick_accuracy={result['test_top_pick_accuracy']:.6f}"
            )
        print(f"chosen_config={comparison['selection']['chosen_config']} (valid log lossのみで判定)")
        print(f"artifact_dir={args.artifacts.resolve()}")
        print(f"elapsed_seconds={elapsed_seconds:.1f}")
        return

    started_at = time.perf_counter()
    metrics = train_and_evaluate(args.data, args.artifacts, model_config=args.model_config)
    elapsed_seconds = time.perf_counter() - started_at

    print(f"model_config={metrics['model_config']}")
    print(f"train_period={metrics['train_start']}..{metrics['train_end']}")
    print(f"valid_period={metrics['valid_start']}..{metrics['valid_end']}")
    print(f"test_period={metrics['test_start']}..{metrics['test_end']}")
    print(f"train_rows={metrics['train_rows']} valid_rows={metrics['valid_rows']} test_rows={metrics['test_rows']}")
    print(f"train_races={metrics['train_races']} valid_races={metrics['valid_races']} test_races={metrics['test_races']}")
    print(f"roc_auc={metrics['roc_auc']:.6f}")
    print(f"log_loss={metrics['log_loss']:.6f}")
    print(f"top_pick_accuracy={metrics['top_pick_accuracy']:.6f}")
    print(f"favorite_pick_accuracy={metrics['favorite_pick_accuracy']:.6f}")
    betting = metrics["betting"]
    print(
        "ai_top_pick recovery_rate="
        f"{betting['ai_top_pick']['recovery_rate']:.4f} "
        f"ci95={betting['ai_top_pick']['recovery_rate_ci95']}"
    )
    print(
        "favorite recovery_rate="
        f"{betting['favorite']['recovery_rate']:.4f} "
        f"ci95={betting['favorite']['recovery_rate_ci95']}"
    )
    print(f"artifact_dir={args.artifacts.resolve()}")
    print(f"elapsed_seconds={elapsed_seconds:.1f}")


if __name__ == "__main__":
    main()
