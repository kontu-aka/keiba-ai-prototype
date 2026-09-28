"""架空データを日付順に分割し、LightGBMモデルを学習・評価する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "sample_races.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RANDOM_SEED = 20260928

# 予想時点で利用可能な列だけを、意図的に列挙する。
NUMERIC_FEATURES = [
    "horse_number",
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
    if data["race_date"].nunique() < 2:
        raise ValueError("時系列分割には2日以上のデータが必要です。")


def temporal_split(data: pd.DataFrame, train_ratio: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ユニーク日付を境にし、古い日付をtrain、新しい日付をtestへ分ける。"""
    unique_dates = np.sort(data["race_date"].unique())
    split_index = min(max(int(len(unique_dates) * train_ratio), 1), len(unique_dates) - 1)
    first_test_date = unique_dates[split_index]
    train = data.loc[data["race_date"] < first_test_date].copy()
    test = data.loc[data["race_date"] >= first_test_date].copy()
    if train["race_date"].max() >= test["race_date"].min():
        raise RuntimeError("trainとtestの日付順序が不正です。")
    return train, test


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_and_evaluate(data_path: Path, artifact_dir: Path) -> dict[str, object]:
    data = pd.read_csv(data_path, parse_dates=["race_date"])
    validate_data(data)
    train, test = temporal_split(data)

    pipeline = build_pipeline()
    pipeline.fit(train[FEATURE_COLUMNS], train["target_win"])
    raw_probability = pipeline.predict_proba(test[FEATURE_COLUMNS])[:, 1]

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

    top_pick_indices = predictions.groupby("race_id")["ai_estimated_win_probability"].idxmax()
    top_pick_accuracy = float(predictions.loc[top_pick_indices, "target_win"].mean())
    metrics: dict[str, object] = {
        "roc_auc": float(roc_auc_score(test["target_win"], raw_probability)),
        "log_loss": float(log_loss(test["target_win"], raw_probability, labels=[0, 1])),
        "top_pick_accuracy": top_pick_accuracy,
        "train_start": train["race_date"].min().strftime("%Y-%m-%d"),
        "train_end": train["race_date"].max().strftime("%Y-%m-%d"),
        "test_start": test["race_date"].min().strftime("%Y-%m-%d"),
        "test_end": test["race_date"].max().strftime("%Y-%m-%d"),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_races": int(train["race_id"].nunique()),
        "test_races": int(test["race_id"].nunique()),
        "feature_columns": FEATURE_COLUMNS,
        "result_columns_excluded": sorted(RESULT_COLUMNS),
        "probability_note": (
            "AI暫定推定勝率はLightGBMのraw probabilityをrace_id単位で正規化した表示用の暫定値"
        ),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "result_columns_excluded": sorted(RESULT_COLUMNS),
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
    parser = argparse.ArgumentParser(description="LightGBMモデルを時系列分割で学習します。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    metrics = train_and_evaluate(args.data, args.artifacts)
    print(f"train_period={metrics['train_start']}..{metrics['train_end']}")
    print(f"test_period={metrics['test_start']}..{metrics['test_end']}")
    print(f"train_rows={metrics['train_rows']}")
    print(f"test_rows={metrics['test_rows']}")
    print(f"test_races={metrics['test_races']}")
    print(f"roc_auc={metrics['roc_auc']:.6f}")
    print(f"log_loss={metrics['log_loss']:.6f}")
    print(f"top_pick_accuracy={metrics['top_pick_accuracy']:.6f}")
    print(f"artifact_dir={args.artifacts.resolve()}")


if __name__ == "__main__":
    main()
