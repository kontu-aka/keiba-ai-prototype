"""架空データ上の購入シミュレーションと確率較正指標を計算するユーティリティ。

ここで計算する回収率・信頼区間は、あくまで架空データ上のシミュレーションであり、
実競馬での収益性や的中を示すものではない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STAKE_PER_RACE_YEN = 100
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260928


def _race_payouts(picked: pd.DataFrame) -> np.ndarray:
    """レースごとに、単勝100円が的中した場合の払戻額（円）を返す。"""
    is_win = picked["target_win"].to_numpy() == 1
    odds = picked["pre_race_odds"].to_numpy()
    return np.where(is_win, odds * STAKE_PER_RACE_YEN, 0.0)


def bootstrap_recovery_rate_ci(
    picked: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """レース単位の復元抽出bootstrapで、回収率の95%信頼区間を求める。"""
    rng = np.random.default_rng(seed)
    n_races = len(picked)
    payouts = _race_payouts(picked)
    sample_indices = rng.integers(0, n_races, size=(iterations, n_races))
    sampled_returns = payouts[sample_indices].sum(axis=1)
    sampled_stake = n_races * STAKE_PER_RACE_YEN
    recovery_rates = sampled_returns / sampled_stake
    lower, upper = np.percentile(recovery_rates, [2.5, 97.5])
    return float(lower), float(upper)


def summarize_betting_strategy(picked: pd.DataFrame) -> dict[str, object]:
    """1レースにつき単勝100円を購入したと仮定した場合の的中率・回収率・95%信頼区間。

    picked は race_id ごとに1行（そのレースで購入したと仮定する1頭）を持つDataFrame。
    """
    n_races = int(len(picked))
    stake_total = float(STAKE_PER_RACE_YEN * n_races)
    payouts = _race_payouts(picked)
    return_total = float(payouts.sum())
    ci_lower, ci_upper = bootstrap_recovery_rate_ci(picked)
    return {
        "n_races": n_races,
        "stake_yen": stake_total,
        "return_yen": return_total,
        "hit_rate": float(picked["target_win"].mean()),
        "recovery_rate": return_total / stake_total if stake_total > 0 else float("nan"),
        "recovery_rate_ci95": [ci_lower, ci_upper],
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def summarize_calibration(picked: pd.DataFrame, probability_column: str) -> dict[str, float]:
    """予想1位の表示確率の平均と、実際の勝率の差を返す（較正状態の確認用）。"""
    avg_predicted = float(picked[probability_column].mean())
    actual_win_rate = float(picked["target_win"].mean())
    return {
        "avg_predicted_probability": avg_predicted,
        "actual_win_rate": actual_win_rate,
        "difference": avg_predicted - actual_win_rate,
    }
