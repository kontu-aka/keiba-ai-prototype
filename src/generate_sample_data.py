"""再現可能な架空の競馬データを生成する。実在データは使用しない。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sample_races.csv"
DEFAULT_SEED = 20260928
DISTANCES = np.array([1200, 1400, 1600, 1800, 2000, 2400])
SURFACES = ("芝", "ダート")
TRACK_CONDITIONS = ("良", "稍重", "重")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def generate_sample_data(
    seed: int = DEFAULT_SEED,
    number_of_dates: int = 640,
    races_per_date: int = 3,
) -> pd.DataFrame:
    """固定seedで、予想時点特徴量とレース結果を持つ架空データを返す。

    number_of_dates / races_per_date はv0.1比でおよそ10倍のレース数を生成する。
    実競馬のデータ量を再現する意図ではなく、架空データ上で評価値のばらつきを
    減らし、評価パイプラインを検証するための規模拡大。
    """
    rng = np.random.default_rng(seed)
    horse_count = 360
    jockey_count = 64

    horse_profiles = {
        f"H{index:03d}": {
            "name": f"サンプルホース{index:03d}",
            "ability": rng.normal(0.0, 0.85),
            "base_age": int(rng.integers(3, 7)),
            "preferred_distance": int(rng.choice(DISTANCES)),
            "preferred_surface": str(rng.choice(SURFACES)),
        }
        for index in range(1, horse_count + 1)
    }
    jockey_ability = rng.normal(0.0, 0.55, jockey_count)

    # ここに入る履歴は、各レース終了後にのみ更新する。
    horse_results: dict[str, list[int]] = defaultdict(list)
    horse_surface_results: dict[tuple[str, str], list[int]] = defaultdict(list)
    horse_distance_results: dict[tuple[str, int], list[int]] = defaultdict(list)
    horse_last_date: dict[str, pd.Timestamp] = {}
    jockey_results: dict[int, list[int]] = defaultdict(list)

    # レース数10倍化後もnumber_of_dates=640で最終日が現在日付より未来にならないよう、
    # v0.1の開始日(2024-01-06)より前倒しした固定日付。
    start_date = pd.Timestamp("2021-05-02")
    race_dates = [start_date + pd.Timedelta(days=3 * offset) for offset in range(number_of_dates)]
    rows: list[dict[str, object]] = []

    for race_date in race_dates:
        available_horses = list(rng.permutation(list(horse_profiles)))
        cursor = 0

        for race_number in range(1, races_per_date + 1):
            field_size = int(rng.integers(8, 14))
            entrants = available_horses[cursor : cursor + field_size]
            cursor += field_size
            distance = int(rng.choice(DISTANCES))
            surface = str(rng.choice(SURFACES, p=[0.65, 0.35]))
            track_condition = str(rng.choice(TRACK_CONDITIONS, p=[0.65, 0.23, 0.12]))
            race_id = f"S{race_date:%Y%m%d}_{race_number:02d}"
            jockey_ids = rng.choice(jockey_count, size=field_size, replace=False)

            pre_race_rows: list[dict[str, object]] = []
            market_scores: list[float] = []
            performance_scores: list[float] = []

            for horse_number, (horse_id, jockey_id) in enumerate(zip(entrants, jockey_ids), start=1):
                profile = horse_profiles[horse_id]
                all_results = horse_results[horse_id]
                recent_results = all_results[-5:]
                surface_results = horse_surface_results[(horse_id, surface)]
                distance_results = horse_distance_results[(horse_id, distance)]
                jockey_history = jockey_results[int(jockey_id)]

                recent_win_rate = (sum(position == 1 for position in recent_results) + 0.5) / (
                    len(recent_results) + 5.0
                )
                recent_top3_rate = (sum(position <= 3 for position in recent_results) + 1.5) / (
                    len(recent_results) + 5.0
                )
                avg_finish_last5 = (sum(recent_results) + 6.0 * 3.0) / (len(recent_results) + 3.0)
                surface_win_rate = (sum(position == 1 for position in surface_results) + 0.5) / (
                    len(surface_results) + 5.0
                )
                distance_top3_rate = (sum(position <= 3 for position in distance_results) + 1.0) / (
                    len(distance_results) + 4.0
                )
                jockey_win_rate = (sum(position == 1 for position in jockey_history) + 1.0) / (
                    len(jockey_history) + 12.0
                )
                days_since_last_race = (
                    int((race_date - horse_last_date[horse_id]).days)
                    if horse_id in horse_last_date
                    else 45
                )
                distance_match = 1.0 - min(abs(distance - int(profile["preferred_distance"])) / 1200.0, 1.0)
                surface_match = 1.0 if surface == profile["preferred_surface"] else -0.35
                recent_form = 0.8 * recent_top3_rate - 0.08 * avg_finish_last5
                carried_weight = round(
                    54.0
                    + 0.45 * (int(profile["base_age"]) - 3)
                    + rng.normal(0.0, 1.0),
                    1,
                )

                # オッズも結果確定前に利用可能な架空の市場情報として生成する。
                market_score = (
                    1.05 * float(profile["ability"])
                    + 0.55 * distance_match
                    + 0.35 * surface_match
                    + 0.35 * float(jockey_ability[int(jockey_id)])
                    + 0.45 * recent_form
                    + rng.normal(0.0, 0.35)
                )
                # 着順には当日の偶然性を加える。targetを特徴量から直接作るリークではない。
                performance_score = (
                    1.15 * float(profile["ability"])
                    + 0.65 * distance_match
                    + 0.45 * surface_match
                    + 0.40 * float(jockey_ability[int(jockey_id)])
                    + 0.40 * recent_form
                    - 0.035 * (carried_weight - 55.0)
                    + rng.gumbel(0.0, 0.85)
                )

                pre_race_rows.append(
                    {
                        "race_id": race_id,
                        "race_date": race_date.strftime("%Y-%m-%d"),
                        "horse_id": horse_id,
                        "horse_name": profile["name"],
                        "horse_number": horse_number,
                        "horse_age": int(profile["base_age"]),
                        "carried_weight": carried_weight,
                        "distance": distance,
                        "surface": surface,
                        "track_condition": track_condition,
                        "recent_win_rate": round(recent_win_rate, 5),
                        "recent_top3_rate": round(recent_top3_rate, 5),
                        "avg_finish_last5": round(avg_finish_last5, 5),
                        "career_starts": len(all_results),
                        "days_since_last_race": days_since_last_race,
                        "surface_win_rate": round(surface_win_rate, 5),
                        "distance_top3_rate": round(distance_top3_rate, 5),
                        "jockey_win_rate": round(jockey_win_rate, 5),
                    }
                )
                market_scores.append(market_score)
                performance_scores.append(performance_score)

            market_probabilities = _softmax(np.asarray(market_scores))
            # 18%の控除相当を含むため、1/oddsのレース内合計は概ね100%を超える。
            odds = np.clip(1.0 / (market_probabilities * 1.18), 1.2, 99.9)
            finish_order = np.argsort(-np.asarray(performance_scores))
            finish_positions = np.empty(field_size, dtype=int)
            finish_positions[finish_order] = np.arange(1, field_size + 1)

            for index, row in enumerate(pre_race_rows):
                row["pre_race_odds"] = round(float(odds[index]), 1)
                row["finish_position"] = int(finish_positions[index])
                row["target_win"] = int(finish_positions[index] == 1)
                rows.append(row)

            # 同一レースの全結果が決まった後に履歴を更新する。
            for index, (horse_id, jockey_id) in enumerate(zip(entrants, jockey_ids)):
                finish_position = int(finish_positions[index])
                horse_results[horse_id].append(finish_position)
                horse_surface_results[(horse_id, surface)].append(finish_position)
                horse_distance_results[(horse_id, distance)].append(finish_position)
                horse_last_date[horse_id] = race_date
                jockey_results[int(jockey_id)].append(finish_position)

    data = pd.DataFrame(rows).sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
    winners_per_race = data.groupby("race_id")["target_win"].sum()
    if not winners_per_race.eq(1).all():
        raise RuntimeError("各レースの1着馬が1頭になっていません。")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="架空のサンプル競馬データを生成します。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    data = generate_sample_data(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"output={args.output.resolve()}")
    print(f"seed={args.seed}")
    print(f"races={data['race_id'].nunique()}")
    print(f"rows={len(data)}")
    print(f"period={data['race_date'].min()}..{data['race_date'].max()}")


if __name__ == "__main__":
    main()
