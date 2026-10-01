"""競馬予想AI Prototype — 応募用Streamlit公開デモ。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.train_model import train_and_evaluate


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_races.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "test_predictions.csv"


st.set_page_config(
    page_title="競馬予想AI デモ",
    page_icon="🏇",
    layout="wide",
)

st.title("🏇 競馬予想AI デモ")
st.write(
    "出走馬データから、機械学習モデルが各馬の「1着になりそう度」を推定する"
    "応募用Prototypeです。"
)
st.warning(
    "このデモは固定seedで生成した架空データを使用しています。"
    "実在レースの予測や、実運用での精度を示すものではありません。"
)


# v0.2のmetrics.jsonが持つべき必須キー。train/valid/testの3分割と、
# 回収率・較正指標（betting/calibration）がこれに含まれる。
# 旧v0.1形式のmetrics.json（train/test 2分割、betting/calibrationなし）は
# ここで不正と判定し、v0.2形式として再生成する。
REQUIRED_METRICS_KEYS = {
    "valid_start",
    "valid_end",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "train_races",
    "valid_races",
    "test_races",
    "betting",
    "calibration",
}


def _is_valid_v0_2_metrics(metrics_data: object) -> bool:
    """ファイルが存在するだけでなく、v0.2形式として必要なキーを持つかを確認する。"""
    if not isinstance(metrics_data, dict):
        return False
    if not REQUIRED_METRICS_KEYS.issubset(metrics_data.keys()):
        return False
    betting = metrics_data.get("betting")
    if not isinstance(betting, dict) or "ai_top_pick" not in betting or "favorite" not in betting:
        return False
    if not isinstance(metrics_data.get("calibration"), dict):
        return False
    return True


def ensure_demo_artifacts() -> None:
    """ローカル生成物がない、またはv0.2形式でない場合は、架空CSVから学習し直す。

    metrics.jsonが存在するだけでは「有効なartifact」とは判定しない。
    旧v0.1形式（train/test 2分割、betting/calibrationなし）が残っている場合も、
    KeyErrorにせずv0.2形式として再生成する。
    """
    artifacts_valid = False
    if METRICS_PATH.exists() and PREDICTIONS_PATH.exists():
        try:
            with METRICS_PATH.open(encoding="utf-8") as file:
                existing_metrics = json.load(file)
            artifacts_valid = _is_valid_v0_2_metrics(existing_metrics)
        except (OSError, json.JSONDecodeError):
            artifacts_valid = False

    if artifacts_valid:
        return

    if not DATA_PATH.exists():
        st.error("公開デモ用のサンプルデータが見つかりません。")
        st.stop()

    with st.spinner("初回起動、またはartifact形式更新のため、デモ用AIモデルを準備しています…"):
        try:
            train_and_evaluate(DATA_PATH, ARTIFACT_DIR)
        except Exception:
            st.error(
                "デモ用AIモデルの準備に失敗しました。"
                "時間をおいて再読み込みしてください。"
            )
            st.stop()


ensure_demo_artifacts()


@st.cache_data
def load_artifacts() -> tuple[dict[str, object], pd.DataFrame]:
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics_data = json.load(file)

    predictions_data = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["race_date"],
    )
    return metrics_data, predictions_data


metrics, predictions = load_artifacts()

st.subheader("① レースを選ぶ")

race_summary = (
    predictions[["race_id", "race_date"]]
    .drop_duplicates()
    .sort_values(["race_date", "race_id"])
)

min_race_date = race_summary["race_date"].min().date()
max_race_date = race_summary["race_date"].max().date()

date_range = st.date_input(
    "日付で絞り込む（testレースのみ表示）",
    value=(min_race_date, max_race_date),
    min_value=min_race_date,
    max_value=max_race_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    range_start, range_end = date_range
else:
    range_start, range_end = min_race_date, max_race_date

filtered_race_summary = race_summary.loc[
    (race_summary["race_date"].dt.date >= range_start)
    & (race_summary["race_date"].dt.date <= range_end)
]
if filtered_race_summary.empty:
    st.warning("選択した日付範囲にレースがありません。日付範囲を見直してください。")
    st.stop()

race_labels = {
    row.race_id: f"{row.race_date:%Y-%m-%d} / {row.race_id}"
    for row in filtered_race_summary.itertuples(index=False)
}

selected_race = st.selectbox(
    "表示するテストレース",
    options=list(race_labels),
    format_func=lambda race_id: race_labels[race_id],
    label_visibility="collapsed",
)

race_table = predictions.loc[
    predictions["race_id"] == selected_race
].copy()

race_table = race_table.sort_values(
    "ai_estimated_win_probability",
    ascending=False,
).reset_index(drop=True)

race_table["ai_prediction_rank"] = range(1, len(race_table) + 1)

top_pick = race_table.iloc[0]
winner_rows = race_table.loc[race_table["finish_position"] == 1]
actual_winner = winner_rows.iloc[0] if not winner_rows.empty else None


st.subheader("② このレースのAI予想")

summary_columns = st.columns(4)

summary_columns[0].metric(
    "AI予想1位",
    str(top_pick["horse_name"]),
)
summary_columns[1].metric(
    "AIの勝ちそう度",
    f"{float(top_pick['ai_estimated_win_probability']):.1%}",
)
summary_columns[2].metric(
    "予想時点オッズ",
    f"{float(top_pick['pre_race_odds']):.1f}倍",
)
summary_columns[3].metric(
    "実際の結果",
    f"{int(top_pick['finish_position'])}着",
)

if int(top_pick["finish_position"]) == 1:
    st.success(
        f"このレースでは、AI予想1位の「{top_pick['horse_name']}」が"
        "実際に1着でした。"
    )
elif actual_winner is not None:
    actual_winner_rank = int(actual_winner["ai_prediction_rank"])
    st.info(
        f"AI予想1位の「{top_pick['horse_name']}」は"
        f"{int(top_pick['finish_position'])}着でした。"
        f" 実際の1着は「{actual_winner['horse_name']}」で、"
        f"AI予想順位は{actual_winner_rank}位でした。"
    )


st.subheader("③ AI予想ランキング")
st.caption(
    "AIが1着になりそうと見た順に並べています。"
    "まずは「AIの勝ちそう度」と「実際の着順」を見比べればOKです。"
)

simple_table = race_table[
    [
        "ai_prediction_rank",
        "horse_name",
        "horse_number",
        "ai_estimated_win_probability",
        "pre_race_odds",
        "finish_position",
    ]
].rename(
    columns={
        "ai_prediction_rank": "AI予想順位",
        "horse_name": "馬名",
        "horse_number": "馬番",
        "ai_estimated_win_probability": "AIの勝ちそう度",
        "pre_race_odds": "オッズ",
        "finish_position": "実際の着順",
    }
)

simple_table["AIの勝ちそう度"] = simple_table[
    "AIの勝ちそう度"
].map(lambda value: f"{value:.1%}")

simple_table["オッズ"] = simple_table["オッズ"].map(
    lambda value: f"{value:.1f}倍"
)

simple_table["実際の着順"] = simple_table["実際の着順"].map(
    lambda value: f"{int(value)}着"
)

st.dataframe(
    simple_table,
    hide_index=True,
    width="stretch",
)


with st.expander("モデルの詳しい評価を見る"):
    st.write(
        "ここから下は、モデルの性能や検証方法を確認したい方向けの情報です。"
    )
    st.write(
        f"**train期間:** {metrics['train_start']} ～ {metrics['train_end']}　　"
        f"**valid期間:** {metrics['valid_start']} ～ {metrics['valid_end']}　　"
        f"**test期間:** {metrics['test_start']} ～ {metrics['test_end']}"
    )
    st.caption(
        "validはモデル設定の選択にのみ使用し、testはモデル調整に使用していません。"
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "ROC-AUC",
        f"{float(metrics['roc_auc']):.3f}",
    )
    metric_columns[1].metric(
        "Log Loss",
        f"{float(metrics['log_loss']):.3f}",
    )
    metric_columns[2].metric(
        "Top-pick accuracy",
        f"{float(metrics['top_pick_accuracy']):.1%}",
    )
    metric_columns[3].metric(
        "testレース数",
        f"{int(metrics['test_races']):,}",
    )
    st.caption(
        f"test出走馬数: {int(metrics['test_rows']):,}行"
    )


with st.expander("購入シミュレーションと確率較正の詳細を見る（架空データ）"):
    st.warning(
        "これは架空データ上の購入シミュレーションです。"
        "実競馬での収益性や的中を示すものではなく、購入を推奨するものでもありません。"
    )

    ai_betting = metrics["betting"]["ai_top_pick"]
    favorite_betting = metrics["betting"]["favorite"]
    calibration = metrics["calibration"]

    st.write("**単勝100円を毎レース購入したと仮定した場合（test全体）**")
    sim_columns = st.columns(2)
    with sim_columns[0]:
        st.markdown("**AI予想1位**")
        st.metric("的中率", f"{ai_betting['hit_rate']:.1%}")
        ai_ci_low, ai_ci_high = ai_betting["recovery_rate_ci95"]
        st.metric(
            "回収率（95%信頼区間）",
            f"{ai_betting['recovery_rate']:.1%}",
            help=f"bootstrap 95%CI: {ai_ci_low:.1%} ～ {ai_ci_high:.1%}",
        )
        st.caption(f"95%CI: {ai_ci_low:.1%} ～ {ai_ci_high:.1%}")
    with sim_columns[1]:
        st.markdown("**1番人気**")
        st.metric("的中率", f"{favorite_betting['hit_rate']:.1%}")
        fav_ci_low, fav_ci_high = favorite_betting["recovery_rate_ci95"]
        st.metric(
            "回収率（95%信頼区間）",
            f"{favorite_betting['recovery_rate']:.1%}",
            help=f"bootstrap 95%CI: {fav_ci_low:.1%} ～ {fav_ci_high:.1%}",
        )
        st.caption(f"95%CI: {fav_ci_low:.1%} ～ {fav_ci_high:.1%}")

    st.caption(
        f"信頼区間はレース単位のbootstrap（{ai_betting['bootstrap_iterations']:,}回、"
        f"固定seed）による95%区間です。"
    )

    st.write("**AI予想1位の確率較正状態（test全体）**")
    calibration_columns = st.columns(3)
    calibration_columns[0].metric(
        "表示勝率の平均",
        f"{calibration['avg_predicted_probability']:.1%}",
    )
    calibration_columns[1].metric(
        "実際の勝率",
        f"{calibration['actual_win_rate']:.1%}",
    )
    calibration_columns[2].metric(
        "差（表示 − 実際）",
        f"{calibration['difference']:+.1%}",
    )
    st.caption(
        "差がゼロに近いほど、表示している勝率と実際の勝率が近いことを意味します。"
        "この差はモデルの精度そのものではなく、較正（キャリブレーション）状態の目安です。"
    )


with st.expander("市場オッズとの詳しい比較を見る"):
    detail_table = race_table[
        [
            "horse_name",
            "ai_estimated_win_probability",
            "pre_race_odds",
            "market_implied_probability",
            "value_gap",
            "finish_position",
        ]
    ].rename(
        columns={
            "horse_name": "馬名",
            "ai_estimated_win_probability": "AI暫定推定勝率",
            "pre_race_odds": "予想時点オッズ",
            "market_implied_probability": "市場暗黙確率",
            "value_gap": "value_gap",
            "finish_position": "実際の着順",
        }
    )

    detail_table["AI暫定推定勝率"] = detail_table[
        "AI暫定推定勝率"
    ].map(lambda value: f"{value:.1%}")

    detail_table["市場暗黙確率"] = detail_table[
        "市場暗黙確率"
    ].map(lambda value: f"{value:.1%}")

    detail_table["value_gap"] = detail_table["value_gap"].map(
        lambda value: f"{value:+.1%}"
    )

    detail_table["実際の着順"] = detail_table["実際の着順"].map(
        lambda value: f"{int(value)}着"
    )

    st.dataframe(
        detail_table,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "市場暗黙確率は「1 / 予想時点オッズ」です。"
        "競馬市場の控除を含むため、レース内合計が100%を超える場合があります。"
        "AI暫定推定勝率とは厳密に同一基準の比較ではないため、"
        "value_gapは参考値です。購入推奨ではありません。"
    )


with st.expander("設計上の注意"):
    st.markdown(
        "- AI暫定推定勝率は、LightGBMのraw probabilityを"
        "race_id単位で合計100%に正規化した表示用の暫定値です。\n"
        "- `finish_position` と `target_win` は評価専用で、"
        "モデル特徴量には使用していません。\n"
        "- `horse_number` は架空データ上で結果と意味のある関係を持たないため、"
        "特徴量から除外しています。\n"
        "- 古い日付から新しい日付の順に train → valid → test と時系列3分割し、"
        "testはモデル調整に使用していません。\n"
        "- 予想時点オッズは結果確定前に利用可能な入力としてモデルにも使用しています。\n"
        "- JRA-VAN / JV-Link、スクレイピング、馬券購入には接続していません。"
    )
