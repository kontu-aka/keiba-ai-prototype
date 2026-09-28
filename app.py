"""競馬予想AI Prototype v0.1 のStreamlit画面。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "artifacts" / "test_predictions.csv"


st.set_page_config(page_title="競馬予想AI Prototype v0.1", page_icon="🏇", layout="wide")
st.title("競馬予想AI Prototype v0.1")
st.warning(
    "この画面は固定seedで生成した架空のサンプルデータによる技術デモです。"
    "実在レースの予測や実運用での精度を示すものではありません。"
)

if not METRICS_PATH.exists() or not PREDICTIONS_PATH.exists():
    st.error("学習成果物がありません。先にサンプルデータ生成とモデル学習を実行してください。")
    st.code(
        ".venv\\Scripts\\python.exe src\\generate_sample_data.py\n"
        ".venv\\Scripts\\python.exe src\\train_model.py",
        language="powershell",
    )
    st.stop()


@st.cache_data
def load_artifacts() -> tuple[dict[str, object], pd.DataFrame]:
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics_data = json.load(file)
    predictions_data = pd.read_csv(PREDICTIONS_PATH, parse_dates=["race_date"])
    return metrics_data, predictions_data


metrics, predictions = load_artifacts()

st.subheader("時系列評価")
st.write(
    f"**train期間:** {metrics['train_start']} ～ {metrics['train_end']}　　"
    f"**test期間:** {metrics['test_start']} ～ {metrics['test_end']}"
)
metric_columns = st.columns(4)
metric_columns[0].metric("ROC-AUC", f"{float(metrics['roc_auc']):.3f}")
metric_columns[1].metric("Log Loss", f"{float(metrics['log_loss']):.3f}")
metric_columns[2].metric("Top-pick accuracy", f"{float(metrics['top_pick_accuracy']):.1%}")
metric_columns[3].metric("testレース数", f"{int(metrics['test_races']):,}")
st.caption(f"test出走馬数: {int(metrics['test_rows']):,}行")

st.subheader("testレース別の予測")
race_summary = (
    predictions[["race_id", "race_date"]]
    .drop_duplicates()
    .sort_values(["race_date", "race_id"])
)
race_labels = {
    row.race_id: f"{row.race_date:%Y-%m-%d} / {row.race_id}"
    for row in race_summary.itertuples(index=False)
}
selected_race = st.selectbox(
    "race_idを選択",
    options=list(race_labels),
    format_func=lambda race_id: race_labels[race_id],
)

race_table = predictions.loc[predictions["race_id"] == selected_race].copy()
race_table = race_table.sort_values("ai_estimated_win_probability", ascending=False)
display_table = race_table[
    [
        "horse_name",
        "horse_number",
        "ai_estimated_win_probability",
        "pre_race_odds",
        "market_implied_probability",
        "value_gap",
        "finish_position",
    ]
].rename(
    columns={
        "horse_name": "馬名",
        "horse_number": "馬番",
        "ai_estimated_win_probability": "AI暫定推定勝率",
        "pre_race_odds": "予想時点オッズ",
        "market_implied_probability": "市場暗黙確率",
        "value_gap": "value_gap",
        "finish_position": "実際の着順",
    }
)
display_table["AI暫定推定勝率"] = display_table["AI暫定推定勝率"].map(lambda value: f"{value:.1%}")
display_table["市場暗黙確率"] = display_table["市場暗黙確率"].map(lambda value: f"{value:.1%}")
display_table["value_gap"] = display_table["value_gap"].map(lambda value: f"{value:+.1%}")
st.dataframe(display_table, hide_index=True, width="stretch")

st.info(
    "AI暫定推定勝率は、各馬を独立に判定したLightGBMのraw probabilityを"
    "race_id単位で合計100%に正規化した表示用の暫定値です。厳密な競走勝率ではありません。"
)
st.caption(
    "市場暗黙確率（1 / 予想時点オッズ）は控除込みで、レース内合計が100%を超える場合があります。"
    "合計100%へ正規化したAI暫定推定勝率と厳密に同一基準の比較ではないため、"
    "value_gapはマイナス方向へ偏りやすい参考値です。正の値でも購入推奨や購入した事実を意味しません。"
    "馬券購入機能はありません。"
)

with st.expander("設計上の注意"):
    st.markdown(
        "- `finish_position` と `target_win` は評価専用で、モデル特徴量には使用していません。\n"
        "- 古い日付をtrain、新しい日付をtestにした時系列分割です。\n"
        "- 予想時点オッズは結果確定前に利用可能な入力としてモデルにも使用しています。\n"
        "- JRA-VAN / JV-Link、スクレイピング、馬券購入には接続していません。"
    )
