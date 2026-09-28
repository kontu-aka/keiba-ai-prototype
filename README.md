# 競馬予想AI Prototype v0.1

応募時に「サンプルデータ生成から画面表示まで、一連の機械学習プロトタイプを開発できる」ことを示すための小さな完成見本です。固定random seedで作る架空データを使い、前処理、LightGBM学習、時系列評価、Streamlit表示までを再現できます。

> **重要:** このプロジェクトは架空のサンプルデータによる技術デモです。実在レースの予測や、実運用で通用する精度を示すものではありません。

## v0.1でできること

- 固定seedによる架空の複数日・複数レースデータの生成
- 予想時点で利用可能な特徴量の前処理
- LightGBMによる各出走馬の1着／非1着の二値分類
- 日付順のtrain/test分割と、ROC-AUC・Log Loss・Top-pick accuracyの算出
- 予想時点オッズから求めた市場暗黙確率との比較
- testレースを選べるStreamlit画面

## v0.1でできないこと

- 実在する馬・レースの予測
- JRA-VAN / JV-Linkとの接続
- Webサイトのスクレイピングや実データ取得
- 自動的な予想ルール変更
- 馬券の購入判断・購入記録・購入処理
- 実運用レベルの精度保証

## セットアップ

Windows PowerShellでプロジェクトフォルダへ移動し、Python 3.12の仮想環境を使用します。

```powershell
cd keiba-ai-prototype
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

すでに `.venv` とライブラリが用意されている場合、再作成や再インストールは不要です。グローバルPythonへインストールする必要はありません。

### 動作確認済み環境

2026-09-28時点で、次の環境によるデータ生成・学習・画面起動を確認しています。

| ライブラリ | バージョン |
|---|---:|
| Python | 3.12.10 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| streamlit | 1.64.0 |
| joblib | 1.6.0 |
| matplotlib | 3.11.2 |

`requirements.txt` はコードから直接使用するライブラリだけを記載しています。matplotlib 3.11.2も確認環境にはインストールされていますが、v0.1のコードでは使用していないため必須依存には含めていません。

## 実行方法

### 1. サンプルデータ生成

```powershell
.venv\Scripts\python.exe src\generate_sample_data.py
```

`data/sample_races.csv` が生成されます。出力には `finish_position` と `target_win` も含まれますが、これらは正解・評価用の結果情報です。

### 2. モデル学習と評価

```powershell
.venv\Scripts\python.exe src\train_model.py
```

古い日付をtrain、新しい日付をtestとして学習し、コンソールに期間と評価値を表示します。`artifacts/` には学習済みモデル、評価値、test予測が生成されます。

### 3. Streamlit起動

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

表示されたローカルURLをブラウザで開きます。停止するときはPowerShellで `Ctrl+C` を押します。

## 評価指標

- **ROC-AUC:** 1着馬を非1着馬より高く順位付けできているかを見る指標です。0.5付近がランダム相当です。
- **Log Loss:** 二値分類の確率予測が正解からどれだけ外れたかを見る指標です。小さいほど良い値ですが、サンプルデータの値を実運用精度として解釈できません。
- **Top-pick accuracy:** 各testレースでAI暫定推定勝率が最も高い馬が、実際に1着だった割合です。モデル全体を評価する唯一の指標ではありません。

## 再現済みの評価結果

固定seed `20260928` で生成したサンプルデータを再生成・再学習した結果です。GitHub上でも確認できるよう、ローカル生成物とは別に主要値をここへ記録しています。

| 項目 | 結果 |
|---|---:|
| train期間 | 2024-01-06 ～ 2024-05-26 |
| test期間 | 2024-05-29 ～ 2024-07-13 |
| testレース数 | 48 |
| test出走馬数 | 494 |
| ROC-AUC | 0.802597 |
| Log Loss | 0.282764 |
| Top-pick accuracy | 0.458333 |

### オッズ基準線との比較（Baseline comparison）

同じtestデータに対する独立検証では、単純なオッズ基準線は次の結果でした。

| 指標 | LightGBMモデル | 単純オッズ基準線 |
|---|---:|---:|
| ROC-AUC | 0.802597 | 0.850430（`1 / odds`） |
| Log Loss | 0.282764 | 0.238296（レース内正規化したオッズ由来確率） |
| Top-pick accuracy | 0.458333 | 0.458333（1番人気勝率） |

現在のv0.1では、単純なオッズ基準線を上回っていません。このPrototypeの目的は実競馬での高精度予測を主張することではなく、時系列train/test分割、リーク防止、再現可能な学習パイプライン、AI推定値と市場情報の比較、独立評価、UIまで含む一連のPrototypeを実装・検証できることを示す点にあります。

## AI暫定推定勝率について

LightGBMは各出走馬を1行ずつ独立した二値分類として扱い、`predict_proba` でraw probabilityを出します。その値は、そのまま厳密な競走勝率とは断定できません。

画面の「AI暫定推定勝率」は、**LightGBMのraw probabilityを `race_id` 単位で正規化し、レース内合計を100%にした表示用の暫定値**です。市場暗黙確率は `1 / 予想時点オッズ`、`value_gap` はAI暫定推定勝率との差です。

`1 / odds` は競馬市場の控除を含むため、同一レース内の合計が100%を超える場合があります（今回のtestデータでは平均約118.2%）。一方、AI暫定推定勝率は合計100%へ正規化されています。そのため `value_gap` は厳密に同一基準へ正規化した比較値ではなく、構造上マイナス方向へ偏りやすい参考値です。`value_gap > 0` であっても、購入推奨や購入した事実を意味しません。

なお、予想時点オッズは結果確定前に利用できる情報としてモデル特徴量にも含めています。そのため、AIと市場の比較は完全に独立した比較ではありません。

## 未来情報を使わない設計

- 特徴量は `src/train_model.py` のリストで明示的に許可しています。
- `finish_position` と `target_win` は特徴量から除外しています。
- 過去成績の特徴量は、サンプル生成時にそのレースより前の結果だけから計算し、レース終了後に履歴を更新します。
- `race_date` の古い期間だけをtrain、新しい期間だけをtestに使い、ランダム分割は行いません。
- 1レースの結果を見て予想ルールを自動変更する処理はありません。

## ファイル構成

```text
keiba-ai-prototype/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ app.py
├─ data/
│  └─ sample_races.csv
├─ src/
│  ├─ generate_sample_data.py
│  └─ train_model.py
└─ artifacts/
   ├─ metrics.json          # 学習時にローカル生成（Git管理対象外）
   ├─ model.joblib          # 学習時にローカル生成（Git管理対象外）
   └─ test_predictions.csv  # 学習時にローカル生成（Git管理対象外）
```

`artifacts/` の3ファイルは `src/train_model.py` の実行時に生成され、`.gitignore` によりGit管理対象外です。公開リポジトリをcloneした直後には存在しないため、上記の手順でデータ生成と学習を行ってください。主要な評価結果はこのREADMEにも記録しています。

## 今後の拡張

データ生成、学習、表示を分離しているため、将来は `data/sample_races.csv` と同じ役割の入力をJRA-VAN由来のデータ変換処理へ差し替えられます。その際も、予想時点で取得できる列だけを特徴量にし、結果情報と時系列境界を厳密に管理する必要があります。v0.1ではJRA-VAN / JV-Linkを導入・接続していません。
