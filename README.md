# 📊💱 CPI×FX 統合経済分析ダッシュボード

**為替とインフレの相互作用を包括的に分析する Snowflake Native App**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=for-the-badge&logo=snowflake&logoColor=white)](https://www.snowflake.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

## 🌟 概要

このダッシュボードは、**Consumer Price Index (CPI)** と **Foreign Exchange (FX)** データの相互関係を分析する高度な経済分析ツールです。Snowflake の豊富な金融データと Cortex AI を活用し、為替変動がインフレに与える影響を多角的に分析します。

## ✨ 主な機能

### 📈 5つの分析モジュール

1. **🔗 相関分析**
   - FX変動とCPI項目間のラグ相関分析
   - コレログラムによる視覚的表現
   - 統計的有意性の検定

2. **📊 弾力性分析**
   - 為替変動がCPI各項目に与える弾力性計算
   - 線形回帰による影響度測定
   - CPIカテゴリ別の感応度比較

3. **🌊 レジーム分析**
   - ボラティリティレジームの自動判定
   - ATR・ボラティリティ指標による市場環境分析
   - 高/低ボラティリティ期の識別

4. **🌍 REER/NEER分析**
   - 実効為替レート（REER）・名目実効為替レート（NEER）の計算
   - 複数通貨ペアによる包括的分析
   - インフレ圧力の評価

5. **📋 統合ダッシュボード**
   - 主要KPIの一覧表示
   - 複合チャートによる総合分析
   - リアルタイム市場状況の把握

### 🤖 AI分析機能

- **Snowflake Cortex AI** による高度な経済分析
- 複数のAIモデル対応（Claude-4-Sonnet, Llama4-Maverick, Mistral-Large2）
- 各分析タブ専用のAI解釈とインサイト生成
- 投資戦略・政策提言の自動生成

## 🛠️ 技術スタック

### データソース
- **Snowflake Marketplace**: 
  - Bureau of Labor Statistics (BLS) CPI データ
  - Cybersyn FX レートデータ
  - リアルタイム経済指標

### フレームワーク・ライブラリ
- **Streamlit**: インタラクティブWebアプリケーション
- **Plotly**: 高度なデータ可視化
- **Pandas/NumPy**: データ処理・数値計算
- **Scikit-learn**: 機械学習・統計分析
- **SciPy**: 統計的検定

### AI・分析
- **Snowflake Cortex**: AI分析エンジン
- **線形回帰**: 弾力性分析
- **相関分析**: ラグ効果測定
- **時系列分析**: レジーム判定

## 📋 前提条件

- Snowflake アカウント（Native App 対応）
- 以下のデータベースへのアクセス権限:
  - `FINANCE__ECONOMICS.CYBERSYN.BUREAU_OF_LABOR_STATISTICS_PRICE_TIMESERIES`
  - `FINANCE__ECONOMICS.CYBERSYN.BUREAU_OF_LABOR_STATISTICS_PRICE_ATTRIBUTES`
  - `FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES`
- Snowflake Cortex AI 機能の有効化

## 🚀 セットアップ・実行方法

### 1. リポジトリのクローン
```bash
git clone https://github.com/kmotokubota/streamlit-cpi-fx-dashboard.git
cd streamlit-cpi-fx-dashboard
```

### 2. 依存関係のインストール
```bash
pip install streamlit pandas numpy plotly scikit-learn scipy
```

### 3. Snowflake Native App での実行
```bash
# Snowflake環境で直接実行
streamlit run integrated_cpi_fx_dashboard.py
```

## 📊 使用方法

### 基本操作

1. **分析設定**
   - サイドバーで分析期間を設定（最大5年間）
   - 通貨ペアを選択（USD/JPY, EUR/USD, GBP/USD, USD/CHF）
   - AIモデルを選択

2. **データ分析**
   - 各タブで専門的な分析を実行
   - インタラクティブなチャートで詳細確認
   - AI分析ボタンで深い洞察を取得

3. **結果の解釈**
   - 統計的指標の確認
   - AI生成レポートの活用
   - 投資・政策判断への応用

### 分析例

#### 相関分析
```
通貨ペア: USD/JPY
CPIカテゴリ: All items
結果: 3ヶ月ラグで相関係数 0.45 (p < 0.05)
→ 円安は3ヶ月後にインフレ圧力として現れる傾向
```

#### 弾力性分析
```
Energy: 弾力性係数 0.12 → USD/JPY 1%上昇で12bp のインフレ圧力
Food: 弾力性係数 0.08 → USD/JPY 1%上昇で8bp のインフレ圧力
```

## 📈 分析指標の詳細

### 相関分析
- **ラグ相関**: 0-6ヶ月のタイムラグを考慮した相関係数
- **統計的有意性**: p値による信頼性評価
- **経済的解釈**: 為替変動の波及メカニズム分析

### 弾力性分析
- **弾力性係数**: 為替1%変動に対するCPI変化率
- **決定係数(R²)**: モデルの説明力
- **影響度(bp)**: ベーシスポイント単位での定量評価

### ボラティリティ指標
- **ATR**: Average True Range（14日移動平均）
- **ボラティリティ**: 20日ローリング標準偏差（年率換算）
- **レジーム判定**: 閾値ベースの自動分類

## 🎯 活用シーン

### 金融機関
- **リスク管理**: 為替変動によるインフレリスクの定量評価
- **投資戦略**: 通貨・債券投資の意思決定支援
- **ヘッジ戦略**: 為替ヘッジの最適化

### 中央銀行・政策当局
- **金融政策**: 為替変動の物価への波及効果分析
- **インフレ予測**: 為替要因を考慮したCPI予測
- **政策効果測定**: 介入・政策変更の影響評価

### 企業・投資家
- **価格戦略**: 為替変動を考慮した価格設定
- **調達戦略**: 輸入コスト変動の予測・対策
- **投資判断**: マクロ経済環境の総合評価

## 🔧 カスタマイズ

### 通貨ペアの追加
```python
# currency_pair選択肢に新しい通貨ペアを追加
currency_pair = st.selectbox(
    "通貨ペア",
    ["USD/JPY", "EUR/USD", "GBP/USD", "USD/CHF", "AUD/USD"],  # 新通貨追加
    index=0
)
```

### 分析期間の拡張
```python
# より長期間の分析に対応
start_date = st.date_input(
    "分析開始日",
    value=today - timedelta(days=365 * 10),  # 10年間に拡張
    min_value=today - timedelta(days=365 * 10),
    max_value=today
)
```

### 新しいCPI項目の追加
```python
# クエリ内のPRODUCT条件を拡張
AND attr.PRODUCT IN (
    'All items', 'All items less food and energy', 'Food', 'Energy',
    'Services less energy services', 'Commodities less food and energy commodities',
    'Housing', 'Transportation'  # 新項目追加
)
```

## 📚 参考資料

- [Snowflake Marketplace - Economic Data](https://app.snowflake.com/marketplace/listings)
- [Bureau of Labor Statistics CPI Documentation](https://www.bls.gov/cpi/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)

## 🤝 コントリビューション

プルリクエストや Issue の報告を歓迎します。以下の点にご協力ください：

1. **バグ報告**: 詳細な再現手順と環境情報
2. **機能提案**: 具体的な使用ケースと実装案
3. **コード改善**: パフォーマンス向上・可読性改善

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 👨‍💻 作成者

**Kenta Motokubota**
- GitHub: [@kmotokubota](https://github.com/kmotokubota)
- 所属: Snowflake Inc.

---

**🚀 Powered by Snowflake Cortex AI ❄️**

このダッシュボードは、Snowflake の Native App プラットフォーム上で動作し、リアルタイムな経済分析と AI による洞察を提供します。