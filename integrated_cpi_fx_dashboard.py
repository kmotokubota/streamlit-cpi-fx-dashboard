import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# --- ページ設定 ---
st.set_page_config(
    page_title="CPI×FX 統合経済分析ダッシュボード",
    page_icon="📊💱",
    layout="wide",
)

# Snowflakeセッションの取得
try:
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
    SNOWFLAKE_AVAILABLE = True
except Exception:
    SNOWFLAKE_AVAILABLE = False

# --- カスタムCSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-top: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #D1D5DB;
        padding-bottom: 0.5rem;
    }
    .alert-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
    }
    .regime-box {
        background-color: #DBEAFE;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .correlation-box {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- データ取得関数 ---
@st.cache_data(ttl=600)
def load_cpi_data(start_date, end_date):
    """CPIデータを取得"""
    if not SNOWFLAKE_AVAILABLE:
        return pd.DataFrame()
    
    extended_start_date = pd.to_datetime(start_date) - pd.DateOffset(months=13)
    
    query = f"""
    SELECT
        ts.DATE,
        ts.VALUE,
        attr.VARIABLE,
        attr.VARIABLE_NAME,
        attr.PRODUCT,
        attr.SEASONALLY_ADJUSTED
    FROM FINANCE__ECONOMICS.CYBERSYN.BUREAU_OF_LABOR_STATISTICS_PRICE_TIMESERIES ts
    JOIN FINANCE__ECONOMICS.CYBERSYN.BUREAU_OF_LABOR_STATISTICS_PRICE_ATTRIBUTES attr
      ON ts.VARIABLE = attr.VARIABLE
    WHERE attr.REPORT = 'Consumer Price Index'
      AND attr.FREQUENCY = 'Monthly'
      AND attr.SEASONALLY_ADJUSTED = TRUE
      AND ts.DATE BETWEEN '{extended_start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
      AND attr.PRODUCT IN (
          'All items', 'All items less food and energy', 'Food', 'Energy',
          'Services less energy services', 'Commodities less food and energy commodities'
      )
      AND ts.VALUE IS NOT NULL
    ORDER BY attr.PRODUCT, ts.DATE
    """
    
    try:
        df = session.sql(query).to_pandas()
        if not df.empty:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.sort_values(by=['PRODUCT', 'DATE'])
            # YoY と MoM を計算
            df['YoY_Change'] = df.groupby('PRODUCT')['VALUE'].pct_change(periods=12) * 100
            df['MoM_Change'] = df.groupby('PRODUCT')['VALUE'].pct_change(periods=1) * 100
        return df
    except Exception as e:
        st.error(f"CPIデータの取得に失敗しました: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_fx_data(start_date, end_date, base_currency='USD', quote_currency='JPY'):
    """FXデータを取得"""
    if not SNOWFLAKE_AVAILABLE:
        return pd.DataFrame()
        
    query = f"""
    SELECT
        DATE,
        VALUE AS EXCHANGE_RATE,
        VARIABLE_NAME,
        BASE_CURRENCY_ID,
        QUOTE_CURRENCY_ID
    FROM
        FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES
    WHERE
        BASE_CURRENCY_ID = '{base_currency}'
        AND QUOTE_CURRENCY_ID = '{quote_currency}'
        AND DATE >= '{start_date}'
        AND DATE <= '{end_date}'
    ORDER BY
        DATE
    """
    
    try:
        df = session.sql(query).to_pandas()
        if not df.empty:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.sort_values('DATE')
            # リターン計算
            df['Daily_Return'] = df['EXCHANGE_RATE'].pct_change() * 100
            df['Monthly_Return'] = df['EXCHANGE_RATE'].pct_change(periods=21) * 100  # 約1ヶ月
            # ボラティリティ指標
            df['ATR'] = df['Daily_Return'].abs().rolling(window=14).mean()
            df['Volatility_20D'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        return df
    except Exception as e:
        st.error(f"FXデータの取得に失敗しました: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_multiple_fx_data_for_reer(start_date, end_date):
    """REER/NEER計算用の複数通貨ペアデータを取得"""
    if not SNOWFLAKE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        # まず利用可能な通貨ペアを確認
        available_pairs_query = """
        SELECT DISTINCT 
            BASE_CURRENCY_ID,
            QUOTE_CURRENCY_ID,
            CONCAT(BASE_CURRENCY_ID, '/', QUOTE_CURRENCY_ID) AS CURRENCY_PAIR
        FROM FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES
        WHERE BASE_CURRENCY_ID = 'USD'
        AND DATE >= CURRENT_DATE - 30
        ORDER BY QUOTE_CURRENCY_ID
        """
        
        available_df = session.sql(available_pairs_query).to_pandas()
        
        if available_df.empty:
            st.warning("利用可能な通貨ペアが見つかりませんでした。")
            return pd.DataFrame()
        
        # 利用可能な通貨から主要通貨を選択
        available_currencies = available_df['QUOTE_CURRENCY_ID'].tolist()
        major_currencies = ['EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD']
        target_currencies = [curr for curr in major_currencies if curr in available_currencies]
        
        if not target_currencies:
            # 主要通貨が見つからない場合は、利用可能な最初の6通貨を使用
            target_currencies = available_currencies[:6]
        
        # データ取得クエリ
        currencies_str = "', '".join(target_currencies)
        query = f"""
        SELECT
            DATE,
            VALUE AS EXCHANGE_RATE,
            BASE_CURRENCY_ID,
            QUOTE_CURRENCY_ID,
            CONCAT(BASE_CURRENCY_ID, '/', QUOTE_CURRENCY_ID) AS CURRENCY_PAIR
        FROM
            FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES
        WHERE
            BASE_CURRENCY_ID = 'USD'
            AND QUOTE_CURRENCY_ID IN ('{currencies_str}')
            AND DATE >= '{start_date}'
            AND DATE <= '{end_date}'
            AND VALUE IS NOT NULL
        ORDER BY
            CURRENCY_PAIR, DATE
        """
        
        df = session.sql(query).to_pandas()
        
        if not df.empty:
            df['DATE'] = pd.to_datetime(df['DATE'])
            st.sidebar.success(f"✅ {len(target_currencies)}通貨のデータを取得: {', '.join(target_currencies)}")
        else:
            st.sidebar.warning("⚠️ 指定期間のデータが見つかりませんでした")
            
        return df
        
    except Exception as e:
        st.error(f"複数通貨データの取得に失敗しました: {e}")
        st.sidebar.error(f"❌ データ取得エラー: {str(e)}")
        return pd.DataFrame()

def calculate_reer_neer(fx_data):
    """REER/NEER指数を計算（簡易版）"""
    if fx_data.empty:
        return pd.DataFrame()
    
    try:
        # 通貨ペア別にピボット
        pivot_df = fx_data.pivot(index='DATE', columns='CURRENCY_PAIR', values='EXCHANGE_RATE')
        
        if pivot_df.empty:
            return pd.DataFrame()
        
        # 利用可能な通貨ペアを取得
        available_pairs = pivot_df.columns.tolist()
        
        # 動的な貿易ウェイト（利用可能な通貨に基づいて均等配分）
        num_pairs = len(available_pairs)
        if num_pairs == 0:
            return pd.DataFrame()
        
        equal_weight = 1.0 / num_pairs
        trade_weights = {pair: equal_weight for pair in available_pairs}
        
        # 優先ウェイト（利用可能な場合）
        priority_weights = {
            'USD/EUR': 0.25,
            'USD/JPY': 0.20,
            'USD/GBP': 0.15,
            'USD/CHF': 0.10,
            'USD/CAD': 0.15,
            'USD/AUD': 0.15
        }
        
        # 優先ウェイトが利用可能な場合は使用
        total_priority_weight = sum(priority_weights.get(pair, 0) for pair in available_pairs)
        if total_priority_weight > 0:
            # 正規化
            for pair in available_pairs:
                if pair in priority_weights:
                    trade_weights[pair] = priority_weights[pair] / total_priority_weight
                else:
                    trade_weights[pair] = 0.1 / total_priority_weight  # 小さなウェイト
        
        # NEER計算（名目実効為替レート）
        neer_components = {}
        for pair in available_pairs:
            pair_data = pivot_df[pair].dropna()
            if len(pair_data) > 0:
                # 基準日（最初の日）を100として指数化
                base_value = pair_data.iloc[0]
                if base_value != 0:
                    weight = trade_weights.get(pair, equal_weight)
                    neer_components[pair] = (pivot_df[pair] / base_value * 100) * weight
        
        if not neer_components:
            return pd.DataFrame()
        
        neer_df = pd.DataFrame(neer_components)
        neer_index = neer_df.sum(axis=1).dropna()
        
        if neer_index.empty:
            return pd.DataFrame()
        
        # REER計算（実質実効為替レート）- 簡易版
        # 実際の実装では各国のCPIデータが必要
        reer_index = neer_index  # 簡易版では同じ値を使用
        
        result_df = pd.DataFrame({
            'DATE': neer_index.index,
            'NEER': neer_index.values,
            'REER': reer_index.values
        })
        
        return result_df
        
    except Exception as e:
        st.error(f"REER/NEER計算でエラーが発生しました: {e}")
        return pd.DataFrame()

# --- 分析関数 ---
def calculate_lag_correlation(cpi_series, fx_series, max_lag=6):
    """ラグ相関を計算"""
    correlations = {}
    
    try:
        # データの前処理
        cpi_clean = cpi_series.dropna()
        fx_clean = fx_series.dropna()
        
        if len(cpi_clean) == 0 or len(fx_clean) == 0:
            return correlations
        
        # 月次データに変換
        cpi_monthly = cpi_clean.resample('M').last().dropna()
        fx_monthly = fx_clean.resample('M').last().dropna()
        
        if len(cpi_monthly) == 0 or len(fx_monthly) == 0:
            return correlations
        
        # 共通期間を取得
        common_dates = cpi_monthly.index.intersection(fx_monthly.index)
        
        if len(common_dates) < 3:  # 最低3データポイント必要
            return correlations
            
        cpi_common = cpi_monthly.loc[common_dates]
        fx_common = fx_monthly.loc[common_dates]
        
        for lag in range(max_lag + 1):
            try:
                if lag == 0:
                    # 同時相関
                    if len(cpi_common) > 2 and len(fx_common) > 2:
                        # データサイズを確認
                        cpi_values = cpi_common.values
                        fx_values = fx_common.values
                        
                        if len(cpi_values) == len(fx_values) and len(cpi_values) > 2:
                            corr, p_value = pearsonr(cpi_values, fx_values)
                            if not (np.isnan(corr) or np.isnan(p_value)):
                                correlations[f'Lag_{lag}'] = {'correlation': corr, 'p_value': p_value}
                else:
                    # ラグ相関（FXがCPIより先行）
                    if len(fx_common) > lag + 2:  # ラグ分を考慮した最小データ数
                        fx_lagged = fx_common.shift(lag).dropna()
                        
                        if len(fx_lagged) > 2:
                            # 共通インデックスを再取得
                            common_idx = fx_lagged.index.intersection(cpi_common.index)
                            
                            if len(common_idx) > 2:
                                fx_values = fx_lagged.loc[common_idx].values
                                cpi_values = cpi_common.loc[common_idx].values
                                
                                if len(fx_values) == len(cpi_values) and len(fx_values) > 2:
                                    corr, p_value = pearsonr(fx_values, cpi_values)
                                    if not (np.isnan(corr) or np.isnan(p_value)):
                                        correlations[f'Lag_{lag}'] = {'correlation': corr, 'p_value': p_value}
            except Exception as e:
                # 個別のラグ計算でエラーが発生した場合はスキップ
                continue
    
    except Exception as e:
        # 全体的なエラーの場合は空の辞書を返す
        st.warning(f"相関計算でエラーが発生しました: {str(e)}")
        return {}
    
    return correlations

def calculate_fx_elasticity(cpi_data, fx_data):
    """FX変動がCPIに与える弾力性を計算"""
    elasticity_results = {}
    
    try:
        # CPIカテゴリ別に弾力性を計算
        cpi_categories = cpi_data['PRODUCT'].unique()
        
        for category in cpi_categories:
            try:
                category_data = cpi_data[cpi_data['PRODUCT'] == category].copy()
                
                if category_data.empty:
                    continue
                
                # 月次データに変換
                category_monthly = category_data.set_index('DATE')['MoM_Change'].resample('M').last().dropna()
                fx_monthly = fx_data.set_index('DATE')['Monthly_Return'].resample('M').last().dropna()
                
                if len(category_monthly) == 0 or len(fx_monthly) == 0:
                    continue
                
                # 共通期間を取得
                common_dates = category_monthly.index.intersection(fx_monthly.index)
                
                if len(common_dates) > 10:  # 最低10データポイント必要
                    cpi_values = category_monthly.loc[common_dates].dropna()
                    fx_values = fx_monthly.loc[common_dates].dropna()
                    
                    # 共通のインデックスを再取得
                    final_common = cpi_values.index.intersection(fx_values.index)
                    
                    if len(final_common) > 5:
                        X_values = fx_values.loc[final_common].values
                        y_values = cpi_values.loc[final_common].values
                        
                        # データの有効性をチェック
                        if len(X_values) == len(y_values) and len(X_values) > 5:
                            # NaNや無限大値をチェック
                            valid_mask = ~(np.isnan(X_values) | np.isnan(y_values) | 
                                         np.isinf(X_values) | np.isinf(y_values))
                            
                            if np.sum(valid_mask) > 5:
                                X_clean = X_values[valid_mask].reshape(-1, 1)
                                y_clean = y_values[valid_mask]
                                
                                # 線形回帰
                                model = LinearRegression()
                                model.fit(X_clean, y_clean)
                                
                                r_squared = model.score(X_clean, y_clean)
                                
                                # 結果の有効性をチェック
                                if not (np.isnan(model.coef_[0]) or np.isnan(r_squared)):
                                    elasticity_results[category] = {
                                        'coefficient': model.coef_[0],
                                        'r_squared': r_squared,
                                        'impact_1pct': model.coef_[0] * 1.0  # 1%変動時の影響
                                    }
            except Exception as e:
                # 個別カテゴリでエラーが発生した場合はスキップ
                continue
    
    except Exception as e:
        st.warning(f"弾力性計算でエラーが発生しました: {str(e)}")
        return {}
    
    return elasticity_results

def detect_volatility_regime(fx_data, atr_threshold=1.5, volatility_threshold=15):
    """ボラティリティレジームを判定"""
    if fx_data.empty:
        return "データ不足"
    
    latest_atr = fx_data['ATR'].iloc[-1] if not pd.isna(fx_data['ATR'].iloc[-1]) else 0
    latest_vol = fx_data['Volatility_20D'].iloc[-1] if not pd.isna(fx_data['Volatility_20D'].iloc[-1]) else 0
    
    if latest_atr > atr_threshold or latest_vol > volatility_threshold:
        return "高ボラティリティ期"
    else:
        return "低ボラティリティ期"


# --- チャート作成関数 ---
def create_correlogram_chart(correlations):
    """コレログラムチャートを作成"""
    if not correlations:
        fig = go.Figure()
        fig.add_annotation(
            text="相関データが不足しています",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    try:
        # 利用可能なラグのリストを作成
        available_lags = []
        corr_values = []
        p_values = []
        
        for key in sorted(correlations.keys()):
            if key.startswith('Lag_'):
                lag_num = int(key.split('_')[1])
                available_lags.append(lag_num)
                corr_values.append(correlations[key]['correlation'])
                p_values.append(correlations[key]['p_value'])
        
        if not corr_values:
            fig = go.Figure()
            fig.add_annotation(
                text="有効な相関データがありません",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # 有意性を色で表現
        colors = ['red' if p < 0.05 else 'lightblue' for p in p_values]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=available_lags,
            y=corr_values,
            marker_color=colors,
            name='相関係数',
            hovertemplate='Lag %{x}ヶ月<br>相関係数: %{y:.3f}<br>p値: %{customdata:.3f}<extra></extra>',
            customdata=p_values
        ))
        
        fig.update_layout(
            title='FX×CPI ラグ相関分析',
            xaxis_title='ラグ（ヶ月）',
            yaxis_title='相関係数',
            yaxis=dict(range=[-1, 1]),
            showlegend=False
        )
        
        # 有意水準ライン
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"チャート作成エラー: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig

def create_elasticity_table(elasticity_results):
    """弾力性テーブルを作成"""
    if not elasticity_results:
        return pd.DataFrame()
    
    table_data = []
    for category, results in elasticity_results.items():
        table_data.append({
            'CPIカテゴリ': category,
            '弾力性係数': f"{results['coefficient']:.4f}",
            'R²': f"{results['r_squared']:.3f}",
            '1%変動時の影響(bp)': f"{results['impact_1pct']*100:.1f}"
        })
    
    return pd.DataFrame(table_data)

# --- AI分析関数 ---
def run_ai_analysis(analysis_type, data_context, ai_model="claude-4-sonnet"):
    """各タブ用のAI分析を実行"""
    if not SNOWFLAKE_AVAILABLE:
        return "Snowflakeセッションが利用できません。"
    
    try:
        # 分析タイプ別のプロンプト作成
        if analysis_type == "correlation":
            prompt = f"""
            # 指示
            あなたは金融市場のエキスパートです。以下のFX×CPI相関分析結果を基に、専門的な解釈と投資・政策への示唆を日本語で提供してください。

            # 分析データ
            {data_context}

            # 回答に含めるべき内容
            1. 相関関係の経済的意味と背景
            2. ラグ効果の解釈（なぜそのタイミングで影響が現れるか）
            3. 投資戦略への示唆
            4. 金融政策への影響
            5. 注意すべきリスク要因

            # 出力形式
            - 簡潔で実用的な分析
            - 専門用語を適切に使用
            - 具体的なアクションプランを含める
            """
            
        elif analysis_type == "elasticity":
            prompt = f"""
            # 指示
            あなたは経済分析の専門家です。以下の為替弾力性分析結果を基に、インフレ圧力と経済政策への示唆を日本語で提供してください。

            # 分析データ
            {data_context}

            # 回答に含めるべき内容
            1. 各CPIカテゴリの為替感応度の解釈
            2. インフレ圧力の波及メカニズム
            3. 中央銀行の政策判断への影響
            4. 企業の価格戦略への示唆
            5. 今後の注目ポイント

            # 出力形式
            - データに基づいた客観的分析
            - 政策担当者向けの実用的な洞察
            - リスクシナリオの提示
            """
            
        elif analysis_type == "regime":
            prompt = f"""
            # 指示
            あなたは市場リスク管理の専門家です。以下のボラティリティレジーム分析を基に、現在の市場環境と対応策を日本語で提供してください。

            # 分析データ
            {data_context}

            # 回答に含めるべき内容
            1. 現在のボラティリティレジームの特徴
            2. このレジームが続く可能性と転換点
            3. リスク管理上の注意点
            4. 投資戦略の調整方針
            5. 監視すべき指標

            # 出力形式
            - リスク管理者向けの実践的アドバイス
            - 市場環境の変化への対応策
            - 定量的な判断基準の提示
            """
            
        elif analysis_type == "reer":
            prompt = f"""
            # 指示
            あなたは国際金融の専門家です。以下の実効為替レート分析を基に、マクロ経済への影響と政策示唆を日本語で提供してください。

            # 分析データ
            {data_context}

            # 回答に含めるべき内容
            1. 実効為替レートの変動要因
            2. 貿易収支・経常収支への影響
            3. 国内インフレ圧力への波及
            4. 競争力への影響評価
            5. 政策対応の選択肢

            # 出力形式
            - マクロ経済政策担当者向けの分析
            - 国際競争力の観点からの評価
            - 中長期的な影響の予測
            """
        else:
            prompt = f"""
            以下のデータを分析して、金融市場への示唆を日本語で提供してください：
            {data_context}
            """

        # AI_COMPLETE実行
        safe_prompt = prompt.replace("'", "''")
        query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{ai_model}', '{safe_prompt}') AS analysis"
        result = session.sql(query).to_pandas()
        
        raw_analysis = result['ANALYSIS'].iloc[0]
        
        # クリーンアップ処理
        formatted_analysis = raw_analysis.replace('\\n', '\n')
        
        # 先頭と末尾のクォーテーションを削除
        if formatted_analysis.startswith('"""') and formatted_analysis.endswith('"""'):
            formatted_analysis = formatted_analysis[3:-3]
        elif formatted_analysis.startswith('"') and formatted_analysis.endswith('"'):
            formatted_analysis = formatted_analysis[1:-1]
        
        # テキストをクリーンアップ
        lines = formatted_analysis.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # マークダウンヘッダーの調整
            if line.startswith('## '):
                line = '### ' + line[3:]
            elif line.startswith('# '):
                line = '## ' + line[2:]
            
            # 空行が連続する場合は1つだけ残す
            if line == '' and len(cleaned_lines) > 0 and cleaned_lines[-1] == '':
                continue
                
            cleaned_lines.append(line)
        
        # 先頭と末尾の空行を削除
        while cleaned_lines and cleaned_lines[0] == '':
            cleaned_lines.pop(0)
        while cleaned_lines and cleaned_lines[-1] == '':
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)

    except Exception as e:
        return f"AI分析でエラーが発生しました: {str(e)}"

# --- メインアプリケーション ---
def main():
    # カスタムCSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 16px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
    .ai-button-container {
        text-align: center;
        margin: 15px 0;
    }
    .ai-result-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .ai-result-title {
        color: #495057;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 2px solid #6c757d;
        padding-bottom: 8px;
    }
    div.stButton > button:first-child {
        background-color: #f0f2f6;
        color: #1f77b4;
        border: 2px solid #1f77b4;
        border-radius: 8px;
        padding: 15px 20px;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin: 10px 0;
        display: block;
        text-align: center;
    }
    div.stButton > button:first-child:hover {
        background-color: #1f77b4;
        color: white;
        transform: none;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
    }
    div.stButton {
        width: 100%;
        margin: 0;
    }
    .ai-button-container {
        width: 100%;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">📊💱 CPI×FX 統合経済分析ダッシュボード</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">為替とインフレの相互作用を包括的に分析 | Powered by Snowflake Cortex AI ❄️</div>', unsafe_allow_html=True)

    if not SNOWFLAKE_AVAILABLE:
        st.error("⚠️ Snowflakeセッションに接続できません。Snowflake Native App環境で実行してください。")
        st.stop()

    # --- サイドバー設定 ---
    with st.sidebar:
        st.header("分析設定")
        
        # 期間設定
        today = datetime.now()
        start_date = st.date_input(
            "分析開始日",
            value=today - timedelta(days=365 * 2),
            min_value=today - timedelta(days=365 * 5),
            max_value=today
        )
        end_date = st.date_input("分析終了日", value=today, max_value=today)
        
        # 通貨ペア選択
        currency_pair = st.selectbox(
            "通貨ペア",
            ["USD/JPY", "EUR/USD", "GBP/USD", "USD/CHF"],
            index=0
        )
        
        base_currency, quote_currency = currency_pair.split('/')
        
        # AI分析設定
        st.markdown("---")
        st.subheader("🤖 AI分析設定")
        
        # グローバルAIモデル設定
        if 'selected_ai_model' not in st.session_state:
            st.session_state.selected_ai_model = "claude-4-sonnet"
        
        st.session_state.selected_ai_model = st.selectbox(
            "AIモデル選択",
            ["claude-4-sonnet", "llama4-maverick", "mistral-large2"],
            index=["claude-4-sonnet", "llama4-maverick", "mistral-large2"].index(st.session_state.selected_ai_model),
            help="分析に使用するAIモデルを選択してください"
        )

    # --- データ読み込み ---
    with st.spinner("❄️ Snowflakeからデータを取得中..."):
        cpi_data = load_cpi_data(start_date, end_date)
        fx_data = load_fx_data(start_date, end_date, base_currency, quote_currency)

    if cpi_data.empty or fx_data.empty:
        st.error("データが取得できませんでした。期間を変更するか、管理者にお問い合わせください。")
        st.stop()

    # --- タブ構成 ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 相関分析", 
        "📈 弾力性分析", 
        "🌊 レジーム分析", 
        "🌍 REER/NEER", 
        "📊 統合ダッシュボード"
    ])
    
    # タブ切り替えの検知（簡易版）
    # 実際のタブ検知は困難なため、各タブでcurrent_tabを設定

    with tab1:
        st.session_state.current_tab = 'correlation'
        st.markdown('<div class="section-title">FX×CPI ラグ相関分析</div>', unsafe_allow_html=True)
        
        # CPIカテゴリ選択
        selected_cpi_category = st.selectbox(
            "CPIカテゴリを選択",
            cpi_data['PRODUCT'].unique(),
            key="cpi_category_corr"
        )
        
        # 相関分析実行
        cpi_category_data = cpi_data[cpi_data['PRODUCT'] == selected_cpi_category]
        
        if not cpi_category_data.empty:
            # MoM変化率で相関分析
            cpi_series = cpi_category_data.set_index('DATE')['MoM_Change']
            fx_series = fx_data.set_index('DATE')['Monthly_Return']
            
            correlations = calculate_lag_correlation(cpi_series, fx_series)
            
            if correlations:
                # コレログラム表示
                correlogram_chart = create_correlogram_chart(correlations)
                st.plotly_chart(correlogram_chart, use_container_width=True)
                
                # 相関結果の解釈
                max_corr_lag = max(correlations.keys(), key=lambda x: abs(correlations[x]['correlation']))
                max_corr_value = correlations[max_corr_lag]['correlation']
                max_corr_lag_num = int(max_corr_lag.split('_')[1])
                
                st.markdown(f"""
                <div class="correlation-box">
                    <h4>📊 相関分析結果</h4>
                    <p><strong>最大相関:</strong> {max_corr_lag_num}ヶ月ラグで相関係数 {max_corr_value:.3f}</p>
                    <p><strong>解釈:</strong> {currency_pair}の変動は{selected_cpi_category}に対して
                    {max_corr_lag_num}ヶ月後に{'正の' if max_corr_value > 0 else '負の'}影響を与える傾向</p>
                </div>
                """, unsafe_allow_html=True)
                
                # AI分析実行ボタン
                st.markdown('<div class="ai-button-container">', unsafe_allow_html=True)
                if st.button("🤖 AI相関分析を実行", key="correlation_ai_button"):
                    # 相関分析用のデータコンテキスト作成
                    correlation_summary = []
                    for key, value in correlations.items():
                        lag_num = key.split('_')[1]
                        correlation_summary.append(f"Lag {lag_num}: 相関係数 {value['correlation']:.3f} (p値: {value['p_value']:.3f})")
                    
                    correlation_context = f"""
                    通貨ペア: {currency_pair}
                    CPIカテゴリ: {selected_cpi_category}
                    分析期間: {start_date} ～ {end_date}
                    
                    ラグ相関結果:
                    {chr(10).join(correlation_summary)}
                    
                    最大相関: {max_corr_lag_num}ヶ月ラグで相関係数 {max_corr_value:.3f}
                    """
                    
                    with st.spinner("🤖 AI分析中..."):
                        ai_analysis = run_ai_analysis("correlation", correlation_context, st.session_state.selected_ai_model)
                        st.session_state.correlation_ai_analysis = ai_analysis
                        st.success("✅ AI分析完了！")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # AI分析結果表示
                if 'correlation_ai_analysis' in st.session_state:
                    st.markdown("""
                    <div class="ai-result-box">
                        <div class="ai-result-title">🧠 AI分析結果</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(st.session_state.correlation_ai_analysis)
                else:
                    st.info("💡 上記の「🚀 AI相関分析を実行」ボタンで詳細な分析を開始できます")

    with tab2:
        st.session_state.current_tab = 'elasticity'
        st.markdown('<div class="section-title">為替変動のCPI弾力性分析</div>', unsafe_allow_html=True)
        
        # 弾力性計算
        elasticity_results = calculate_fx_elasticity(cpi_data, fx_data)
        
        if elasticity_results:
            # 弾力性テーブル表示
            elasticity_df = create_elasticity_table(elasticity_results)
            st.dataframe(elasticity_df, use_container_width=True)
            
            # 弾力性チャート
            categories = list(elasticity_results.keys())
            impacts = [elasticity_results[cat]['impact_1pct']*100 for cat in categories]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=categories,
                y=impacts,
                marker_color='lightcoral',
                name='1%変動時の影響(bp)'
            ))
            
            fig.update_layout(
                title=f'{currency_pair} 1%変動時のCPI項目別影響',
                xaxis_title='CPIカテゴリ',
                yaxis_title='影響 (ベーシスポイント)',
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # AI分析実行ボタン
            st.markdown('<div class="ai-button-container">', unsafe_allow_html=True)
            if st.button("🤖 AI弾力性分析を実行", key="elasticity_ai_button"):
                # 弾力性分析用のデータコンテキスト作成
                elasticity_summary = []
                for category, results in elasticity_results.items():
                    elasticity_summary.append(f"{category}: 弾力性係数 {results['coefficient']:.4f}, R² {results['r_squared']:.3f}, 1%変動時影響 {results['impact_1pct']*100:.1f}bp")
                
                elasticity_context = f"""
                通貨ペア: {currency_pair}
                分析期間: {start_date} ～ {end_date}
                
                弾力性分析結果:
                {chr(10).join(elasticity_summary)}
                """
                
                with st.spinner("🤖 AI分析中..."):
                    ai_analysis = run_ai_analysis("elasticity", elasticity_context, st.session_state.selected_ai_model)
                    st.session_state.elasticity_ai_analysis = ai_analysis
                    st.success("✅ AI分析完了！")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # AI分析結果表示
            if 'elasticity_ai_analysis' in st.session_state:
                st.markdown("""
                <div class="ai-result-box">
                    <div class="ai-result-title">🧠 AI分析結果</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(st.session_state.elasticity_ai_analysis)
            else:
                st.info("💡 上記の「🚀 AI弾力性分析を実行」ボタンで詳細な分析を開始できます")

    with tab3:
        st.session_state.current_tab = 'regime'
        st.markdown('<div class="section-title">ボラティリティレジーム分析</div>', unsafe_allow_html=True)
        
        # レジーム判定
        current_regime = detect_volatility_regime(fx_data)
        
        regime_color = "high" if "高ボラティリティ" in current_regime else "low"
        
        st.markdown(f"""
        <div class="regime-box">
            <h4>🌊 現在のマーケットレジーム</h4>
            <p><strong>{currency_pair}:</strong> <span style="color: {'red' if regime_color == 'high' else 'green'}; font-weight: bold;">{current_regime}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # ボラティリティ推移チャート
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'{currency_pair} 価格推移', 'ボラティリティ指標'],
            vertical_spacing=0.1
        )
        
        # 価格チャート
        fig.add_trace(
            go.Scatter(
                x=fx_data['DATE'],
                y=fx_data['EXCHANGE_RATE'],
                mode='lines',
                name='為替レート',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # ボラティリティチャート
        fig.add_trace(
            go.Scatter(
                x=fx_data['DATE'],
                y=fx_data['Volatility_20D'],
                mode='lines',
                name='20日ボラティリティ',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True, height=800)
        
        # AI分析実行ボタン
        st.markdown('<div class="ai-button-container">', unsafe_allow_html=True)
        if st.button("🤖 AIレジーム分析を実行", key="regime_ai_button"):
            # AI分析用のデータコンテキスト作成
            latest_atr = fx_data['ATR'].iloc[-1] if not pd.isna(fx_data['ATR'].iloc[-1]) else 0
            latest_vol = fx_data['Volatility_20D'].iloc[-1] if not pd.isna(fx_data['Volatility_20D'].iloc[-1]) else 0
            latest_price = fx_data['EXCHANGE_RATE'].iloc[-1]
            monthly_return = fx_data['Monthly_Return'].iloc[-1] if not pd.isna(fx_data['Monthly_Return'].iloc[-1]) else 0
            
            regime_context = f"""
            通貨ペア: {currency_pair}
            現在のレジーム: {current_regime}
            最新為替レート: {latest_price:.2f}
            月次リターン: {monthly_return:.2f}%
            ATR (14日): {latest_atr:.3f}
            ボラティリティ (20日): {latest_vol:.1f}%
            分析期間: {start_date} ～ {end_date}
            """
            
            with st.spinner("🤖 AI分析中..."):
                ai_analysis = run_ai_analysis("regime", regime_context, st.session_state.selected_ai_model)
                st.session_state.regime_ai_analysis = ai_analysis
                st.success("✅ AI分析完了！")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # AI分析結果表示
        if 'regime_ai_analysis' in st.session_state:
            st.markdown("""
            <div class="ai-result-box">
                <div class="ai-result-title">🧠 AI分析結果</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.regime_ai_analysis)
        else:
            st.info("💡 上記の「🚀 AIレジーム分析を実行」ボタンで詳細な分析を開始できます")

    with tab4:
        st.session_state.current_tab = 'reer'
        st.markdown('<div class="section-title">実効為替レート分析 (REER/NEER)</div>', unsafe_allow_html=True)
        
        # REER/NEERデータ取得
        with st.spinner("実効為替レートを計算中..."):
            multi_fx_data = load_multiple_fx_data_for_reer(start_date, end_date)
            
            if not multi_fx_data.empty:
                reer_neer_data = calculate_reer_neer(multi_fx_data)
                
                if not reer_neer_data.empty:
                    # REER/NEERチャート
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=reer_neer_data['DATE'],
                        y=reer_neer_data['NEER'],
                        mode='lines',
                        name='NEER (名目実効為替レート)',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=reer_neer_data['DATE'],
                        y=reer_neer_data['REER'],
                        mode='lines',
                        name='REER (実質実効為替レート)',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='USD実効為替レート推移',
                        xaxis_title='日付',
                        yaxis_title='指数 (基準日=100)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 実効為替レートの解釈
                    latest_neer = reer_neer_data['NEER'].iloc[-1]
                    latest_reer = reer_neer_data['REER'].iloc[-1]
                    neer_change = ((latest_neer / reer_neer_data['NEER'].iloc[0]) - 1) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "現在のNEER",
                            f"{latest_neer:.1f}",
                            f"{neer_change:+.1f}% vs 基準日"
                        )
                    
                    with col2:
                        st.metric(
                            "現在のREER", 
                            f"{latest_reer:.1f}",
                            f"{neer_change:+.1f}% vs 基準日"
                        )
                    
                    # インフレ圧力の解釈
                    if neer_change > 5:
                        pressure_type = "ディスインフレ圧力"
                        pressure_color = "blue"
                    elif neer_change < -5:
                        pressure_type = "インフレ圧力"
                        pressure_color = "red"
                    else:
                        pressure_type = "中立"
                        pressure_color = "green"
                    
                    st.markdown(f"""
                    <div class="correlation-box">
                        <h4>🌍 実効為替レート分析</h4>
                        <p><strong>USD実効為替レート:</strong> 基準日比 {neer_change:+.1f}%</p>
                        <p><strong>インフレ圧力:</strong> <span style="color: {pressure_color}; font-weight: bold;">{pressure_type}</span></p>
                        <p><strong>解釈:</strong> {'USDの実効為替レート上昇により輸入価格下落圧力' if neer_change > 0 else 'USDの実効為替レート下落により輸入価格上昇圧力' if neer_change < 0 else '実効為替レートは概ね安定'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 通貨別寄与度
                    st.subheader("通貨別寄与度分析")
                    
                    # 最新の通貨別レート表示
                    latest_rates = multi_fx_data.groupby('CURRENCY_PAIR')['EXCHANGE_RATE'].last()
                    initial_rates = multi_fx_data.groupby('CURRENCY_PAIR')['EXCHANGE_RATE'].first()
                    rate_changes = ((latest_rates / initial_rates) - 1) * 100
                    
                    rate_df = pd.DataFrame({
                        '通貨ペア': rate_changes.index,
                        '期間変化率(%)': rate_changes.values,
                        '最新レート': latest_rates.values
                    }).round(3)
                    
                    st.dataframe(rate_df, use_container_width=True)
                    
                    # AI分析実行ボタン
                    st.markdown('<div class="ai-button-container">', unsafe_allow_html=True)
                    if st.button("🤖 AI実効為替レート分析を実行", key="reer_ai_button"):
                        # REER/NEER分析用のデータコンテキスト作成
                        reer_context = f"""
                        分析期間: {start_date} ～ {end_date}
                        現在のNEER: {latest_neer:.1f}
                        現在のREER: {latest_reer:.1f}
                        基準日比変化: {neer_change:+.1f}%
                        インフレ圧力: {pressure_type}
                        
                        通貨別変化率:
                        {chr(10).join([f"{row['通貨ペア']}: {row['期間変化率(%)']:+.2f}%" for _, row in rate_df.iterrows()])}
                        """
                        
                        with st.spinner("🤖 AI分析中..."):
                            ai_analysis = run_ai_analysis("reer", reer_context, st.session_state.selected_ai_model)
                            st.session_state.reer_ai_analysis = ai_analysis
                            st.success("✅ AI分析完了！")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # AI分析結果表示
                    if 'reer_ai_analysis' in st.session_state:
                        st.markdown("""
                        <div class="ai-result-box">
                            <div class="ai-result-title">🧠 AI分析結果</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(st.session_state.reer_ai_analysis)
                    else:
                        st.info("💡 上記の「🚀 AI実効為替レート分析を実行」ボタンで詳細な分析を開始できます")
                else:
                    st.warning("REER/NEERの計算ができませんでした。")
            else:
                st.warning("複数通貨データが取得できませんでした。")

    with tab5:
        st.markdown('<div class="section-title">統合ダッシュボード</div>', unsafe_allow_html=True)
        
        # KPIメトリクス
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_fx = fx_data['EXCHANGE_RATE'].iloc[-1]
            fx_change = fx_data['Monthly_Return'].iloc[-1]
            st.metric(
                f"{currency_pair}",
                f"{latest_fx:.2f}",
                f"{fx_change:+.2f}% (月次)"
            )
        
        with col2:
            latest_cpi_all = cpi_data[cpi_data['PRODUCT'] == 'All items']['YoY_Change'].iloc[-1]
            st.metric(
                "総合CPI (YoY)",
                f"{latest_cpi_all:.2f}%"
            )
        
        with col3:
            latest_vol = fx_data['Volatility_20D'].iloc[-1]
            st.metric(
                "FXボラティリティ",
                f"{latest_vol:.1f}%"
            )
        
        with col4:
            regime_status = detect_volatility_regime(fx_data)
            st.metric(
                "マーケットレジーム",
                regime_status
            )
        
        # 統合チャート
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f'{currency_pair} 為替レート',
                'CPI主要項目 (YoY)',
                'FX-CPI相関 (ローリング12ヶ月)'
            ],
            vertical_spacing=0.08
        )
        
        # 為替レート
        fig.add_trace(
            go.Scatter(
                x=fx_data['DATE'],
                y=fx_data['EXCHANGE_RATE'],
                mode='lines',
                name=currency_pair,
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # CPI主要項目
        major_cpi_items = ['All items', 'All items less food and energy', 'Food', 'Energy']
        colors = ['black', 'red', 'green', 'orange']
        
        for item, color in zip(major_cpi_items, colors):
            item_data = cpi_data[cpi_data['PRODUCT'] == item]
            if not item_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=item_data['DATE'],
                        y=item_data['YoY_Change'],
                        mode='lines',
                        name=item,
                        line=dict(color=color)
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(height=900, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()