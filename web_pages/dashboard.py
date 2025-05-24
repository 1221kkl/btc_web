import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os

def show():
    st.markdown("""
    <style>
    .big-title {font-size: 2.2rem; font-weight: bold; color: #0A6EBD;}
    .desc {font-size: 1.1rem; color: #333;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="big-title">比特币价格预测分析系统 · 仪表盘</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc">一站式LSTM预测流程，自动可视化，适合论文和演示。</div>', unsafe_allow_html=True)
    st.divider()

    # 数据摘要卡片
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed/bitcoin_data_cleaned.csv")
    pred_path = os.path.join(os.path.dirname(__file__), "../output/predicted_today_close.txt")
    real_pred = None
    if os.path.exists(pred_path):
        with open(pred_path, "r") as f:
            real_pred = f.read().strip()
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        latest = df.iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("最新收盘价", f"{float(latest['close']):,.2f} USD", delta=f"{latest['close']-df.iloc[-2]['close']:+.2f}")
        col2.metric("24h成交量", f"{float(latest['volume']):,.2f} BTC")
        col3.metric("历史最高", f"{df['high'].max():,.2f} USD")
        col4.metric("历史最低", f"{df['low'].min():,.2f} USD")
        if real_pred:
            col5.metric("预计今日收盘价", real_pred + " USD", delta=None)
        else:
            col5.metric("预计今日收盘价", "请先运行模型评估", delta=None)
        st.divider()

        # 价格走势
        st.markdown("#### 近180天收盘价走势（可用于论文截图）")
        fig = go.Figure()
        show_days = min(180, len(df))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df["open_time"].tail(show_days)),
            y=df["close"].tail(show_days),
            mode='lines',
            name='收盘价',
            line=dict(color='#1976D2', width=2)
        ))
        fig.update_layout(height=450, width=1100, margin=dict(t=10, b=10, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("请先完成数据采集与预处理。")