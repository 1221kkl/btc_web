import streamlit as st
import os
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

def show():
    st.markdown("""
    <span style='font-size:1.8rem;font-weight:bold;color:#0A6EBD;'>特征工程与指标可视化</span>
    """, unsafe_allow_html=True)
    st.markdown("自动生成技术指标，筛选关键特征，辅助提升模型预测能力。")
    st.divider()

    cleaned_path = os.path.join(os.path.dirname(__file__), "../data/processed/bitcoin_data_cleaned.csv")
    lstm_data_path = os.path.join(os.path.dirname(__file__), "../data/processed/lstm_data.npz")
    if st.button("生成技术指标与LSTM输入数据"):
        with st.spinner("特征工程处理中..."):
            result = subprocess.run(["python", "feature_engineering.py"], capture_output=True, text=True)
        st.success("特征生成完成！")
        with st.expander("详细日志"): st.code(result.stdout + '\n' + result.stderr)
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        # 动态判断有哪些技术指标
        tech_cols = []
        name_map = {"MA_7":"7日均线", "EMA_7":"7日指数均线", "RSI":"相对强弱", "MACD":"MACD", "Volatility":"波动率"}
        for col in ["MA_7", "EMA_7", "RSI", "MACD", "Volatility"]:
            if col in df.columns:
                tech_cols.append(col)
        if tech_cols:
            st.markdown("#### 技术指标曲线（近90天）")
            fig, ax = plt.subplots(figsize=(9, 3.5))
            df = df.tail(90)
            for col in tech_cols:
                ax.plot(df["open_time"], df[col], label=name_map.get(col, col))
            ax.set_xticks(df["open_time"][::15])
            ax.set_xticklabels(df["open_time"][::15], rotation=30)
            ax.legend()
            ax.set_title("关键技术指标走势")
            st.pyplot(fig)
        else:
            st.info("未检测到MA/EMA/RSI/MACD等技术指标，请检查特征工程脚本。")
        # 相关性热力图
        st.markdown("#### 指标相关性热力图")
        if tech_cols:
            corr = df[tech_cols].corr()
            st.dataframe(corr.style.background_gradient(cmap='RdYlBu'))
        else:
            st.info("暂无可用指标计算相关性。")
    else:
        st.info("请先生成清洗后数据文件。")