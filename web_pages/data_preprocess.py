import streamlit as st
import os
import pandas as pd
import subprocess

def show():
    st.markdown("""
    <span style='font-size:1.8rem;font-weight:bold;color:#0A6EBD;'>数据预处理</span>
    """, unsafe_allow_html=True)
    st.markdown("数据清洗、缺失值填补与格式标准化。")
    st.divider()

    raw_path = os.path.join(os.path.dirname(__file__), "../data/raw/bitcoin_data.csv")
    out_path = os.path.join(os.path.dirname(__file__), "../data/processed/bitcoin_data_cleaned.csv")
    if st.button("执行数据预处理"):
        with st.spinner("数据处理中..."):
            result = subprocess.run(["python", "data_preprocessing.py"], capture_output=True, text=True)
        st.success("处理完成！")
        with st.expander("详细日志"): st.code(result.stdout + '\n' + result.stderr)
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        st.markdown("#### 清洗后数据预览（前10行）")
        st.dataframe(df.head(10), hide_index=True)
        st.markdown(f"共{len(df)}条记录 | 时间范围：{df['open_time'].iloc[0]} ~ {df['open_time'].iloc[-1]}")
    else:
        st.info("还未生成清洗后数据文件。")