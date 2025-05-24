import streamlit as st
import os
import pandas as pd
import shutil

def show():
    st.markdown("""
    <span style='font-size:1.8rem;font-weight:bold;color:#0A6EBD;'>数据采集</span>
    """, unsafe_allow_html=True)
    st.markdown("本模块模拟从主流交易所采集比特币历史数据，演示数据已内置。")
    st.divider()

    backup_path = os.path.join(os.path.dirname(__file__), "../data/raw/bitcoin_data_backup.csv")
    target_path = os.path.join(os.path.dirname(__file__), "../data/raw/bitcoin_data.csv")

    if st.button("一键采集比特币历史数据"):
        if os.path.exists(backup_path):
            shutil.copy(backup_path, target_path)
            st.success("采集完成！")
        else:
            st.error("未找到备份数据文件，请先放置 bitcoin_data_backup.csv 到 data/raw/ 下。")
    if os.path.exists(target_path):
        df = pd.read_csv(target_path)
        st.markdown("#### 数据预览（前20行，点击放大，可截图）")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True, height=500)
        st.markdown("#### 下载原始数据")
        st.download_button("下载CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="bitcoin_data.csv")
        with st.expander("查看完整字段解释"):
            st.markdown("""
            - **open_time**: 开盘时间
            - **open/high/low/close**: 开盘/最高/最低/收盘价
            - **volume**: 成交量
            """)
    else:
        st.info("请先点击上方按钮完成采集（或放置数据文件）。")