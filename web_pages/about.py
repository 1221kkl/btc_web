import streamlit as st

def show():
    st.markdown("""
    <span style='font-size:2rem;font-weight:bold;color:#0A6EBD;'>系统简介</span>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:1.1rem;line-height:1.75em;'>
    <b>本系统基于LSTM神经网络实现比特币价格预测，覆盖数据采集、清洗、特征工程、模型训练调优、评估与可视化全流程。</b><br>
    <ul>
    <li><b>数据采集：</b>自动采集主流交易所历史K线数据</li>
    <li><b>数据处理：</b>缺失值填补、异常处理、归一化</li>
    <li><b>特征工程：</b>生成MA、EMA、RSI、MACD等关键指标</li>
    <li><b>LSTM模型：</b>定制结构，支持超参数调优</li>
    <li><b>评估与可视化：</b>多种指标与曲线图，支持论文截图</li>
    </ul>
    <b>系统亮点：</b>全流程自动化、一键可视化、支持论文与路演、界面美观高效。
    </div>
    """, unsafe_allow_html=True)