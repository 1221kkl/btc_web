import streamlit as st
import os
import subprocess

def show():
    st.markdown("""
    <span style='font-size:1.8rem;font-weight:bold;color:#0A6EBD;'>模型评估与可视化</span>
    """, unsafe_allow_html=True)
    st.markdown("多指标综合评估模型表现，预测结果交互式展示。")
    st.divider()
    if st.button("一键评估与生成全部可视化图表"):
        with st.spinner("评估中..."):
            result = subprocess.run(["python", "model_evaluation.py"], capture_output=True, text=True)
        st.success("评估完成！")
        with st.expander("详细日志"): st.code(result.stdout + '\n' + result.stderr)

    output_dir = os.path.join(os.path.dirname(__file__), "../output/")
    st.markdown("#### 预测结果对比")
    col1, col2 = st.columns(2)
    base_img = os.path.join(output_dir, "baseline_results.png")
    tuned_img = os.path.join(output_dir, "tuned_results.png")
    if os.path.exists(base_img):
        col1.image(base_img, caption="基础模型")
    if os.path.exists(tuned_img):
        col2.image(tuned_img, caption="调优模型")
    perf_img = os.path.join(output_dir, "performance_compare.png")
    if os.path.exists(perf_img):
        st.markdown("#### 性能提升对比")
        st.image(perf_img)
    st.markdown("""
    <details>
    <summary>模型评估说明</summary>
    <ul>
    <li><b>RMSE/MAE：</b>越低越好，反映模型误差大小</li>
    <li><b>R²：</b>越高越好，反映模型拟合优度</li>
    <li>对比图可用于论文展示</li>
    </ul>
    </details>
    """, unsafe_allow_html=True)