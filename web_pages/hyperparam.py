import streamlit as st
import subprocess

def show():
    st.markdown("""
    <span style='font-size:1.8rem;font-weight:bold;color:#0A6EBD;'>超参数调优</span>
    """, unsafe_allow_html=True)
    st.markdown("自动贝叶斯搜索最佳LSTM结构与参数，提升预测精度。")
    st.divider()
    st.markdown("#### 调优说明")
    st.info("点击下方按钮开始贝叶斯优化。建议先完成特征工程和基线训练。")
    if st.button("启动超参数调优"):
        with st.spinner("调优中（过程较久，请耐心等待）..."):
            result = subprocess.run(["python", "hyperparameter_tuning.py"], capture_output=True, text=True)
        st.success("调优完成！")
        with st.expander("调优日志"): st.code(result.stdout + '\n' + result.stderr)