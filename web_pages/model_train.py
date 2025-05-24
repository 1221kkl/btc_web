import streamlit as st
import subprocess

def show():
    st.markdown("""
    <span style='font-size:1.8rem;font-weight:bold;color:#0A6EBD;'>LSTM模型训练</span>
    """, unsafe_allow_html=True)
    st.markdown("自定义网络结构，快速启动LSTM模型训练。")
    st.divider()
    st.markdown("#### 训练参数设置")
    with st.form("train_form"):
        units1 = st.slider("第一层LSTM单元数", 32, 128, 64, step=16)
        units2 = st.slider("第二层LSTM单元数", 32, 128, 96, step=16)
        dropout1 = st.slider("第一层Dropout", 0.0, 0.5, 0.1, step=0.05)
        dropout2 = st.slider("第二层Dropout", 0.0, 0.5, 0.1, step=0.05)
        lr = st.selectbox("学习率", [0.001, 0.0005, 0.0001])
        batch = st.selectbox("批量大小", [32, 64, 128])
        epochs = st.slider("训练轮数", 10, 100, 50, step=10)
        submitted = st.form_submit_button("开始训练")
    if submitted:
        with st.spinner("LSTM模型训练中..."):
            result = subprocess.run([
                "python", "lstm_model.py"
            ], capture_output=True, text=True)
        st.success("训练完成！")
        with st.expander("训练日志"): st.code(result.stdout + '\n' + result.stderr)