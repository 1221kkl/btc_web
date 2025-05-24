import streamlit as st

st.set_page_config(page_title="比特币LSTM预测系统", layout="wide")

# ------------------- 侧边栏字体美化CSS -------------------
st.markdown("""
    <style>
    /* Streamlit 1.24+ 侧边栏导航字体美化 */
    [data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1.4rem !important;
        font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial Rounded MT Bold', 'Arial', 'sans-serif' !important;
        font-weight: 700 !important;
        color: #0A6EBD !important;
        letter-spacing: 1px;
        margin-bottom: 1.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- 顶部横幅 -------------------
st.markdown("""
    <div style="background:#0A6EBD;padding:20px 0 10px 0;text-align:center;">
        <span style="color:white;font-size:2.1rem;font-weight:bold;">比特币价格智能预测分析平台</span>
    </div>
""", unsafe_allow_html=True)

# ------------------- 页面导航 -------------------
pages = {
    "仪表盘首页": "dashboard",
    "数据采集": "data_collect",
    "数据预处理": "data_preprocess",
    "特征工程": "feature_eng",
    "LSTM模型训练": "model_train",
    "超参数调优": "hyperparam",
    "模型评估与可视化": "evaluation",
    "系统简介": "about"
}

choice = st.sidebar.radio("导航", list(pages.keys()))
page = pages[choice]
exec(f"import web_pages.{page} as cur_page; cur_page.show()")