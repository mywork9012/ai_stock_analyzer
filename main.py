# main.py
"""
AI股票量化分析软件 V1.0
Streamlit 主入口

运行方式:
    streamlit run main.py
"""

import streamlit as st
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 页面配置（必须是第一个 Streamlit 调用）──────────────────
st.set_page_config(
    page_title="AI量化分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局样式 ────────────────────────────────────────────────
st.markdown("""
<style>
/* ── 侧边栏：强制深色背景 + 白色文字，解决字体看不清问题 ── */
[data-testid="stSidebar"] {
    background-color: #1a1f2e !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #e8eaf0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.97em;
    padding: 4px 0;
}
[data-testid="stSidebar"] hr {
    border-color: #3a3f52 !important;
}

/* ── 指标卡片 ── */
.stMetric {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 12px;
}
.stMetric label { color: #90CAF9 !important; font-size: 0.85em; }

/* ── 通用 ── */
.stButton > button { border-radius: 6px; }
div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="初始化系统组件...")
def init_state():
    """
    初始化所有核心组件（仅执行一次，缓存到会话生命周期）
    """
    from core import load_config, setup_logging
    from core.data_manager  import StockDataFetcher
    from core.predict_engine import LSTMPredictor
    from core.signal_generator import SignalGenerator
    from core.risk_manager  import RiskController
    from core.backtest_engine import BacktestRunner

    config = load_config()
    setup_logging(config.get("system", {}).get("log_level", "INFO"))

    return {
        "config":     config,
        "fetcher":    StockDataFetcher(config),
        "predictor":  LSTMPredictor(config),
        "signal_gen": SignalGenerator(config),
        "risk":       RiskController(config),
        "backtest":   BacktestRunner(config),
    }


def main():
    # ── 初始化共享状态 ──
    try:
        state = init_state()
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        st.info("请检查 config.yaml 是否存在，以及依赖是否已安装（pip install -r requirements.txt）")
        st.stop()

    # ── 侧边栏导航 ──
    from ui.components.widgets import sidebar_navigation
    sidebar_navigation()

    st.sidebar.markdown("## 📈 AI量化分析 V1.0")
    st.sidebar.markdown("---")

    pages = {
        "🏠 首页·总览":   "home",
        "📈 单股分析":    "stock_analysis",
        "🤖 AI预测":      "ai_predict",
        "📡 信号中心":    "signals",
        "📋 回测报告":    "backtest",
        "⚙️ 系统配置":    "settings",
    }

    page_choice = st.sidebar.radio(
        "导航",
        list(pages.keys()),
        label_visibility="collapsed",
    )

    # 股票池快览
    watchlist = state["config"].get("watchlist", {})
    if watchlist:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**股票池**")
        for symbol, name in list(watchlist.items())[:8]:
            st.sidebar.caption(f"• {symbol} {name}")
        if len(watchlist) > 8:
            st.sidebar.caption(f"... 共 {len(watchlist)} 只")

    st.sidebar.markdown("---")
    st.sidebar.caption("V1.0 · 数据来源: baostock")
    st.sidebar.caption("⚠️ 仅供学习参考，不构成投资建议")

    # ── 页面路由 ──
    page_key = pages[page_choice]
    try:
        if page_key == "home":
            from ui.pages.home import render
        elif page_key == "stock_analysis":
            from ui.pages.stock_analysis import render
        elif page_key == "ai_predict":
            from ui.pages.ai_predict import render
        elif page_key == "signals":
            from ui.pages.signals import render
        elif page_key == "backtest":
            from ui.pages.backtest import render
        elif page_key == "settings":
            from ui.pages.settings import render
        else:
            st.error("页面不存在")
            return

        render(state)

    except ImportError as e:
        st.error(f"页面模块加载失败: {e}")
        st.code(str(e))
    except Exception as e:
        st.error(f"页面渲染异常: {e}")
        import traceback
        with st.expander("错误详情"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
