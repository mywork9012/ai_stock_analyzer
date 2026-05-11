# ui/components/widgets.py
"""
可复用Streamlit UI组件
- 股票代码搜索框
- 指标卡片
- 信号徽章
- 数据加载状态
"""

import streamlit as st
import pandas as pd
from typing import Optional


def stock_search_box(
    stock_list: pd.DataFrame,
    key: str = "stock_search",
    label: str = "股票代码 / 名称搜索",
    default: str = "600519",
) -> Optional[str]:
    """
    股票代码搜索输入框（支持代码+名称模糊搜索）

    Args:
        stock_list: DataFrame，含 symbol 和 name 列
        key: Streamlit组件key
        label: 输入框标签
        default: 默认股票代码

    Returns:
        选中的股票代码，如 "600519"
    """
    if stock_list.empty:
        code = st.text_input(label, value=default, key=key,
                             placeholder="输入股票代码，如 600519")
        return code.strip()

    # 构建搜索选项
    options = {
        f"{row['symbol']} {row['name']}": row["symbol"]
        for _, row in stock_list.iterrows()
    }

    # 默认选项
    default_display = next(
        (k for k, v in options.items() if v == default), list(options.keys())[0]
    )

    selected = st.selectbox(
        label,
        options=list(options.keys()),
        index=list(options.keys()).index(default_display) if default_display in options else 0,
        key=key,
    )
    return options.get(selected, default)


def metric_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
):
    """
    指标卡片（带中文解释悬浮提示）
    """
    st.metric(label=title, value=value, delta=delta,
              delta_color=delta_color, help=help_text)


def signal_badge(signal_type: str, probability: float, symbol: str = "") -> None:
    """
    信号徽章展示
    """
    # 中国股市配色：买入=红色，卖出=绿色
    badge_styles = {
        "strong_buy":  ("🔴 强买入", "#B71C1C"),   # 深红
        "buy":         ("🟥 买入",   "#E53935"),   # 主红
        "hold":        ("⬜ 观望",   "#616161"),   # 灰色
        "sell":        ("🟩 卖出",   "#00897B"),   # 主绿
        "strong_sell": ("🟢 强卖出", "#004D40"),   # 深绿
    }
    label, color = badge_styles.get(signal_type, ("⚪ 观望", "#616161"))

    st.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:12px 20px;
            border-radius:8px;
            text-align:center;
            font-size:1.2em;
            font-weight:bold;
            color:white;
            margin:8px 0;
        ">
            {label}<br>
            <span style="font-size:0.8em;font-weight:normal;">
                次日上涨概率: {probability:.1%}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_status_banner(risk_summary: dict) -> None:
    """
    风控状态横幅
    """
    circuit = risk_summary.get("circuit_breaker", False)
    defensive = risk_summary.get("defensive_mode", False)
    daily_loss = risk_summary.get("daily_loss_pct", 0)
    daily_limit = risk_summary.get("daily_loss_limit", 0.03)

    if circuit:
        until = risk_summary.get("circuit_breaker_until", "")
        st.error(f"⛔ 连续亏损熔断触发 | 暂停交易至: {until[:16]}")
    elif defensive:
        st.warning("🛡️ 防御模式已激活 | 策略回撤超阈值，仅允许卖出操作")
    elif daily_loss > daily_limit * 0.7:
        st.warning(f"⚠️ 日亏损接近熔断阈值 | 当日亏损: {daily_loss:.1%} / 上限: {daily_limit:.1%}")


def loading_placeholder(msg: str = "数据加载中..."):
    """加载占位符"""
    return st.empty()


def render_signal_table(signals_df: pd.DataFrame) -> None:
    """
    渲染信号汇总表格（带颜色高亮）
    """
    if signals_df.empty:
        st.info("暂无信号数据")
        return

    def highlight_signal(row):
        colors = {
            "strong_buy":  "background-color: rgba(211,47,47,0.3)",
            "buy":         "background-color: rgba(245,124,0,0.2)",
            "hold":        "",
            "sell":        "background-color: rgba(56,142,60,0.2)",
            "strong_sell": "background-color: rgba(27,94,32,0.3)",
        }
        style = colors.get(row.get("_type", "hold"), "")
        return [style] * len(row)

    display_df = signals_df.drop(columns=["_type"], errors="ignore")

    # 中国股市配色：买入行红色背景，卖出行绿色背景
    styled = display_df.style.apply(
        lambda row: [
            (
                "background-color: rgba(229,57,53,0.25)" if "买" in str(row.get("信号", ""))
                else "background-color: rgba(38,166,154,0.20)" if "卖" in str(row.get("信号", ""))
                else ""
            )
        ] * len(row),
        axis=1,
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_trade_stats(report: dict) -> None:
    """渲染回测绩效卡片（4列）"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "年化收益率",
            f"{report.get('annual_return', 0):.1%}",
            help_text="策略年化收益率，已扣除交易成本",
        )
    with col2:
        metric_card(
            "Sharpe比率",
            f"{report.get('sharpe_ratio', 0):.2f}",
            help_text="风险调整后收益，>1为良好，>2为优秀",
        )
    with col3:
        metric_card(
            "最大回撤",
            f"{report.get('max_drawdown', 0):.1%}",
            help_text="策略历史最大亏损幅度（越小越好）",
        )
    with col4:
        metric_card(
            "胜率",
            f"{report.get('win_rate', 0):.1%}",
            help_text="盈利交易占总交易次数的比例",
        )

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        metric_card("总交易次数", str(report.get("total_trades", 0)))
    with col6:
        metric_card("Calmar比率",
                    f"{report.get('calmar_ratio', 0):.2f}",
                    help_text="年化收益/最大回撤，越高越好")
    with col7:
        bhr = report.get("buy_hold_return")
        metric_card("买入持有收益",
                    f"{bhr:.1%}" if bhr is not None else "N/A",
                    help_text="同期买入持有策略收益，用于基准对比")
    with col8:
        total = report.get("total_return")
        metric_card("总收益率",
                    f"{total:.1%}" if total is not None else "N/A",
                    help_text="回测期间累计收益率")


def sidebar_navigation() -> None:
    """侧边栏导航美化"""
    st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("## 🤖 AI量化分析")
    st.sidebar.markdown("---")


def format_large_number(n: float) -> str:
    """格式化大数字（万/亿）"""
    if n >= 1e8:
        return f"{n/1e8:.2f}亿"
    elif n >= 1e4:
        return f"{n/1e4:.2f}万"
    return f"{n:.2f}"
