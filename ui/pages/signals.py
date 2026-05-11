# ui/pages/signals.py
"""信号中心页"""

import streamlit as st
import pandas as pd


def render(state: dict) -> None:
    from ui.components.widgets import render_signal_table, risk_status_banner

    config     = state["config"]
    signal_gen = state["signal_gen"]
    risk       = state["risk"]

    st.title("📡 信号中心")

    risk_status_banner(risk.get_risk_summary())

    signals_cache = st.session_state.get("signals_cache", {})
    if not signals_cache:
        st.info("暂无信号数据，请先前往「首页」刷新数据")
        return

    watchlist = config.get("watchlist", {})

    # 筛选控制
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_type = st.multiselect(
            "信号类型",
            ["强买入", "买入", "观望", "卖出", "强卖出"],
            default=["强买入", "买入", "卖出", "强卖出"],
        )
    with col_f2:
        filter_conf = st.multiselect(
            "置信度",
            ["高", "中", "低"],
            default=["高", "中"],
        )
    with col_f3:
        filter_vol = st.multiselect(
            "波动状态",
            ["低波动", "中等", "高波动"],
            default=["低波动", "中等", "高波动"],
        )

    # 汇总信号
    signals_df = signal_gen.summarize_signals(signals_cache)
    signals_df["名称"] = signals_df["代码"].map(watchlist)

    # 应用筛选
    type_map = {
        "强买入": "strong_buy", "买入": "buy",
        "观望": "hold",
        "卖出": "sell",  "强卖出": "strong_sell",
    }
    selected_types = [type_map[t] for t in filter_type if t in type_map]
    conf_map = {"高": "高", "中": "中", "低": "低"}
    vol_map  = {"低波动": "低波动", "中等": "中等", "高波动": "高波动"}

    filtered = signals_df[
        signals_df["_type"].isin(selected_types) &
        signals_df["置信度"].isin([conf_map[c] for c in filter_conf]) &
        signals_df["波动状态"].isin([vol_map[v] for v in filter_vol])
    ]

    st.markdown(f"**筛选结果: {len(filtered)} 只**")

    display_cols = ["代码", "名称", "信号", "概率", "阈值", "建议仓位", "置信度", "原因", "时间"]
    render_signal_table(
        filtered[[c for c in display_cols if c in filtered.columns]].assign(
            **{"_type": filtered["_type"]}
        )
    )

    # 风控黑名单展示
    with st.expander("风控黑名单", expanded=False):
        blacklist_df = risk.get_blacklist()
        if blacklist_df.empty:
            st.success("当前无黑名单股票")
        else:
            st.dataframe(blacklist_df, use_container_width=True, hide_index=True)
