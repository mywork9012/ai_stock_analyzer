# ui/pages/home.py
"""
首页·总览
- baostock 连接状态检测
- 今日信号汇总（30只股票）
- 逐股进度反馈，避免"一直转圈"
"""

import streamlit as st
import pandas as pd
from datetime import datetime


def render(state: dict) -> None:
    from ui.components.widgets import risk_status_banner

    config     = state["config"]
    fetcher    = state["fetcher"]
    predictor  = state["predictor"]
    signal_gen = state["signal_gen"]
    risk       = state["risk"]

    st.title("📊 AI量化分析 · 今日概览")
    st.caption(f"数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ── 风控状态 ──────────────────────────────────────────────
    risk_status_banner(risk.get_risk_summary())

    # ── baostock 连接状态 ─────────────────────────────────────
    _show_connection_status()

    # ── 股票池检查 ────────────────────────────────────────────
    watchlist = config.get("watchlist", {})
    if not watchlist:
        st.warning("股票池为空，请在【系统配置 → 股票池】中添加股票")
        return

    # ── 未训练模型提示 ────────────────────────────────────────
    untrained = [s for s in watchlist if not _model_exists(s)]
    if untrained:
        st.info(
            f"💡 {len(untrained)} 只股票尚未训练模型，请前往【系统配置 → 模型训练】执行训练"
            f"（{', '.join(list(untrained)[:3])}{'...' if len(untrained) > 3 else ''}）"
        )

    # ── 刷新按钮（首次进入不自动拉取，避免长时间无反馈）────────
    col_btn, col_tip = st.columns([2, 4])
    with col_btn:
        do_refresh = st.button("🔄 刷新今日数据", use_container_width=True, type="primary")
    with col_tip:
        if "signals_cache" in st.session_state:
            last_time = st.session_state.get("signals_last_update", "")
            st.caption(f"上次更新: {last_time}" if last_time else "")
        else:
            st.caption("点击按钮获取今日行情与信号")

    # ── 执行刷新 ──────────────────────────────────────────────
    if do_refresh:
        _do_fetch(watchlist, config, fetcher, predictor, signal_gen)

    # ── 展示结果 ──────────────────────────────────────────────
    signals_cache = st.session_state.get("signals_cache")

    if not signals_cache:
        st.markdown("---")
        st.info("📭 暂无数据，点击上方【刷新今日数据】开始获取")
        return

    # 统计
    all_sigs   = list(signals_cache.values())
    buy_count  = sum(1 for s in all_sigs if s.signal_type in ("buy", "strong_buy"))
    sell_count = sum(1 for s in all_sigs if s.signal_type in ("sell", "strong_sell"))
    hold_count = sum(1 for s in all_sigs if s.signal_type == "hold")
    fail_count = st.session_state.get("fetch_fail_count", 0)

    st.markdown("### 信号统计")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔴 买入信号", buy_count,  help="强买入 + 买入")
    c2.metric("⚪ 观望信号", hold_count)
    c3.metric("🟢 卖出信号", sell_count, help="强卖出 + 卖出")
    c4.metric("📊 覆盖股票", len(signals_cache))
    if fail_count:
        c5.metric("⚠️ 获取失败", fail_count)

    # 信号明细表
    st.markdown("### 今日信号明细")
    filter_type = st.radio(
        "筛选", ["全部", "买入类", "观望", "卖出类"],
        horizontal=True, key="home_filter",
    )

    signals_df = signal_gen.summarize_signals(signals_cache)
    signals_df["名称"] = signals_df["代码"].map(watchlist)
    cols = ["代码", "名称", "信号", "概率", "阈值", "波动状态", "建议仓位", "置信度"]
    signals_df = signals_df[[c for c in cols if c in signals_df.columns]]

    if filter_type == "买入类":
        filtered = signals_df[signals_df["信号"].str.contains("买入")]
    elif filter_type == "观望":
        filtered = signals_df[signals_df["信号"].str.contains("观望")]
    elif filter_type == "卖出类":
        filtered = signals_df[signals_df["信号"].str.contains("卖出")]
    else:
        filtered = signals_df

    if filtered.empty:
        st.info("当前筛选条件下无信号")
    else:
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=380)

    st.markdown("---")
    st.markdown(
        "📌 左侧导航：**单股分析** 查看K线 | **AI预测** 查看概率详情 | **回测报告** 验证策略"
    )


# ── 内部函数 ──────────────────────────────────────────────────

def _show_connection_status():
    """检测并显示 baostock 连接状态"""
    status_key = "bs_status"

    # 每次页面加载检测一次（缓存在 session）
    if status_key not in st.session_state:
        try:
            import baostock as bs
            ret = bs.login()
            if ret.error_code == "0":
                bs.logout()
                st.session_state[status_key] = ("ok", "baostock 连接正常")
            else:
                st.session_state[status_key] = ("err", f"baostock 登录失败: {ret.error_msg}")
        except ImportError:
            st.session_state[status_key] = (
                "warn",
                "baostock 未安装，请执行: pip install baostock",
            )
        except Exception as e:
            st.session_state[status_key] = ("err", f"baostock 异常: {e}")

    level, msg = st.session_state[status_key]
    if level == "ok":
        st.success(f"✅ 数据源状态: {msg}", icon="📡")
    elif level == "warn":
        st.warning(f"⚠️ {msg}")
    else:
        st.error(f"❌ 数据源错误: {msg}")


def _do_fetch(watchlist, config, fetcher, predictor, signal_gen):
    """逐股获取数据并实时反馈进度"""
    from core.factor_engine import FactorCalculator

    calc          = FactorCalculator(config)
    symbols       = list(watchlist.keys())
    total         = len(symbols)
    signals_cache = {}
    df_map        = {}
    fail_count    = 0

    # 进度容器
    progress_bar  = st.progress(0, text="准备中...")
    status_box    = st.empty()   # 显示当前正在处理的股票
    error_box     = st.empty()   # 汇总失败信息

    errors = []

    for i, symbol in enumerate(symbols):
        name = watchlist.get(symbol, symbol)
        pct  = (i + 1) / total

        progress_bar.progress(pct, text=f"[{i+1}/{total}] 正在获取 {symbol} {name}...")
        status_box.caption(f"⏳ 当前: {symbol} {name} | 已完成: {i}/{total}")

        try:
            df_daily   = fetcher.fetch_daily(symbol, force_refresh=False)
            if df_daily.empty:
                errors.append(f"{symbol} 行情为空")
                fail_count += 1
                continue

            df_fund    = fetcher.fetch_fundamental(symbol)
            df_factors = calc.calc_all(df_daily, df_fund)
            df_map[symbol] = df_factors

            pred = predictor.predict(symbol, df_factors)
            acc  = predictor.get_accuracy_stats(symbol)
            sig  = signal_gen.generate(symbol, pred, df_factors, acc)
            signals_cache[symbol] = sig

        except Exception as e:
            errors.append(f"{symbol}: {e}")
            fail_count += 1

    # 清理进度组件
    progress_bar.empty()
    status_box.empty()

    # 保存结果
    st.session_state["signals_cache"]      = signals_cache
    st.session_state["df_map"]             = df_map
    st.session_state["fetch_fail_count"]   = fail_count
    st.session_state["signals_last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 结果提示
    if signals_cache:
        st.success(f"✅ 数据获取完成：{len(signals_cache)}/{total} 只股票成功")
    if errors:
        with error_box.expander(f"⚠️ {fail_count} 只股票获取失败（点击查看详情）"):
            for err in errors:
                st.caption(err)


def _model_exists(symbol: str) -> bool:
    from models.trainer import ModelTrainer
    return ModelTrainer.model_exists(symbol)
