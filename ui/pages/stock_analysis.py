# ui/pages/stock_analysis.py
"""
单股分析页
- K线图 + 技术指标叠加
- 因子热力图
- 基本面指标
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta


def render(state: dict) -> None:
    from ui.components.charts import make_candlestick, make_rsi_chart
    from ui.components.widgets import stock_search_box
    from core.factor_engine import FactorCalculator

    config   = state["config"]
    fetcher  = state["fetcher"]
    watchlist = config.get("watchlist", {})

    st.title("📈 单股分析")

    # ── 控制栏 ──
    col_sym, col_start, col_end = st.columns([2, 1, 1])

    stock_list = fetcher.fetch_stock_list()
    with col_sym:
        symbol = stock_search_box(
            stock_list, key="analysis_symbol", default="600519"
        )
    with col_start:
        start = st.date_input(
            "开始日期",
            value=datetime.now() - timedelta(days=365),
            key="analysis_start",
        )
    with col_end:
        end = st.date_input("结束日期", value=datetime.now(), key="analysis_end")

    # ── 技术指标选择 ──
    st.markdown("**叠加技术指标**")
    ind_col = st.columns(6)
    ind_opts = {
        "SMA5":     "sma5",
        "SMA20":    "sma20",
        "SMA60":    "sma60",
        "EMA12":    "ema12",
        "布林带":   ["bb_upper", "bb_mid", "bb_lower"],
        "成交量":   "volume",
    }
    selected_inds = []
    for i, (label, cols) in enumerate(ind_opts.items()):
        with ind_col[i % 6]:
            if st.checkbox(label, value=(label in ["SMA20", "SMA5"]),
                           key=f"ind_{label}"):
                if isinstance(cols, list):
                    selected_inds.extend(cols)
                else:
                    selected_inds.append(cols)

    # ── 加载数据 ──
    cache_key = f"analysis_{symbol}_{start}_{end}"
    if cache_key not in st.session_state:
        with st.spinner(f"加载 {symbol} 数据..."):
            # 后复权数据用于因子计算和K线图（技术指标连续性）
            df_daily = fetcher.fetch_daily(
                symbol,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
            if df_daily.empty:
                st.error(f"未能获取 {symbol} 的行情数据")
                st.markdown(
                    "**常见原因及解决方法：**\n\n"
                    "1. **网络波动**：点击页面其他地方再重新选择股票\n\n"
                    "2. **baostock 会话超时**：刷新整个页面（F5）重新登录\n\n"
                    "3. **股票代码错误**：请确认代码为6位纯数字（如 600519）\n\n"
                    "4. **非交易日**：baostock 可能返回空数据，属正常现象"
                )
                return

            # 不复权数据仅用于价格展示（与行情软件一致）
            df_raw = fetcher.fetch_daily_raw(symbol)
            df_fund = fetcher.fetch_fundamental(symbol)
            calc = FactorCalculator(config)
            df_factors = calc.calc_all(df_daily, df_fund)
            # 若不复权数据获取失败，降级用复权数据展示
            df_display = df_raw if not df_raw.empty else df_daily
            st.session_state[cache_key] = {
                "daily": df_daily,
                "display": df_display,
                "factors": df_factors,
            }

    data = st.session_state.get(cache_key, {})
    df_daily   = data.get("daily", pd.DataFrame())
    df_display = data.get("display", df_daily)   # 展示用（不复权）
    df_factors = data.get("factors", pd.DataFrame())

    if df_daily.empty:
        return

    # ── 股票基本信息（用不复权价格展示，与普通交易软件一致）──
    name = watchlist.get(symbol, symbol)

    # df_display 是不复权数据（fetch_daily_raw），若获取失败则为后复权数据
    # 取最新一行和前一行用于计算涨跌幅
    latest = df_display.iloc[-1]
    prev   = df_display.iloc[-2] if len(df_display) > 1 else latest

    def _f(row, col, default=0.0):
        """安全取浮点数，兼容 NaN 和类型错误"""
        try:
            v = float(row[col])
            return v if v == v else default
        except Exception:
            return default

    close_val  = _f(latest, "close")
    high_val   = _f(latest, "high")
    low_val    = _f(latest, "low")
    prev_close = _f(prev,   "close", close_val)
    pct_chg    = (close_val - prev_close) / prev_close if prev_close else 0.0
    vol_val    = _f(latest, "volume")
    turn_val   = _f(latest, "turnover")

    # 数据日期
    data_date = ""
    try:
        data_date = str(latest["date"].date()) if hasattr(latest["date"], "date") else str(latest["date"])[:10]
    except Exception:
        pass

    st.markdown(f"### {name}（{symbol}）")
    st.caption(f"📅 最新数据日期: **{data_date}**（价格为不复权，与普通交易软件一致；K线图使用后复权数据保证技术指标连续性）")

    info_cols = st.columns(5)
    info_cols[0].metric("最新收盘", f"¥{close_val:.2f}", delta=f"{pct_chg:+.2%}")
    info_cols[1].metric("当日最高", f"¥{high_val:.2f}")
    info_cols[2].metric("当日最低", f"¥{low_val:.2f}")
    info_cols[3].metric("成交量",   f"{vol_val/1e4:.1f}万手" if vol_val > 0 else "—")
    if turn_val:
        info_cols[4].metric("换手率", f"{turn_val:.2f}%")

    # ── K线主图 ──
    # 加载信号数据（如有）
    signals_df = None
    df_signals_from_cache = st.session_state.get("df_map", {}).get(symbol)
    if df_signals_from_cache is not None:
        sigs = st.session_state.get("signals_cache", {})
        if sigs:
            signals_list = []
            for sym, sig in sigs.items():
                if sym == symbol:
                    signals_list.append({"date": pd.Timestamp.now().normalize(),
                                         "signal_type": sig.signal_type})
            if signals_list:
                signals_df = pd.DataFrame(signals_list)

    fig_candle = make_candlestick(
        df=df_daily,
        title=f"{name} K线图",
        indicators=selected_inds,
        signals_df=signals_df,
        height=580,
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # ── RSI图 ──
    if "rsi14" in df_factors.columns:
        fig_rsi = make_rsi_chart(df_factors, height=180)
        st.plotly_chart(fig_rsi, use_container_width=True)

    # ── 基本面指标 ──
    st.markdown("### 基本面指标")
    fund_cols = ["pe_ttm", "pb", "ps_ttm", "roe_ttm"]
    available_fund = {c: df_factors[c].iloc[-1] for c in fund_cols if c in df_factors.columns}

    if available_fund:
        f_cols = st.columns(len(available_fund))
        labels = {"pe_ttm": "市盈率(TTM)", "pb": "市净率(PB)",
                  "ps_ttm": "市销率(TTM)", "roe_ttm": "ROE(TTM)%"}
        for i, (col_name, val) in enumerate(available_fund.items()):
            if not pd.isna(val):
                f_cols[i].metric(labels.get(col_name, col_name), f"{val:.2f}")
    else:
        st.info("暂无基本面数据，baostock 财务数据为季度更新，首次加载需约10秒")

    # ── 因子热力图（近30日因子分布）──
    with st.expander("查看因子热力图（近30日）", expanded=False):
        _render_factor_heatmap(df_factors)


def _render_factor_heatmap(df_factors: pd.DataFrame) -> None:
    """渲染因子热力图"""
    if df_factors.empty:
        st.info("无因子数据")
        return

    # 选取主要因子
    show_cols = ["rsi14", "macd_dif", "bb_pct", "atr_ratio", "volume_ratio",
                 "kdj_k", "kdj_d", "price_to_sma5", "price_to_sma20"]
    available = [c for c in show_cols if c in df_factors.columns]

    if not available:
        st.info("因子列不足，无法绘制热力图")
        return

    df_heat = df_factors[["date"] + available].tail(30).copy()
    df_heat = df_heat.set_index("date")[available]

    import plotly.graph_objects as go
    fig = go.Figure(
        go.Heatmap(
            z=df_heat.T.values,
            x=[str(d.date()) for d in df_heat.index],
            y=available,
            colorscale="RdYlGn",
            zmid=0,
        )
    )
    fig.update_layout(
        title="因子值热力图（近30日）",
        height=280,
        template="plotly_dark",
        margin=dict(l=100, r=20, t=40, b=20),
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig, use_container_width=True)
