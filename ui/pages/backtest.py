# ui/pages/backtest.py
"""
回测报告页
- 参数配置（起止日期、策略）
- 资金曲线 + 回撤阴影
- 绩效指标卡
- 月度收益热力图
- 交易明细表
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def render(state: dict) -> None:
    from ui.components.charts import make_equity_curve, make_monthly_heatmap
    from ui.components.widgets import render_trade_stats, stock_search_box

    config      = state["config"]
    fetcher     = state["fetcher"]
    predictor   = state["predictor"]
    signal_gen  = state["signal_gen"]
    backtest    = state["backtest"]

    st.title("📋 回测报告")

    # ── 参数配置栏 ──────────────────────────────────────────
    st.markdown("### 回测参数")
    col1, col2, col3, col4 = st.columns(4)

    stock_list = fetcher.fetch_stock_list()
    with col1:
        symbol = stock_search_box(
            stock_list, key="bt_symbol", default="600519"
        )
    with col2:
        start_date = st.date_input(
            "开始日期",
            value=datetime.now() - timedelta(days=365 * 3),
            key="bt_start",
        )
    with col3:
        end_date = st.date_input(
            "结束日期",
            value=datetime.now(),
            key="bt_end",
        )
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_bt = st.button("▶ 运行回测", use_container_width=True, type="primary")

    # 高级参数
    with st.expander("高级参数", expanded=False):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            init_cash = st.number_input(
                "初始资金（元）", value=1_000_000, step=100_000, min_value=10_000
            )
        with adv2:
            commission = st.number_input(
                "佣金率", value=0.00025, format="%.5f", step=0.00001
            )
        with adv3:
            slippage = st.number_input(
                "滑点", value=0.001, format="%.4f", step=0.0001
            )
        # 将自定义参数注入回测引擎（临时覆盖）
        backtest.init_cash  = init_cash
        backtest.commission = commission
        backtest.slippage   = slippage

    # ── 运行回测 ────────────────────────────────────────────
    bt_key = f"bt_{symbol}_{start_date}_{end_date}"

    if run_bt:
        with st.spinner(f"正在对 {symbol} 执行回测，请稍候..."):
            from core.factor_engine import FactorCalculator
            calc = FactorCalculator(config)

            df_daily   = fetcher.fetch_daily(
                symbol,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )
            if df_daily.empty:
                st.error("行情数据获取失败，请检查股票代码或日期范围")
                return

            df_fund    = fetcher.fetch_fundamental(symbol)
            df_factors = calc.calc_all(df_daily, df_fund)

            # 获取基准指数数据
            df_benchmark = fetcher.get_index_data(
                config.get("backtest", {}).get("benchmark", "000300")
            )

            report = backtest.run(
                symbol=symbol,
                df_factors=df_factors,
                predictor=predictor,
                signal_gen=signal_gen,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                df_benchmark=df_benchmark if not df_benchmark.empty else None,
            )
            st.session_state[bt_key] = report

    report = st.session_state.get(bt_key)

    if not report:
        st.info("配置参数后点击「运行回测」按钮开始")
        _show_disclaimer()
        return

    if not report.get("success"):
        reason = report.get("reason", "未知错误")
        msg    = report.get("message", "")
        if reason == "insufficient_data":
            st.error("⚠️ 回测区间数据不足")
            if msg:
                st.markdown(msg)
            else:
                st.markdown(
                    f"模型需要至少 **65个交易日** 的数据才能生成信号。\n\n"
                    "**解决方法**：将回测开始日期往前调整至少3个月，或选择更长的回测区间。"
                )
        else:
            st.error(f"回测失败: {reason}")
            if msg:
                st.caption(msg)
        return

    # 降级引擎提示
    if report.get("engine") == "manual_fallback":
        st.warning("⚠️ VectorBT 未安装，已使用内置简化回测引擎（部分指标可能不准确）")

    watchlist = config.get("watchlist", {})
    name = watchlist.get(symbol, symbol)
    st.markdown(f"### {name}（{symbol}）| {report['start']} ~ {report['end']}")

    # ── 核心绩效指标 ────────────────────────────────────────
    st.markdown("#### 核心绩效指标")
    render_trade_stats(report)

    # 基准对比
    bm_ret = report.get("benchmark_return")
    bh_ret = report.get("buy_hold_return", 0)
    ann_ret = report.get("annual_return", 0)

    if bm_ret is not None or bh_ret:
        st.markdown("#### 策略 vs 基准")
        cmp1, cmp2, cmp3 = st.columns(3)
        cmp1.metric(
            "策略年化收益",
            f"{ann_ret:.1%}",
            help="已扣除佣金、印花税、滑点",
        )
        cmp2.metric(
            "买入持有收益",
            f"{bh_ret:.1%}",
            delta=f"{ann_ret - bh_ret:+.1%} vs 持有",
        )
        if bm_ret is not None:
            cmp3.metric(
                "沪深300收益",
                f"{bm_ret:.1%}",
                delta=f"{ann_ret - bm_ret:+.1%} vs 基准",
            )

    st.markdown("---")

    # ── 资金曲线 ────────────────────────────────────────────
    st.markdown("#### 资金曲线")
    equity_data = report.get("equity_curve", [])
    if equity_data:
        fig_eq = make_equity_curve(
            equity_data=equity_data,
            symbol=symbol,
            init_cash=report.get("init_cash", 1_000_000),
            buy_hold_return=bh_ret,
            height=420,
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("资金曲线数据不可用")

    st.markdown("---")

    # ── 月度收益热力图 ──────────────────────────────────────
    st.markdown("#### 月度收益热力图")
    monthly = report.get("monthly_returns", {})
    if monthly:
        fig_heat = make_monthly_heatmap(monthly, height=280)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("月度收益数据不可用（简化回测引擎暂不支持）")

    st.markdown("---")

    # ── 交易明细 ────────────────────────────────────────────
    st.markdown("#### 交易明细")
    trades = report.get("trades", [])
    if trades:
        df_trades = pd.DataFrame(trades)

        # 美化列名
        rename_map = {
            "entry_price": "买入价",
            "exit_price":  "卖出价",
            "pnl_pct":     "盈亏%",
            "exit_date":   "卖出日期",
            "Entry Timestamp": "买入时间",
            "Exit Timestamp":  "卖出时间",
            "PnL":         "盈亏(元)",
            "Return":      "收益率",
        }
        df_trades = df_trades.rename(columns=rename_map)

        # 格式化数值列
        for col in df_trades.columns:
            if "%" in col or "收益" in col or "盈亏%" in col:
                try:
                    df_trades[col] = df_trades[col].apply(
                        lambda v: f"{v:.2%}" if isinstance(v, (int, float)) else v
                    )
                except Exception:
                    pass
            elif "价" in col or "(元)" in col:
                try:
                    df_trades[col] = df_trades[col].apply(
                        lambda v: f"¥{v:.2f}" if isinstance(v, (int, float)) else v
                    )
                except Exception:
                    pass

        st.dataframe(
            df_trades,
            use_container_width=True,
            hide_index=True,
            height=min(400, 40 + len(df_trades) * 38),
        )

        # 导出按钮
        csv = pd.DataFrame(trades).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇ 下载交易明细 CSV",
            data=csv,
            file_name=f"{symbol}_trades_{report['start']}_{report['end']}.csv",
            mime="text/csv",
        )
    else:
        st.info("回测期间无成交记录")

    _show_disclaimer()


def _show_disclaimer():
    st.markdown("---")
    st.caption(
        "⚠️ 免责声明：回测结果基于历史数据，不代表未来收益。"
        "交易成本已计入（印花税0.1%+佣金0.025%+滑点0.1%）。"
        "历史回测存在幸存者偏差，仅供参考。"
    )
