# ui/pages/settings.py
"""
系统配置页
- 股票池管理
- 模型训练（单只 / 批量）
- 数据源配置（baostock 主 / akshare 备用）
- 风控阈值调整
- 数据注册表查看
"""

import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def render(state: dict) -> None:
    config    = state["config"]
    fetcher   = state["fetcher"]
    predictor = state["predictor"]

    st.title("⚙️ 系统配置")

    tab_stock, tab_train, tab_data, tab_risk, tab_registry = st.tabs(
        ["📋 股票池", "🧠 模型训练", "🌐 数据源", "🛡️ 风控参数", "📂 数据注册表"]
    )

    # ══════════════════════════════════════════════════════
    # TAB 1：股票池管理
    # ══════════════════════════════════════════════════════
    with tab_stock:
        st.markdown("### 股票池管理")
        st.caption("最多支持30只股票，修改后点击保存，重启后生效")

        watchlist = config.get("watchlist", {})

        if watchlist:
            df_watch = pd.DataFrame(
                [{"代码": k, "名称": v} for k, v in watchlist.items()]
            )
            st.dataframe(df_watch, use_container_width=True, hide_index=True)
        else:
            st.info("股票池为空")

        st.markdown("---")

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_symbol = st.text_input("股票代码", placeholder="如 600519", key="add_symbol")
        with col2:
            new_name = st.text_input("股票名称", placeholder="如 贵州茅台", key="add_name")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ 添加", use_container_width=True):
                if new_symbol and new_name:
                    if len(watchlist) >= 30:
                        st.error("股票池已达30只上限")
                    elif new_symbol in watchlist:
                        st.warning(f"{new_symbol} 已在股票池中")
                    else:
                        watchlist[new_symbol] = new_name
                        _save_watchlist(watchlist)
                        st.success(f"✅ 已添加 {new_symbol} {new_name}")
                        st.rerun()
                else:
                    st.warning("请填写代码和名称")

        if watchlist:
            col_del1, col_del2 = st.columns([3, 1])
            with col_del1:
                del_choice = st.selectbox(
                    "删除股票",
                    [f"{k} {v}" for k, v in watchlist.items()],
                    key="del_stock",
                )
            with col_del2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ 删除", use_container_width=True):
                    del_symbol = del_choice.split(" ")[0]
                    watchlist.pop(del_symbol, None)
                    _save_watchlist(watchlist)
                    st.success(f"已删除 {del_symbol}")
                    st.rerun()

    # ══════════════════════════════════════════════════════
    # TAB 2：模型训练
    # ══════════════════════════════════════════════════════
    with tab_train:
        st.markdown("### 模型训练")
        st.caption(
            "训练使用最近3年数据，验证集为最近3个月（时序划分）。"
            "30只股票批量训练约需1~4小时，建议在周末执行。"
        )

        from models.trainer import ModelTrainer, MODEL_DIR

        watchlist = config.get("watchlist", {})
        all_meta  = ModelTrainer.get_all_model_meta()

        st.markdown("#### 已训练模型状态")
        status_rows = []
        for symbol, name in watchlist.items():
            meta = next((m for m in all_meta if m["symbol"] == symbol), None)
            status_rows.append({
                "代码":     symbol,
                "名称":     name,
                "状态":     "✅ 已训练" if meta else "❌ 未训练",
                "训练日期":  meta["train_date"][:10] if meta else "—",
                "验证准确率": f"{meta['val_acc']:.1%}" if meta else "—",
                "训练轮次":  meta.get("epochs_trained", "—") if meta else "—",
            })
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 执行训练")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("**单只股票训练**")
            if watchlist:
                train_choice = st.selectbox(
                    "选择股票",
                    [f"{k} {v}" for k, v in watchlist.items()],
                    key="train_single_sym",
                )
                if st.button("🚀 训练此股票", use_container_width=True, key="btn_train_single"):
                    _run_single_train(train_choice.split(" ")[0], config, fetcher, predictor)

        with col_t2:
            st.markdown("**批量训练（全部股票池）**")
            st.warning(f"将训练 {len(watchlist)} 只股票，约需1~4小时")
            if st.button("🚀 批量训练全部", use_container_width=True,
                         key="btn_train_all", type="secondary"):
                _run_batch_train(list(watchlist.keys()), config, fetcher, predictor)

        st.markdown("---")
        st.markdown("**快速积累准确率样本**")
        st.caption("训练完成后，可对历史数据批量预测，快速在AI预测页看到准确率数据。")
        col_bf1, col_bf2 = st.columns([2, 2])
        with col_bf1:
            if watchlist:
                bf_choice = st.selectbox(
                    "选择回填股票",
                    [f"{k} {v}" for k, v in watchlist.items()],
                    key="backfill_sym",
                )
            else:
                bf_choice = None
        with col_bf2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 回填历史准确率（60日）", use_container_width=True,
                         key="btn_backfill"):
                if bf_choice:
                    bf_symbol = bf_choice.split(" ")[0]
                    _run_backfill(bf_symbol, config, fetcher, predictor)

        with st.expander("模型参数（只读，修改请编辑 config.yaml）"):
            st.json(config.get("model", {}))

    # ══════════════════════════════════════════════════════
    # TAB 3：数据源配置
    # ══════════════════════════════════════════════════════
    with tab_data:
        st.markdown("### 数据源配置")
        st.info(
            "**主数据源：baostock**（免费、稳定、专为A股设计）\n\n"
            "**备用数据源：akshare**（主源失败时自动切换）\n\n"
            "两个数据源均无需注册账号，开箱即用。"
        )

        data_cfg = config.get("data", {})

        st.markdown("#### 连接状态检测")
        col_bs, col_ak = st.columns(2)

        with col_bs:
            if st.button("🔍 检测 baostock", use_container_width=True, key="test_bs"):
                with st.spinner("连接测试中..."):
                    _test_baostock()

        with col_ak:
            if st.button("🔍 检测 akshare（备用）", use_container_width=True, key="test_ak"):
                with st.spinner("连接测试中..."):
                    _test_akshare()

        st.markdown("---")
        st.markdown("#### 缓存设置")

        cache_days = st.slider(
            "本地缓存有效期（天）",
            min_value=1, max_value=30,
            value=data_cfg.get("cache_days", 7),
            help="行情数据缓存天数，到期后自动重新拉取",
        )
        adjust_type = st.selectbox(
            "复权方式",
            ["hfq（后复权）", "qfq（前复权）"],
            index=0 if data_cfg.get("adjust_type", "hfq") == "hfq" else 1,
        )

        if st.button("💾 保存缓存配置", key="save_data"):
            _save_config_section("data", {
                **data_cfg,
                "primary_source":  "baostock",
                "fallback_source": "akshare",
                "cache_days":      cache_days,
                "adjust_type":     adjust_type.split("（")[0],
            })
            st.success("✅ 缓存配置已保存")

    # ══════════════════════════════════════════════════════
    # TAB 4：风控参数
    # ══════════════════════════════════════════════════════
    with tab_risk:
        st.markdown("### 风控参数配置")

        risk_cfg = config.get("risk", {})

        st.markdown("**个股级风控**")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            stop_loss = st.slider(
                "固定止损 (%)", 3, 20,
                int(risk_cfg.get("stop_loss_pct", 0.08) * 100),
                help="亏损超过此比例时强制平仓",
            ) / 100
        with rc2:
            trailing = st.slider(
                "移动止损 (%)", 2, 15,
                int(risk_cfg.get("trailing_stop_pct", 0.05) * 100),
                help="从最高点回撤超过此比例时减仓",
            ) / 100
        with rc3:
            max_pos = st.slider(
                "单只最大仓位 (%)", 5, 40,
                int(risk_cfg.get("max_position_pct", 0.20) * 100),
            ) / 100

        st.markdown("**策略级风控**")
        sc1, sc2 = st.columns(2)
        with sc1:
            daily_limit = st.slider(
                "日亏损熔断 (%)", 1, 10,
                int(risk_cfg.get("daily_loss_limit", 0.03) * 100),
            ) / 100
        with sc2:
            max_dd = st.slider(
                "最大回撤防御触发 (%)", 5, 30,
                int(risk_cfg.get("max_drawdown_defensive", 0.15) * 100),
            ) / 100

        if st.button("💾 保存风控配置", key="save_risk"):
            _save_config_section("risk", {
                **risk_cfg,
                "stop_loss_pct":          stop_loss,
                "trailing_stop_pct":      trailing,
                "max_position_pct":       max_pos,
                "daily_loss_limit":       daily_limit,
                "max_drawdown_defensive": max_dd,
            })
            st.success("✅ 风控参数已保存，重启后生效")

        st.markdown("---")
        st.markdown("**当前风控状态**")
        risk_ctrl = state.get("risk")
        if risk_ctrl:
            summary = risk_ctrl.get_risk_summary()
            rs1, rs2, rs3 = st.columns(3)
            rs1.metric("日内亏损", f"{summary.get('daily_loss_pct', 0):.2%}")
            rs2.metric("连续亏损笔数", summary.get("consecutive_losses", 0))
            rs3.metric(
                "熔断状态",
                "⛔ 已触发" if summary.get("circuit_breaker") else "✅ 正常",
            )

    # ══════════════════════════════════════════════════════
    # TAB 5：数据注册表
    # ══════════════════════════════════════════════════════
    with tab_registry:
        st.markdown("### 数据注册表")
        st.caption("记录各股票数据的最后获取时间、行数和质量状态")

        if st.button("🔄 刷新注册表", key="refresh_registry"):
            st.rerun()

        try:
            df_reg = fetcher.get_registry_info()
            if df_reg.empty:
                st.info("注册表为空，请先在首页刷新数据")
            else:
                df_reg.columns = ["代码", "数据类型", "最新日期", "获取时间", "行数", "质量"]
                st.dataframe(df_reg, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"注册表读取失败: {e}")

        st.markdown("---")
        if st.button("🗑️ 清除所有 Parquet 缓存", key="clear_cache",
                     help="删除本地缓存，下次访问时重新从 baostock 拉取"):
            n = _clear_parquet_cache()
            st.success(f"✅ 已清除 {n} 个缓存文件")


# ── 数据源连接测试 ────────────────────────────────────────────

def _test_baostock():
    try:
        import baostock as bs
        ret = bs.login()
        if ret.error_code != "0":
            st.error(f"❌ baostock 登录失败: {ret.error_msg}")
            return
        rs = bs.query_history_k_data_plus(
            "sh.600519", "date,close",
            start_date="2024-12-01", end_date="2024-12-10",
            frequency="d", adjustflag="1",
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        if rows:
            st.success(f"✅ baostock 连接正常，测试获取 {len(rows)} 条数据（贵州茅台近期日线）")
        else:
            st.warning("baostock 登录成功，但测试数据为空，可能是非交易日")
    except ImportError:
        st.error("❌ baostock 未安装，请执行: pip install baostock")
    except Exception as e:
        st.error(f"❌ baostock 异常: {e}")


def _test_akshare():
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol="600519", period="daily",
            start_date="20241201", end_date="20241210",
            adjust="hfq",
        )
        if df is not None and not df.empty:
            st.success(f"✅ akshare（备用源）连接正常，测试获取 {len(df)} 条数据")
        else:
            st.warning("akshare 连接成功但返回空数据")
    except ImportError:
        st.warning("akshare 未安装（可选备用源），执行: pip install akshare")
    except Exception as e:
        st.warning(f"akshare（备用源）异常: {e}（不影响主数据源 baostock）")


# ── 训练工具函数 ──────────────────────────────────────────────

def _run_single_train(symbol: str, config: dict, fetcher, predictor):
    from core.factor_engine import FactorCalculator
    from models.trainer import ModelTrainer

    p_text = st.empty()
    p_bar  = st.progress(0)
    try:
        p_text.text(f"[{symbol}] 获取历史数据...")
        p_bar.progress(20)
        df_daily = fetcher.fetch_daily(symbol, force_refresh=True)
        df_fund  = fetcher.fetch_fundamental(symbol)
        if df_daily.empty:
            st.error(f"[{symbol}] 数据获取失败")
            return

        p_text.text(f"[{symbol}] 计算因子...")
        p_bar.progress(40)
        df_factors = FactorCalculator(config).calc_all(df_daily, df_fund)

        p_text.text(f"[{symbol}] 训练模型中（可能需要数分钟）...")
        p_bar.progress(60)
        result = ModelTrainer(config).train(symbol, df_factors)
        p_bar.progress(100)
        p_text.empty()
        p_bar.empty()

        if result.get("success"):
            predictor.invalidate_cache(symbol)
            st.success(
                f"✅ [{symbol}] 训练完成 | "
                f"验证准确率: {result.get('val_acc', 0):.1%} | "
                f"训练轮次: {result.get('epochs', 0)}"
            )
        else:
            st.error(f"❌ [{symbol}] 训练失败: {result.get('reason')}")
    except Exception as e:
        p_text.empty()
        p_bar.empty()
        st.error(f"❌ 训练异常: {e}")


def _run_batch_train(symbols: list, config: dict, fetcher, predictor):
    from core.factor_engine import FactorCalculator
    from models.trainer import ModelTrainer

    total    = len(symbols)
    progress = st.progress(0)
    status   = st.empty()
    results  = []

    for i, symbol in enumerate(symbols):
        status.text(f"训练进度 {i+1}/{total}: {symbol}")
        try:
            df_daily   = fetcher.fetch_daily(symbol)
            df_fund    = fetcher.fetch_fundamental(symbol)
            df_factors = FactorCalculator(config).calc_all(df_daily, df_fund)
            result     = ModelTrainer(config).train(symbol, df_factors)
            predictor.invalidate_cache(symbol)
            results.append({
                "symbol":  symbol,
                "success": result.get("success"),
                "val_acc": result.get("val_acc", 0),
                "reason":  result.get("reason", ""),
            })
        except Exception as e:
            results.append({"symbol": symbol, "success": False,
                             "val_acc": 0, "reason": str(e)})
        progress.progress((i + 1) / total)

    status.empty()
    progress.empty()
    ok = sum(1 for r in results if r["success"])
    st.success(f"✅ 批量训练完成: {ok}/{total} 只成功")

    df_r = pd.DataFrame(results)
    df_r.columns = ["代码", "成功", "验证准确率", "失败原因"]
    st.dataframe(df_r, use_container_width=True, hide_index=True)


# ── 配置 I/O ─────────────────────────────────────────────────

def _save_watchlist(watchlist: dict):
    _save_config_section("watchlist", watchlist)


def _save_config_section(section: str, data):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        full[section] = data
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(full, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)
    except Exception as e:
        st.error(f"配置保存失败: {e}")


def _run_backfill(symbol: str, config: dict, fetcher, predictor):
    """回填指定股票的历史预测准确率"""
    from core.factor_engine import FactorCalculator
    try:
        df_daily   = fetcher.fetch_daily(symbol)
        df_fund    = fetcher.fetch_fundamental(symbol)
        df_factors = FactorCalculator(config).calc_all(df_daily, df_fund)
        n = predictor.backfill_history(symbol, df_factors, days=60)
        if n > 0:
            st.success(f"✅ [{symbol}] 成功回填 {n} 条历史预测记录")
        else:
            st.info(f"[{symbol}] 无需回填（记录已存在）或模型尚未训练")
    except Exception as e:
        st.error(f"[{symbol}] 回填失败: {e}")


def _clear_parquet_cache() -> int:
    parquet_dir = Path(__file__).parent.parent.parent / "data" / "parquet"
    deleted = 0
    for f in parquet_dir.glob("*.parquet"):
        try:
            f.unlink()
            deleted += 1
        except Exception:
            pass
    return deleted
