# ui/pages/ai_predict.py
"""
AI预测页
- 次日涨跌概率仪表盘（同步 ≤1秒）
- 历史预测准确率 + 一键回填历史记录
- SHAP特征贡献图（按需触发）
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime


def render(state: dict) -> None:
    from ui.components.charts import make_probability_gauge
    from ui.components.widgets import stock_search_box, signal_badge

    config     = state["config"]
    fetcher    = state["fetcher"]
    predictor  = state["predictor"]
    signal_gen = state["signal_gen"]

    st.title("🤖 AI 预测中心")

    stock_list = fetcher.fetch_stock_list()
    watchlist  = config.get("watchlist", {})

    # ── 股票选择 ──────────────────────────────────────────────
    col_sym, col_btn = st.columns([3, 1])
    with col_sym:
        symbol = stock_search_box(stock_list, key="predict_symbol", default="600519")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        do_predict = st.button("🚀 开始预测", use_container_width=True)

    # ── 执行预测 ──────────────────────────────────────────────
    cache_key = f"pred_{symbol}"
    if do_predict or cache_key not in st.session_state:
        with st.spinner(f"加载 {symbol} 数据并推理..."):
            from core.factor_engine import FactorCalculator
            df_daily   = fetcher.fetch_daily(symbol)
            df_fund    = fetcher.fetch_fundamental(symbol)
            calc       = FactorCalculator(config)
            df_factors = calc.calc_all(df_daily, df_fund)

            if df_factors.empty:
                st.error("数据加载失败，请检查网络连接或稍后重试")
                return

            pred_result = predictor.predict(symbol, df_factors)
            acc_stats   = predictor.get_accuracy_stats(symbol)
            signal      = signal_gen.generate(symbol, pred_result, df_factors, acc_stats)

            st.session_state[cache_key] = {
                "pred":    pred_result,
                "acc":     acc_stats,
                "sig":     signal,
                "df":      df_factors,
            }

    cached = st.session_state.get(cache_key)
    if not cached:
        st.info("请点击【开始预测】按钮")
        return

    pred_result = cached["pred"]
    acc_stats   = cached["acc"]
    signal      = cached["sig"]
    df_factors  = cached["df"]
    name        = watchlist.get(symbol, symbol)

    st.markdown(f"### {name}（{symbol}）— 次日预测")

    # ── 预测结果 ──────────────────────────────────────────────
    if not pred_result.get("success"):
        reason = pred_result.get("reason", "")
        if reason == "no_model":
            st.warning(
                f"股票 {symbol} 尚未训练模型。\n\n"
                "请前往【系统配置 → 模型训练】执行训练后再返回此页。"
            )
        else:
            st.error(f"预测失败: {pred_result.get('message', '未知错误')}")
        return

    if pred_result.get("degraded"):
        st.warning("⚠️ 模型输出异常，已降级为观望信号")

    col_gauge, col_signal = st.columns([1, 1])
    prob = pred_result.get("probability", 0.5)

    with col_gauge:
        fig_gauge = make_probability_gauge(prob, symbol)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_signal:
        st.markdown("<br><br>", unsafe_allow_html=True)
        signal_badge(signal.signal_type, prob, symbol)
        st.markdown(f"**触发原因**: {signal.reason}")
        st.markdown(f"**建议仓位**: {signal.suggested_pos:.0%}")
        state_cn = {"low": "低波动", "medium": "中等", "high": "高波动"}.get(
            signal.market_state, "中等"
        )
        st.markdown(f"**市场波动**: {state_cn}")
        conf_cn = {"high": "🟢 高", "medium": "🟡 中", "low": "🔴 低"}.get(
            pred_result.get("confidence", "low"), "低"
        )
        st.markdown(f"**预测置信度**: {conf_cn}")

    # 模型信息
    meta = pred_result.get("model_meta", {})
    if meta:
        with st.expander("模型信息", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("训练日期",   meta.get("train_date", "")[:10])
            c2.metric("验证准确率", f"{meta.get('val_acc', 0):.1%}")
            c3.metric("特征数量",   len(meta.get("feature_cols", [])))
            c4.metric("训练轮次",   meta.get("epochs_trained", 0))

    st.markdown("---")

    # ── 历史预测准确率 ────────────────────────────────────────
    st.markdown("### 📊 历史预测准确率")

    # 回填按钮
    col_fill, col_tip = st.columns([1, 3])
    with col_fill:
        do_backfill = st.button(
            "📥 回填历史准确率",
            key="backfill_btn",
            help="对最近60个交易日的历史数据批量预测，快速积累准确率样本",
        )
    with col_tip:
        st.caption(
            "首次使用时点击【回填历史准确率】，系统将自动对最近60个交易日"
            "的历史数据做批量预测，约需30~60秒，之后准确率数据即可显示。"
        )

    if do_backfill:
        with st.spinner("正在回填历史预测记录（最近60个交易日）..."):
            df_for_fill = df_factors
            n = predictor.backfill_history(symbol, df_for_fill, days=60)
            if n > 0:
                st.success(f"✅ 成功回填 {n} 条历史预测记录，准确率数据已更新")
                # 刷新准确率
                acc_stats = predictor.get_accuracy_stats(symbol)
                st.session_state[cache_key]["acc"] = acc_stats
            else:
                st.info("无需回填（记录已存在）或模型尚未训练")

    # 显示准确率
    has_enough = acc_stats.get("has_enough_data", False)
    hint       = acc_stats.get("hint", "")

    if not has_enough:
        st.info(hint if hint else "预测记录不足，请点击上方【回填历史准确率】")
    else:
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric(
            "近期准确率",
            f"{acc_stats.get('accuracy', 0):.1%}",
            help="基于有实际结果的历史预测计算",
        )
        ac2.metric("正确次数", acc_stats.get("correct", 0))
        ac3.metric("统计样本", acc_stats.get("total", 0))

        recent = acc_stats.get("recent_predictions", [])
        if recent:
            df_hist = pd.DataFrame(recent)
            df_hist["预测方向"] = df_hist["probability"].apply(
                lambda p: "⬆️ 看涨" if p > 0.5 else "⬇️ 看跌"
            )
            df_hist["实际结果"] = df_hist["actual_up"].apply(
                lambda v: "✅ 涨" if v == 1 else ("❌ 跌" if v == 0 else "⏳ 待定")
            )
            df_hist["概率"] = df_hist["probability"].apply(lambda p: f"{p:.1%}")
            st.dataframe(
                df_hist[["date", "预测方向", "概率", "实际结果"]].rename(
                    columns={"date": "日期"}
                ),
                use_container_width=True,
                hide_index=True,
                height=280,
            )

    st.markdown("---")

    # ── SHAP 特征贡献图 ───────────────────────────────────────
    st.markdown("### 🔍 AI解释（Top-5贡献因子）")
    shap_key = f"shap_{symbol}"

    col_shap_btn, col_shap_tip = st.columns([1, 3])
    with col_shap_btn:
        do_explain = st.button("查看AI解释", key="shap_btn",
                               help="计算特征贡献度（约10~30秒）")
    with col_shap_tip:
        st.caption("⏱️ 按需触发，结果缓存24小时")

    if do_explain:
        with st.spinner("正在计算特征贡献度..."):
            shap_result = _compute_importance(symbol, df_factors, pred_result)
            st.session_state[shap_key] = shap_result

    shap_result = st.session_state.get(shap_key)
    if shap_result:
        _render_importance(shap_result)


# ── 特征重要性计算（置换法）────────────────────────────────────

def _compute_importance(symbol: str, df_factors: pd.DataFrame, pred_result: dict) -> dict:
    from models.trainer import ModelTrainer
    import torch
    import numpy as np

    model, scaler, meta = ModelTrainer.load_model(symbol)
    if model is None or meta is None:
        return {"error": "模型未加载"}

    feature_cols = meta.get("feature_cols", [])
    seq_len      = meta.get("seq_len", 60)
    available    = [c for c in feature_cols if c in df_factors.columns]
    if not available:
        return {"error": "特征列不可用"}

    df_recent = df_factors.tail(seq_len)[available].ffill().bfill().fillna(0)
    vals = df_recent.values.astype(np.float32)
    if scaler:
        try:
            vals = scaler.transform(vals)
        except Exception:
            pass

    x_base = torch.FloatTensor(vals).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        base_prob = model(x_base).item()

    importances = []
    for i, feat in enumerate(available):
        x_perm = x_base.clone()
        x_perm[0, :, i] = 0.0
        with torch.no_grad():
            perm_prob = model(x_perm).item()
        importances.append({
            "feature":    feat,
            "importance": base_prob - perm_prob,
        })

    importances.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return {"features": importances[:10], "base_prob": base_prob}


def _render_importance(result: dict) -> None:
    if "error" in result:
        st.warning(f"解释计算失败: {result['error']}")
        return

    features = result.get("features", [])[:5]
    if not features:
        st.info("无特征数据")
        return

    label_map = {
        "rsi14": "RSI(14)", "macd_dif": "MACD-DIF", "macd_bar": "MACD柱",
        "macd_dea": "MACD-DEA", "bb_pct": "布林带位置", "bb_width": "布林带宽",
        "atr14": "ATR波动", "atr_ratio": "ATR比率",
        "sma5": "SMA5", "sma20": "SMA20", "sma60": "SMA60",
        "ema12": "EMA12", "ema26": "EMA26",
        "volume_ratio": "量比", "obv": "OBV",
        "kdj_k": "KDJ-K", "kdj_d": "KDJ-D", "kdj_j": "KDJ-J",
        "price_to_sma5": "价格/SMA5", "price_to_sma20": "价格/SMA20",
        "pe_ttm": "PE(TTM)", "pb": "PB", "roe_ttm": "ROE",
    }

    df_plot = pd.DataFrame(features)
    df_plot["label"]  = df_plot["feature"].map(label_map).fillna(df_plot["feature"])
    # 贡献度>0 推升上涨（红色），<0 压低概率（绿色）
    df_plot["color"]  = df_plot["importance"].apply(
        lambda v: "#E53935" if v > 0 else "#26A69A"
    )
    df_plot["text"]   = df_plot["importance"].apply(lambda v: f"{v:+.4f}")

    fig = go.Figure(go.Bar(
        x=df_plot["importance"],
        y=df_plot["label"],
        orientation="h",
        marker_color=df_plot["color"],
        text=df_plot["text"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top-5 特征贡献度（红=推升概率，绿=压低概率）",
        height=280,
        template="plotly_dark",
        xaxis_title="贡献度",
        margin=dict(l=130, r=80, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"基准预测概率: {result.get('base_prob', 0):.1%} | "
        "基于特征置换法计算（将该特征置零后概率变化量）"
    )
