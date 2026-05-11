# ui/components/charts.py
"""
可复用图表组件（Plotly）
配色遵循中国股市习惯：红色=上涨/买入，绿色=下跌/卖出
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 中国股市配色常量 ──────────────────────────────────────────
CN_UP     = "#E53935"   # 主红（上涨、阳线、买入信号）
CN_UP_L   = "#EF9A9A"   # 浅红（热力图正收益）
CN_UP_D   = "#B71C1C"   # 深红（强买入）
CN_DOWN   = "#26A69A"   # 主绿（下跌、阴线、卖出信号）
CN_DOWN_L = "#80CBC4"   # 浅绿（热力图负收益）
CN_DOWN_D = "#004D40"   # 深绿（强卖出）
NEUTRAL   = "#9E9E9E"   # 灰色（观望、中性）
BLUE      = "#2196F3"   # 蓝色（资金曲线、策略净值）


def make_candlestick(
    df: pd.DataFrame,
    title: str = "",
    indicators: Optional[list] = None,
    signals_df: Optional[pd.DataFrame] = None,
    height: int = 600,
) -> go.Figure:
    """
    K线图 + 技术指标叠加 + 信号标注
    阳线红色，阴线绿色（中国股市习惯）
    """
    if df.empty:
        return go.Figure()

    indicators = indicators or []
    has_macd   = any(c in df.columns for c in ["macd_dif", "macd_dea"])
    row_count  = 3 if has_macd else 2
    row_heights = [0.55, 0.25, 0.20] if has_macd else [0.70, 0.30]

    fig = make_subplots(
        rows=row_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=["", "MACD", "成交量"] if has_macd else ["", "成交量"],
    )

    # ── K线：阳线红色，阴线绿色 ──
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name="K线",
            increasing_line_color=CN_UP,
            decreasing_line_color=CN_DOWN,
            increasing_fillcolor=CN_UP,
            decreasing_fillcolor=CN_DOWN,
        ),
        row=1, col=1,
    )

    # ── 均线/布林带叠加 ──
    line_colors = {
        "sma5":     "#FF9800",
        "sma10":    "#FFD54F",
        "sma20":    "#CE93D8",
        "sma60":    "#80DEEA",
        "ema12":    "#A5D6A7",
        "ema26":    "#FFAB91",
        "bb_upper": "#78909C",
        "bb_lower": "#78909C",
        "bb_mid":   "#90A4AE",
    }
    dash_styles = {"bb_upper": "dash", "bb_lower": "dash", "bb_mid": "dot"}

    for ind in indicators:
        if ind in df.columns and ind in line_colors:
            fig.add_trace(
                go.Scatter(
                    x=df["date"], y=df[ind],
                    name=ind.upper(),
                    line=dict(
                        color=line_colors[ind],
                        width=1,
                        dash=dash_styles.get(ind, "solid"),
                    ),
                    opacity=0.85,
                ),
                row=1, col=1,
            )

    # 布林带填充
    if "bb_upper" in indicators and "bb_lower" in indicators:
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([df["date"], df["date"][::-1]]),
                    y=pd.concat([df["bb_upper"], df["bb_lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(120,144,156,0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="布林带",
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # ── 信号标注：买入红色三角向上，卖出绿色三角向下 ──
    if signals_df is not None and not signals_df.empty:
        buy_signals  = signals_df[signals_df["signal_type"].isin(["buy", "strong_buy"])]
        sell_signals = signals_df[signals_df["signal_type"].isin(["sell", "strong_sell"])]

        if not buy_signals.empty:
            bp = df.merge(buy_signals[["date"]], on="date", how="inner")
            if not bp.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bp["date"], y=bp["low"] * 0.98,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=12, color=CN_UP),
                        name="买入信号",
                    ),
                    row=1, col=1,
                )

        if not sell_signals.empty:
            sp = df.merge(sell_signals[["date"]], on="date", how="inner")
            if not sp.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sp["date"], y=sp["high"] * 1.02,
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=12, color=CN_DOWN),
                        name="卖出信号",
                    ),
                    row=1, col=1,
                )

    # ── MACD：柱正红负绿 ──
    if has_macd and "macd_dif" in df.columns:
        macd_row = 2
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["macd_dif"], name="DIF",
                       line=dict(color="#2196F3", width=1)),
            row=macd_row, col=1,
        )
        if "macd_dea" in df.columns:
            fig.add_trace(
                go.Scatter(x=df["date"], y=df["macd_dea"], name="DEA",
                           line=dict(color="#FF9800", width=1)),
                row=macd_row, col=1,
            )
        if "macd_bar" in df.columns:
            bar_colors = df["macd_bar"].apply(
                lambda v: CN_UP if v >= 0 else CN_DOWN
            )
            fig.add_trace(
                go.Bar(x=df["date"], y=df["macd_bar"], name="MACD柱",
                       marker_color=bar_colors),
                row=macd_row, col=1,
            )

    # ── 成交量：阳线红，阴线绿 ──
    vol_row = 3 if has_macd else 2
    vol_colors = df.apply(
        lambda r: CN_UP if r["close"] >= r["open"] else CN_DOWN, axis=1
    )
    fig.add_trace(
        go.Bar(x=df["date"], y=df["volume"], name="成交量",
               marker_color=vol_colors, opacity=0.7),
        row=vol_row, col=1,
    )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0.3)",
        ),
        margin=dict(l=50, r=20, t=60, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    if has_macd:
        fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="成交量", row=vol_row, col=1)
    return fig


def make_equity_curve(
    equity_data: list,
    symbol: str = "",
    init_cash: float = 1_000_000,
    buy_hold_return: Optional[float] = None,
    height: int = 400,
) -> go.Figure:
    """
    资金曲线图 + 回撤阴影
    资金曲线用蓝色，回撤区域用浅红（亏损=负向=红色）
    """
    if not equity_data:
        return go.Figure()

    df_eq = pd.DataFrame(equity_data)
    df_eq["date"]     = pd.to_datetime(df_eq["date"])
    df_eq["return"]   = (df_eq["equity"] - init_cash) / init_cash
    rolling_max       = df_eq["equity"].cummax()
    df_eq["drawdown"] = (df_eq["equity"] - rolling_max) / rolling_max

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.70, 0.30],
        subplot_titles=[f"{symbol} 资金曲线", "回撤（%）"],
    )

    fig.add_trace(
        go.Scatter(
            x=df_eq["date"], y=df_eq["equity"],
            name="策略净值",
            line=dict(color=BLUE, width=2),
            hovertemplate="日期: %{x}<br>资产: ¥%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    if buy_hold_return is not None:
        bh = [
            init_cash * (1 + buy_hold_return * i / max(len(df_eq) - 1, 1))
            for i in range(len(df_eq))
        ]
        fig.add_trace(
            go.Scatter(
                x=df_eq["date"], y=bh,
                name="买入持有",
                line=dict(color=NEUTRAL, width=1, dash="dash"),
                opacity=0.6,
            ),
            row=1, col=1,
        )

    # 回撤区域：红色填充（回撤=亏损=负向=红色，符合中国习惯）
    fig.add_trace(
        go.Scatter(
            x=df_eq["date"], y=df_eq["drawdown"] * 100,
            name="回撤%",
            fill="tozeroy",
            line=dict(color=CN_UP, width=0.5),
            fillcolor="rgba(229,57,53,0.25)",
            hovertemplate="回撤: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=height,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="资产 (¥)", row=1, col=1, tickformat=",.0f")
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
    return fig


def make_monthly_heatmap(monthly_returns: dict, height: int = 300) -> go.Figure:
    """
    月度收益热力图
    正收益（赚钱）→ 红色，负收益（亏钱）→ 绿色（中国股市习惯）
    """
    if not monthly_returns:
        return go.Figure()

    data = {}
    for key, val in monthly_returns.items():
        year, month = key.split("-")
        year, month = int(year), int(month)
        if year not in data:
            data[year] = {}
        data[year][month] = val * 100

    years  = sorted(data.keys())
    months = list(range(1, 13))
    month_labels = ["1月","2月","3月","4月","5月","6月",
                    "7月","8月","9月","10月","11月","12月"]

    z    = [[data.get(y, {}).get(m, None) for m in months] for y in years]
    text = [[f"{v:.1f}%" if v is not None else "" for v in row] for row in z]

    # 正收益红色，负收益绿色（中国习惯）
    fig = go.Figure(go.Heatmap(
        z=z, x=month_labels, y=[str(y) for y in years],
        text=text, texttemplate="%{text}",
        colorscale=[
            [0.0,  CN_DOWN_D],   # 深绿（大跌）
            [0.35, CN_DOWN],     # 主绿（小跌）
            [0.5,  "#F5F5F5"],   # 白色（持平）
            [0.65, CN_UP],       # 主红（小涨）
            [1.0,  CN_UP_D],     # 深红（大涨）
        ],
        zmid=0,
        colorbar=dict(title="收益率%"),
    ))
    fig.update_layout(
        title="月度收益热力图（红=盈利，绿=亏损）",
        height=height,
        template="plotly_dark",
        margin=dict(l=50, r=20, t=40, b=20),
    )
    return fig


def make_probability_gauge(probability: float, symbol: str = "") -> go.Figure:
    """
    概率仪表盘
    高概率（看涨）→ 红色，低概率（看跌）→ 绿色（中国习惯）
    """
    pct = probability * 100

    if pct >= 60:
        color, label = CN_UP,   "看涨"
    elif pct >= 50:
        color, label = "#FF7043","偏多"
    elif pct >= 40:
        color, label = NEUTRAL,  "中性"
    else:
        color, label = CN_DOWN,  "看跌"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"{symbol} 次日上涨概率", "font": {"size": 16}},
        delta={"reference": 50, "valueformat": ".1f"},
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  40],  "color": f"rgba(38,166,154,0.2)"},   # 绿（看跌区）
                {"range": [40, 55],  "color": "rgba(158,158,158,0.2)"},   # 灰（中性区）
                {"range": [55, 100], "color": f"rgba(229,57,53,0.2)"},    # 红（看涨区）
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": 55,
            },
        },
    ))
    fig.add_annotation(
        x=0.5, y=0.2, xref="paper", yref="paper",
        text=f"<b>{label}</b>",
        font=dict(size=20, color=color),
        showarrow=False,
    )
    fig.update_layout(
        height=300,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def make_rsi_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """RSI图，超买线红色，超卖线绿色"""
    if "rsi14" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["rsi14"], name="RSI(14)",
                   line=dict(color=BLUE, width=1.5))
    )
    # 超买70：红色警示线（接近顶部，可能下跌）
    fig.add_hline(y=70, line_dash="dash", line_color=CN_UP,
                  annotation_text="超买70", annotation_position="right")
    # 超卖30：绿色警示线（接近底部，可能反弹）
    fig.add_hline(y=30, line_dash="dash", line_color=CN_DOWN,
                  annotation_text="超卖30", annotation_position="right")
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.05)", line_width=0)

    fig.update_layout(
        title="RSI(14)",
        height=height,
        template="plotly_dark",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=50, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig
