# core/backtest_engine.py
"""
回测引擎（VectorBT封装）
- 事件驱动信号重放
- 交易成本：印花税0.1%（卖出）+ 佣金0.025%（双向）+ 滑点0.1%
- 报告：年化收益/Sharpe/MDD/Calmar/胜率/盈亏比
- 基准对比：沪深300
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    回测运行器

    用法:
        runner = BacktestRunner(config)
        report = runner.run(
            symbol="600519",
            df_factors=df_factors,
            predictor=predictor,
            signal_gen=signal_gen,
            start_date="20220101",
            end_date="20241231",
        )
    """

    def __init__(self, config: dict):
        self.config = config
        bt_cfg = config.get("backtest", {})
        self.init_cash    = bt_cfg.get("init_cash", 1_000_000)
        self.commission   = bt_cfg.get("commission", 0.00025)     # 双向0.025%
        self.stamp_duty   = bt_cfg.get("stamp_duty", 0.001)       # 卖出0.1%
        self.slippage     = bt_cfg.get("slippage", 0.001)         # 0.1%
        self.benchmark    = bt_cfg.get("benchmark", "000300")
        self.seq_len      = config.get("model", {}).get("sequence_length", 60)

    # ── 主回测入口 ───────────────────────────────────────────

    def run(
        self,
        symbol: str,
        df_factors: pd.DataFrame,
        predictor,
        signal_gen,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        df_benchmark: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        运行回测

        Args:
            symbol: 股票代码
            df_factors: 含所有因子的DataFrame
            predictor: LSTMPredictor实例
            signal_gen: SignalGenerator实例
            start_date: 回测开始日期 "YYYYMMDD"
            end_date: 回测结束日期 "YYYYMMDD"
            df_benchmark: 基准指数数据

        Returns:
            完整回测报告字典
        """
        if df_factors.empty:
            return {"success": False, "reason": "no_data"}

        logger.info(f"[{symbol}] 开始回测 {start_date} ~ {end_date}")

        # 日期过滤
        # 注意：需要保留 seq_len 天预热数据用于生成信号，不能直接按 start_date 截断
        df_full = df_factors.copy().sort_values("date").reset_index(drop=True)
        if end_date:
            df_full = df_full[df_full["date"] <= pd.to_datetime(end_date)]

        # 计算用户期望的回测起始位置
        user_start = pd.to_datetime(start_date) if start_date else df_full["date"].iloc[0]

        # df 需要包含 seq_len 天预热窗口，信号生成时只在 user_start 后输出
        warmup_start_idx = max(0, df_full[df_full["date"] >= user_start].index.min() - self.seq_len)
        df = df_full.iloc[warmup_start_idx:].reset_index(drop=True)

        if len(df) < self.seq_len + 5:
            min_days = self.seq_len + 5
            return {
                "success": False,
                "reason": "insufficient_data",
                "message": (
                    f"回测区间数据不足（仅 {len(df)} 行，需至少 {min_days} 行）。"
                    f"请将回测开始日期至少提前到 {min_days} 个交易日之前，"
                    "或确认该股票在此区间内有足够交易记录。"
                ),
            }

        # 生成历史信号序列
        signals_df = self._generate_signal_series(symbol, df, predictor, signal_gen, user_start)

        if signals_df.empty:
            return {"success": False, "reason": "signal_generation_failed"}

        # 执行回测
        try:
            report = self._run_vectorbt(symbol, df, signals_df, df_benchmark)
            report["success"] = True
            logger.info(
                f"[{symbol}] 回测完成 | "
                f"年化:{report.get('annual_return', 0):.1%} | "
                f"Sharpe:{report.get('sharpe_ratio', 0):.2f} | "
                f"MDD:{report.get('max_drawdown', 0):.1%}"
            )
            return report
        except Exception as e:
            logger.error(f"[{symbol}] 回测执行失败: {e}")
            # 降级到手动回测
            return self._run_manual_backtest(symbol, df, signals_df, df_benchmark)

    # ── 信号序列生成 ─────────────────────────────────────────

    def _generate_signal_series(
        self, symbol: str, df: pd.DataFrame, predictor, signal_gen,
        user_start: "pd.Timestamp | None" = None,
    ) -> pd.DataFrame:
        """
        遍历历史数据，在每个交易日生成信号
        严格使用T日收盘后数据，模拟T+1执行
        """
        records = []
        model, scaler, meta = None, None, None

        # 尝试加载已训练模型
        from models.trainer import ModelTrainer
        model, scaler, meta = ModelTrainer.load_model(symbol)

        if model is None:
            logger.warning(f"[{symbol}] 无训练模型，使用规则信号（基于技术指标）")

        for i in range(self.seq_len, len(df)):
            df_window = df.iloc[: i + 1].copy()   # T日及之前（严格无未来数据）
            current_date = df["date"].iloc[i]
            # 只在用户指定回测区间内记录信号（预热窗口内的跳过）
            if current_date < user_start:
                continue

            # 生成预测
            if model is not None:
                pred_result = predictor.predict(symbol, df_window)
            else:
                pred_result = self._rule_based_signal(df_window)

            # 生成信号
            signal = signal_gen.generate(symbol, pred_result, df_window)

            records.append({
                "date":        current_date,
                "close":       df["close"].iloc[i],
                "signal_type": signal.signal_type,
                "probability": signal.probability,
                "suggested_pos": signal.suggested_pos,
            })

        return pd.DataFrame(records)

    def _rule_based_signal(self, df: pd.DataFrame) -> dict:
        """
        无模型时的规则信号（基于技术指标）
        用于回测框架测试或模型未训练时
        """
        if len(df) < 2:
            return {"success": True, "probability": 0.5, "confidence": "low"}

        signals = []

        # RSI信号
        if "rsi14" in df.columns:
            rsi = df["rsi14"].iloc[-1]
            if not pd.isna(rsi):
                if rsi < 30:
                    signals.append(0.65)
                elif rsi > 70:
                    signals.append(0.35)
                else:
                    signals.append(0.5 + (50 - rsi) / 200)

        # MACD信号
        if "macd_dif" in df.columns and "macd_dea" in df.columns:
            dif = df["macd_dif"].iloc[-1]
            dea = df["macd_dea"].iloc[-1]
            if not pd.isna(dif) and not pd.isna(dea):
                if dif > dea:
                    signals.append(0.58)
                else:
                    signals.append(0.42)

        prob = float(np.mean(signals)) if signals else 0.5
        return {"success": True, "probability": prob, "confidence": "low", "degraded": False}

    # ── VectorBT回测执行 ──────────────────────────────────────

    def _run_vectorbt(
        self,
        symbol: str,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
        df_benchmark: Optional[pd.DataFrame],
    ) -> dict:
        """使用VectorBT执行向量化回测"""
        import vectorbt as vbt

        merged = df.merge(signals_df[["date", "signal_type"]], on="date", how="left")
        merged["signal_type"] = merged["signal_type"].fillna("hold")

        close = merged.set_index("date")["close"]
        entries = (merged["signal_type"].isin(["buy", "strong_buy"])).values
        exits   = (merged["signal_type"].isin(["sell", "strong_sell"])).values

        # 综合费率：买入（佣金+滑点）= 0.00025+0.001 = 0.00125
        #           卖出（佣金+印花税+滑点）= 0.00025+0.001+0.001 = 0.00225
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            direction="longonly",
            init_cash=self.init_cash,
            fees=self.commission + self.slippage,   # 买入费率
            slippage=0.0,                           # 已含在fees中
            freq="1D",
        )

        stats = pf.stats()
        trades = pf.trades.records_readable if hasattr(pf.trades, "records_readable") else pd.DataFrame()

        # 基准收益
        benchmark_return = None
        if df_benchmark is not None and not df_benchmark.empty:
            bm = df_benchmark.merge(merged[["date"]], on="date", how="inner")
            if len(bm) > 1:
                benchmark_return = float(bm["close"].iloc[-1] / bm["close"].iloc[0] - 1)

        # 月度收益热力图数据
        monthly_returns = self._calc_monthly_returns(pf)

        # ── 安全解析 stats（兼容 vectorbt 0.26~0.28）──────────
        def _safe_pct(key, default=0.0):
            v = stats.get(key, default)
            try:
                f = float(v)
                return 0.0 if (f != f or f == float("inf") or f == float("-inf")) else f / 100
            except Exception:
                return default / 100

        def _safe_float(key, default=0.0):
            v = stats.get(key, default)
            try:
                f = float(v)
                return 0.0 if (f != f or f == float("inf") or f == float("-inf")) else f
            except Exception:
                return default

        def _safe_days(key):
            """解析 Timedelta / NaT / str 为天数"""
            v = stats.get(key)
            if v is None:
                return 0.0
            try:
                import pandas as _pd
                if _pd.isnull(v):
                    return 0.0
                if hasattr(v, "days"):
                    return float(v.days)
                return float(str(v).split(" ")[0])
            except Exception:
                return 0.0

        # vectorbt 0.28+ 没有 Annual Return，手动计算
        total_ret = _safe_pct("Total Return [%]")
        equity_curve_tmp = self._extract_equity_curve(pf)
        n_days = len(equity_curve_tmp) or 1
        annual_return = (1 + total_ret) ** (252 / n_days) - 1

        return {
            # 核心指标
            "annual_return":    annual_return,
            "total_return":     total_ret,
            "sharpe_ratio":     _safe_float("Sharpe Ratio"),
            "max_drawdown":     _safe_pct("Max Drawdown [%]"),
            "calmar_ratio":     _safe_float("Calmar Ratio"),
            "win_rate":         _safe_pct("Win Rate [%]"),
            "profit_factor":    _safe_float("Profit Factor"),
            # 交易统计
            "total_trades":     int(stats.get("Total Closed Trades", 0) or 0),
            "avg_holding_days": _safe_days("Avg Winning Trade Duration"),
            "best_trade":       _safe_pct("Best Trade [%]"),
            "worst_trade":      _safe_pct("Worst Trade [%]"),
            # 对比基准
            "benchmark_return":   benchmark_return,
            "buy_hold_return":    float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0,
            # 数据
            "equity_curve":       equity_curve_tmp,
            "monthly_returns":    monthly_returns,
            "trades":             trades.to_dict("records") if not trades.empty else [],
            "signals_df":         signals_df.to_dict("records"),
            # 元信息
            "symbol":  symbol,
            "start":   str(merged["date"].iloc[0].date()),
            "end":     str(merged["date"].iloc[-1].date()),
            "init_cash": self.init_cash,
        }

    # ── 手动回测（VectorBT不可用时的降级方案）───────────────────

    def _run_manual_backtest(
        self,
        symbol: str,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
        df_benchmark: Optional[pd.DataFrame],
    ) -> dict:
        """纯Pandas实现的简化回测（降级方案）"""
        logger.info(f"[{symbol}] 使用降级回测引擎（Pandas实现）")

        merged = df.merge(signals_df[["date", "signal_type"]], on="date", how="left")
        merged["signal_type"] = merged["signal_type"].fillna("hold")
        merged = merged.set_index("date")

        cash = float(self.init_cash)
        shares = 0.0
        entry_price = 0.0
        equity_curve = []
        trades = []

        for date, row in merged.iterrows():
            close = row["close"]
            sig = row["signal_type"]
            portfolio_value = cash + shares * close

            if sig in ("buy", "strong_buy") and shares == 0 and cash > 0:
                buy_cost = cash * self.config.get("position", {}).get("base_position_pct", 0.10)
                fee = buy_cost * (self.commission + self.slippage)
                shares = (buy_cost - fee) / close
                cash -= buy_cost
                entry_price = close

            elif sig in ("sell", "strong_sell") and shares > 0:
                sell_value = shares * close
                fee = sell_value * (self.commission + self.stamp_duty + self.slippage)
                pnl = sell_value - fee - (shares * entry_price)
                trades.append({
                    "entry_price": entry_price,
                    "exit_price": close,
                    "pnl_pct": (close - entry_price) / entry_price,
                    "exit_date": str(date.date()),
                })
                cash += sell_value - fee
                shares = 0.0
                entry_price = 0.0

            equity_curve.append({
                "date": str(date.date()),
                "equity": cash + shares * close,
            })

        # 计算绩效
        equity = [e["equity"] for e in equity_curve]
        total_return = (equity[-1] - self.init_cash) / self.init_cash if equity else 0
        trading_days = len(equity)
        annual_return = (1 + total_return) ** (252 / max(trading_days, 1)) - 1

        daily_returns = pd.Series(equity).pct_change().dropna()
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

        equity_series = pd.Series(equity)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        winning_trades = [t for t in trades if t["pnl_pct"] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0

        return {
            "success": True,
            "annual_return":   annual_return,
            "total_return":    total_return,
            "sharpe_ratio":    sharpe,
            "max_drawdown":    max_dd,
            "calmar_ratio":    calmar,
            "win_rate":        win_rate,
            "profit_factor":   0.0,
            "total_trades":    len(trades),
            "avg_holding_days": 0,
            "best_trade":      max((t["pnl_pct"] for t in trades), default=0),
            "worst_trade":     min((t["pnl_pct"] for t in trades), default=0),
            "benchmark_return": None,
            "buy_hold_return": float(df["close"].iloc[-1] / df["close"].iloc[0] - 1) if len(df) > 1 else 0,
            "equity_curve":    equity_curve,
            "monthly_returns": {},
            "trades":          trades,
            "signals_df":      signals_df.to_dict("records"),
            "symbol":  symbol,
            "start":   str(df["date"].iloc[0].date()),
            "end":     str(df["date"].iloc[-1].date()),
            "init_cash": self.init_cash,
            "engine":  "manual_fallback",
        }

    # ── 工具方法 ─────────────────────────────────────────────

    def _extract_equity_curve(self, pf) -> list:
        """提取资金曲线"""
        try:
            value = pf.value()
            return [
                {"date": str(d.date()), "equity": float(v)}
                for d, v in value.items()
            ]
        except Exception:
            return []

    def _calc_monthly_returns(self, pf) -> dict:
        """计算月度收益（用于热力图）"""
        try:
            value = pf.value()
            monthly = value.resample("ME").last().pct_change()
            result = {}
            for date, ret in monthly.items():
                if not pd.isna(ret):
                    key = f"{date.year}-{date.month:02d}"
                    result[key] = round(float(ret), 4)
            return result
        except Exception:
            return {}
