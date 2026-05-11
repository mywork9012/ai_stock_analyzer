# core/signal_generator.py
"""
信号生成模块
- 将AI概率转换为买入/观望/卖出信号
- 动态阈值：根据ATR波动率状态自适应调整
- 支持波段模式（V1.0）
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """交易信号数据类"""
    symbol:        str
    signal_type:   str          # strong_buy / buy / hold / sell / strong_sell
    probability:   float        # AI概率
    threshold:     float        # 当前动态阈值
    market_state:  str          # low / medium / high 波动状态
    confidence:    str          # high / medium / low
    suggested_pos: float        # 建议仓位比例
    reason:        str          # 信号触发原因说明
    signal_time:   str = field(default_factory=lambda: datetime.now().isoformat())
    degraded:      bool = False # 是否为降级信号

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def signal_cn(self) -> str:
        """中文信号名称"""
        return {
            "strong_buy":  "强买入",
            "buy":         "买入",
            "hold":        "观望",
            "sell":        "卖出",
            "strong_sell": "强卖出",
        }.get(self.signal_type, "观望")

    @property
    def signal_color(self) -> str:
        """信号颜色（中国股市习惯：红=买入，绿=卖出）"""
        return {
            "strong_buy":  "#B71C1C",
            "buy":         "#E53935",
            "hold":        "#9E9E9E",
            "sell":        "#00897B",
            "strong_sell": "#004D40",
        }.get(self.signal_type, "#9E9E9E")


class SignalGenerator:
    """
    信号生成器

    用法:
        gen = SignalGenerator(config)
        signal = gen.generate("600519", pred_result, df_factors)
    """

    SIGNAL_LABELS = {
        "strong_buy":  "强买入 🔴",
        "buy":         "买入 🟠",
        "hold":        "观望 ⚪",
        "sell":        "卖出 🟢",
        "strong_sell": "强卖出 🟩",
    }

    def __init__(self, config: dict):
        self.config = config
        signal_cfg = config.get("signal", {})
        self.dynamic_threshold = signal_cfg.get("dynamic_threshold", True)
        threshold_cfg = signal_cfg.get("thresholds", {})

        self.thresholds = {
            "low":    {"buy": threshold_cfg.get("low_volatility", {}).get("buy", 0.58),
                       "sell": threshold_cfg.get("low_volatility", {}).get("sell", 0.45)},
            "medium": {"buy": threshold_cfg.get("medium_volatility", {}).get("buy", 0.55),
                       "sell": threshold_cfg.get("medium_volatility", {}).get("sell", 0.45)},
            "high":   {"buy": threshold_cfg.get("high_volatility", {}).get("buy", 0.62),
                       "sell": threshold_cfg.get("high_volatility", {}).get("sell", 0.40)},
        }

        position_cfg = config.get("position", {})
        self.base_pos  = position_cfg.get("base_position_pct", 0.10)
        self.min_pos   = position_cfg.get("min_position_pct", 0.05)
        self.max_pos   = position_cfg.get("max_position_pct", 0.20)
        self.win_rate_base = position_cfg.get("win_rate_base", 0.55)

    # ── 主入口 ───────────────────────────────────────────────

    def generate(
        self,
        symbol: str,
        pred_result: dict,
        df_factors: pd.DataFrame,
        accuracy_stats: Optional[dict] = None,
    ) -> TradeSignal:
        """
        生成交易信号

        Args:
            symbol: 股票代码
            pred_result: 预测引擎输出 {'probability': 0.63, 'confidence': 'high', ...}
            df_factors: 因子数据（用于ATR状态判断）
            accuracy_stats: 历史准确率统计（用于仓位调整）

        Returns:
            TradeSignal
        """
        if not pred_result.get("success"):
            return self._hold_signal(symbol, 0.5, "medium", pred_result.get("message", "无预测数据"))

        prob = pred_result.get("probability", 0.5)
        confidence = pred_result.get("confidence", "medium")
        degraded = pred_result.get("degraded", False)

        # 降级信号：直接返回观望
        if degraded:
            return self._hold_signal(symbol, prob, "medium", "模型异常，降级为观望")

        # 判断市场波动状态
        market_state = self._get_market_state(df_factors)

        # 获取动态阈值
        buy_threshold, sell_threshold = self.get_dynamic_threshold(market_state)

        # 确定信号类型
        signal_type, reason = self._classify_signal(
            prob, buy_threshold, sell_threshold, df_factors
        )

        # 计算建议仓位
        suggested_pos = self._calc_position(signal_type, prob, market_state, accuracy_stats)

        return TradeSignal(
            symbol=symbol,
            signal_type=signal_type,
            probability=prob,
            threshold=buy_threshold,
            market_state=market_state,
            confidence=confidence,
            suggested_pos=suggested_pos,
            reason=reason,
            degraded=degraded,
        )

    def generate_batch(
        self,
        symbols: list,
        pred_results: dict,
        df_map: dict,
        accuracy_map: Optional[dict] = None,
    ) -> dict:
        """批量生成信号"""
        signals = {}
        for symbol in symbols:
            pred = pred_results.get(symbol, {"success": False})
            df = df_map.get(symbol, pd.DataFrame())
            acc = (accuracy_map or {}).get(symbol)
            signals[symbol] = self.generate(symbol, pred, df, acc)
        return signals

    # ── 阈值计算 ─────────────────────────────────────────────

    def get_dynamic_threshold(self, market_state: str) -> tuple:
        """
        获取动态买卖阈值

        Returns:
            (buy_threshold, sell_threshold)
        """
        if not self.dynamic_threshold:
            base = self.config.get("signal", {}).get("base_buy_threshold", 0.55)
            return base, 0.45

        t = self.thresholds.get(market_state, self.thresholds["medium"])
        return t["buy"], t["sell"]

    def _get_market_state(self, df_factors: pd.DataFrame) -> str:
        """根据ATR比率判断市场波动状态"""
        if df_factors.empty or "atr_ratio" not in df_factors.columns:
            return "medium"

        latest = df_factors["atr_ratio"].dropna()
        if latest.empty:
            return "medium"

        ratio = latest.iloc[-1]
        if ratio < 0.8:
            return "low"
        elif ratio > 1.2:
            return "high"
        return "medium"

    # ── 信号分类 ─────────────────────────────────────────────

    def _classify_signal(
        self,
        prob: float,
        buy_threshold: float,
        sell_threshold: float,
        df_factors: pd.DataFrame,
    ) -> tuple:
        """
        信号分类逻辑

        Returns:
            (signal_type, reason)
        """
        strong_buy_threshold = buy_threshold + 0.05
        strong_sell_threshold = 0.40

        # 检查强卖出：概率<0.40且连续3日下降
        if prob < strong_sell_threshold and self._is_consecutive_decline(df_factors, n=3):
            return "strong_sell", f"概率{prob:.1%} < 40% 且连续3日预测下行"

        # 信号分类
        if prob >= strong_buy_threshold:
            return "strong_buy", f"概率{prob:.1%} ≥ 强买入阈值{strong_buy_threshold:.1%}"
        elif prob >= buy_threshold:
            return "buy", f"概率{prob:.1%} ≥ 买入阈值{buy_threshold:.1%}"
        elif prob < sell_threshold:
            return "sell", f"概率{prob:.1%} < 卖出阈值{sell_threshold:.1%}"
        else:
            return "hold", f"概率{prob:.1%} 处于阈值区间，建议观望"

    def _is_consecutive_decline(
        self, df_factors: pd.DataFrame, n: int = 3
    ) -> bool:
        """检查收盘价是否连续N日下降"""
        if df_factors.empty or "close" not in df_factors.columns or len(df_factors) < n + 1:
            return False
        closes = df_factors["close"].tail(n + 1).values
        return all(closes[i] > closes[i + 1] for i in range(n))

    def _hold_signal(
        self, symbol: str, prob: float, market_state: str, reason: str
    ) -> TradeSignal:
        """创建默认观望信号"""
        return TradeSignal(
            symbol=symbol,
            signal_type="hold",
            probability=prob,
            threshold=self.thresholds.get(market_state, self.thresholds["medium"])["buy"],
            market_state=market_state,
            confidence="low",
            suggested_pos=0.0,
            reason=reason,
        )

    # ── 仓位计算 ─────────────────────────────────────────────

    def _calc_position(
        self,
        signal_type: str,
        prob: float,
        market_state: str,
        accuracy_stats: Optional[dict],
    ) -> float:
        """
        计算建议仓位（凯利公式变体）

        仓位 = 基础仓位 × 胜率调整系数 × 波动率调整系数
        """
        if signal_type not in ("strong_buy", "buy"):
            return 0.0

        # 胜率调整系数
        recent_win_rate = (accuracy_stats or {}).get("accuracy") or self.win_rate_base
        win_rate_adj = min(1.5, max(0.5, recent_win_rate / self.win_rate_base))

        # 波动率调整系数（高波动降仓）
        vol_adj = {"low": 1.2, "medium": 1.0, "high": 0.6}.get(market_state, 1.0)

        # 强信号额外加权
        signal_adj = 1.5 if signal_type == "strong_buy" else 1.0

        pos = self.base_pos * win_rate_adj * vol_adj * signal_adj
        pos = round(min(self.max_pos, max(self.min_pos, pos)), 4)
        return pos

    # ── 信号汇总 ─────────────────────────────────────────────

    @staticmethod
    def summarize_signals(signals: dict) -> pd.DataFrame:
        """
        将信号字典转为DataFrame（用于信号中心页面展示）

        Args:
            signals: {symbol: TradeSignal}

        Returns:
            DataFrame
        """
        rows = []
        for symbol, sig in signals.items():
            rows.append({
                "代码":    symbol,
                "信号":    sig.signal_cn,
                "概率":    f"{sig.probability:.1%}",
                "阈值":    f"{sig.threshold:.1%}",
                "波动状态": {"low": "低波动", "medium": "中等", "high": "高波动"}.get(sig.market_state, "-"),
                "建议仓位": f"{sig.suggested_pos:.0%}",
                "置信度":  {"high": "高", "medium": "中", "low": "低"}.get(sig.confidence, "-"),
                "原因":    sig.reason,
                "时间":    sig.signal_time[:16].replace("T", " "),
                "_type":   sig.signal_type,   # 内部字段，用于着色
            })
        return pd.DataFrame(rows)
