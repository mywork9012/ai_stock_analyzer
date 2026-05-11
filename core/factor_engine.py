# core/factor_engine.py
"""
因子计算模块
- 优先使用 pandas_ta 计算技术指标
- pandas_ta 不可用时自动降级为纯 numpy/pandas 实现（保证始终有特征输出）
- 基本面因子: PE/PB/ROE 等季频数据前向填充至日频
- 因子预处理: Winsorize 缩尾（1%/99%）
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# baostock 返回的非特征列，在 _get_feature_cols 中一并排除
_BASE_EXCLUDE = {
    "date", "code",                              # 元数据
    "open", "high", "low", "close",              # 原始价格（模型不直接用）
    "volume", "amount", "amplitude",
    "pct_change", "price_change", "turnover",
    "tradestatus", "adjustflag",
}


class FactorCalculator:
    """
    因子计算器

    用法:
        calc = FactorCalculator(config)
        df_factors = calc.calc_all(df_daily, df_fundamental)
    """

    def __init__(self, config: dict):
        self.config = config
        self._has_ta = self._check_pandas_ta()

    @staticmethod
    def _check_pandas_ta() -> bool:
        try:
            import pandas_ta  # noqa
            return True
        except ImportError:
            logger.warning(
                "pandas_ta 未安装，将使用内置因子计算（功能完整）。"
                "可执行 pip install pandas_ta 获得更多指标。"
            )
            return False

    # ── 主入口 ────────────────────────────────────────────────

    def calc_all(
        self,
        df_daily: pd.DataFrame,
        df_fundamental: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if df_daily is None or df_daily.empty:
            return pd.DataFrame()

        df = df_daily.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # 1. 技术因子
        df = self.calc_technical(df)

        # 2. 基本面因子（前向填充到日频）
        if df_fundamental is not None and not df_fundamental.empty:
            df = self._merge_fundamental(df, df_fundamental)

        # 3. Winsorize 缩尾（不做 Z-Score，由 trainer 的 StandardScaler 统一处理）
        df = self._winsorize(df)

        feat_cols = self.get_feature_columns(df)
        logger.info(f"因子计算完成，有效特征列 {len(feat_cols)} 个，共 {len(df)} 行")
        return df

    # ── 技术因子 ──────────────────────────────────────────────

    def calc_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术面因子，pandas_ta 优先，失败时自动用内置实现"""
        if self._has_ta:
            try:
                return self._calc_with_pandas_ta(df)
            except Exception as e:
                logger.warning(f"pandas_ta 计算异常（{e}），切换至内置实现")
        return self._calc_builtin(df)

    # ── pandas_ta 实现 ────────────────────────────────────────

    def _calc_with_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas_ta as ta
        df = df.copy()

        # 均线
        for p in [5, 10, 20, 60]:
            df[f"sma{p}"] = ta.sma(df["close"], length=p)
        for p in [12, 26]:
            df[f"ema{p}"] = ta.ema(df["close"], length=p)

        # 价格相对均线位置
        for p in [5, 20]:
            col = f"sma{p}"
            if col in df.columns:
                df[f"price_to_sma{p}"] = df["close"] / df[col].replace(0, np.nan) - 1

        # RSI
        df["rsi14"] = ta.rsi(df["close"], length=14)

        # MACD
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            cols = macd.columns.tolist()
            if len(cols) >= 3:
                df["macd_dif"] = macd.iloc[:, 0]
                df["macd_bar"] = macd.iloc[:, 1]
                df["macd_dea"] = macd.iloc[:, 2]

        # 布林带
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None and not bb.empty:
            for key, col in [("BBL", "bb_lower"), ("BBM", "bb_mid"),
                              ("BBU", "bb_upper"), ("BBB", "bb_width"),
                              ("BBP", "bb_pct")]:
                filtered = bb.filter(like=key)
                if not filtered.empty:
                    df[col] = filtered.iloc[:, 0]

        # ATR
        df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # OBV
        df["obv"] = ta.obv(df["close"], df["volume"])

        # KDJ（用 Stochastic 替代）
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=9, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            k_col = stoch.filter(like="STOCHk")
            d_col = stoch.filter(like="STOCHd")
            if not k_col.empty and not d_col.empty:
                df["kdj_k"] = k_col.iloc[:, 0]
                df["kdj_d"] = d_col.iloc[:, 0]
                df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

        # ATR 比率（用于动态阈值）
        df["atr14_sma20"] = df["atr14"].rolling(20).mean()
        df["atr_ratio"]   = df["atr14"] / df["atr14_sma20"].replace(0, np.nan)

        # 量比
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(5).mean().replace(0, np.nan)
        df["vol_sma5"]     = df["volume"].rolling(5).mean()

        return df

    # ── 内置纯 numpy/pandas 实现（零依赖备用）─────────────────

    def _calc_builtin(self, df: pd.DataFrame) -> pd.DataFrame:
        """完全不依赖 pandas_ta 的技术因子计算"""
        df = df.copy()
        c = df["close"]
        h = df["high"]
        lo = df["low"]
        v = df["volume"]

        # ── 均线 ──
        for p in [5, 10, 20, 60]:
            df[f"sma{p}"] = c.rolling(p).mean()
        df["ema12"] = c.ewm(span=12, adjust=False).mean()
        df["ema26"] = c.ewm(span=26, adjust=False).mean()

        # 价格相对均线
        for p in [5, 20]:
            col = f"sma{p}"
            df[f"price_to_sma{p}"] = c / df[col].replace(0, np.nan) - 1

        # ── RSI(14) ──
        df["rsi14"] = self._rsi(c, 14)

        # ── MACD(12,26,9) ──
        dif = df["ema12"] - df["ema26"]
        dea = dif.ewm(span=9, adjust=False).mean()
        df["macd_dif"] = dif
        df["macd_dea"] = dea
        df["macd_bar"] = 2 * (dif - dea)

        # ── 布林带(20,2) ──
        sma20  = c.rolling(20).mean()
        std20  = c.rolling(20).std(ddof=0)
        df["bb_mid"]   = sma20
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        band_width     = df["bb_upper"] - df["bb_lower"]
        df["bb_width"] = band_width / sma20.replace(0, np.nan) * 100
        df["bb_pct"]   = (c - df["bb_lower"]) / band_width.replace(0, np.nan)

        # ── ATR(14) ──
        df["atr14"] = self._atr(h, lo, c, 14)
        df["atr14_sma20"] = df["atr14"].rolling(20).mean()
        df["atr_ratio"]   = df["atr14"] / df["atr14_sma20"].replace(0, np.nan)

        # ── OBV ──
        direction     = np.sign(c.diff()).fillna(0)
        df["obv"]     = (v * direction).cumsum()

        # ── KDJ(9,3,3) ──
        low9  = lo.rolling(9).min()
        high9 = h.rolling(9).max()
        rsv   = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
        df["kdj_k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(alpha=1/3, adjust=False).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

        # ── 量比 ──
        df["volume_ratio"] = v / v.rolling(5).mean().replace(0, np.nan)
        df["vol_sma5"]     = v.rolling(5).mean()

        return df

    # ── 内置指标计算 ──────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta  = close.diff()
        gain   = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss   = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs     = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series,
             close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    # ── 基本面因子合并 ────────────────────────────────────────

    def _merge_fundamental(
        self, df_daily: pd.DataFrame, df_fund: pd.DataFrame
    ) -> pd.DataFrame:
        """将季频基本面数据前向填充到日线（merge_asof 严格无未来数据）"""
        if "date" not in df_fund.columns or df_fund.empty:
            return df_daily
        fund_cols = [c for c in df_fund.columns if c != "date"]
        try:
            merged = pd.merge_asof(
                df_daily.sort_values("date"),
                df_fund[["date"] + fund_cols].sort_values("date"),
                on="date",
                direction="backward",
            )
            return merged
        except Exception as e:
            logger.warning(f"基本面数据合并失败: {e}")
            return df_daily

    # ── Winsorize 缩尾 ────────────────────────────────────────

    def _winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """对因子列进行 1%/99% 缩尾，抑制极端值影响"""
        factor_cols = self.get_feature_columns(df)
        for col in factor_cols:
            q_lo = df[col].quantile(0.01)
            q_hi = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q_lo, upper=q_hi)
        return df

    # ── 特征列选取 ────────────────────────────────────────────

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        返回可用于模型训练/推理的因子列名。
        排除原始 OHLCV、元数据列、_raw 备份列，
        要求数值型且至少 50% 非空。
        """
        exclude = _BASE_EXCLUDE | {
            c for c in df.columns if c.endswith("_raw")
        }
        return [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().mean() > 0.5
        ]

    def get_market_state(self, df: pd.DataFrame) -> str:
        """根据 ATR 比率判断当前市场波动状态: low / medium / high"""
        if "atr_ratio" not in df.columns:
            return "medium"
        ratio = df["atr_ratio"].dropna()
        if ratio.empty:
            return "medium"
        v = ratio.iloc[-1]
        if pd.isna(v):
            return "medium"
        if v < 0.8:
            return "low"
        if v > 1.2:
            return "high"
        return "medium"
