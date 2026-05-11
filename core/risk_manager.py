# core/risk_manager.py
"""
风控管理模块（三层风控体系）
第一层: 个股级 —— 固定止损/移动止损/时间止损/仓位上限/涨跌停过滤
第二层: 策略级 —— 日亏损熔断/连续亏损熔断/最大回撤防御模式
第三层: 系统级 —— 黑名单/ST过滤/新股过滤
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "database" / "metadata.db"


@dataclass
class RiskCheckResult:
    """风控检查结果"""
    passed:       bool           # 是否通过
    level:        str            # 触发层级: stock / strategy / system / none
    rule:         str            # 触发规则名称
    action:       str            # 建议动作: block / force_sell / reduce / warn
    message:      str            # 说明


class RiskController:
    """
    风控控制器

    用法:
        risk = RiskController(config)

        # 买入前检查
        result = risk.check_buy("600519", amount, portfolio_value)
        if not result.passed:
            print(result.message)

        # 持仓风控检查（每日执行）
        actions = risk.check_holding("600519", entry_price, current_price, holding_days, peak_price)
    """

    def __init__(self, config: dict):
        self.config = config
        risk_cfg = config.get("risk", {})

        # 个股级参数
        self.stop_loss_pct         = risk_cfg.get("stop_loss_pct", 0.08)
        self.trailing_stop_pct     = risk_cfg.get("trailing_stop_pct", 0.05)
        self.time_stop_days        = risk_cfg.get("time_stop_days", 20)
        self.time_stop_min_return  = risk_cfg.get("time_stop_min_return", 0.02)
        self.max_position_pct      = risk_cfg.get("max_position_pct", 0.20)

        # 策略级参数
        self.daily_loss_limit      = risk_cfg.get("daily_loss_limit", 0.03)
        self.consec_loss_count     = risk_cfg.get("consecutive_loss_count", 5)
        self.max_drawdown_def      = risk_cfg.get("max_drawdown_defensive", 0.15)

        # 系统级参数
        self.blacklist_trigger     = risk_cfg.get("blacklist_trigger_count", 3)
        self.blacklist_days        = risk_cfg.get("blacklist_days", 30)
        self.min_listing_days      = risk_cfg.get("min_listing_days", 60)
        self.exclude_st            = risk_cfg.get("exclude_st", True)

        # 运行时状态
        self._daily_loss: float = 0.0
        self._strategy_pnl: float = 0.0
        self._peak_value: float = 1.0
        self._circuit_breaker: bool = False
        self._circuit_breaker_until: Optional[datetime] = None
        self._defensive_mode: bool = False
        self._consecutive_losses: int = 0
        self._stop_counts: dict = {}   # symbol → int

        self._init_state_db()

    # ── 第一层：个股级风控 ────────────────────────────────────

    def check_buy(
        self,
        symbol: str,
        amount: float,
        portfolio_value: float,
        stock_name: str = "",
    ) -> RiskCheckResult:
        """
        买入前风控检查

        Returns:
            RiskCheckResult（passed=False时禁止买入）
        """
        # 系统级检查
        sys_check = self._check_system_level(symbol, stock_name)
        if not sys_check.passed:
            return sys_check

        # 策略级检查
        strat_check = self._check_strategy_level()
        if not strat_check.passed:
            return strat_check

        # 仓位上限检查
        if portfolio_value > 0:
            pos_pct = amount / portfolio_value
            if pos_pct > self.max_position_pct:
                return RiskCheckResult(
                    passed=False,
                    level="stock",
                    rule="仓位上限",
                    action="block",
                    message=(
                        f"[{symbol}] 买入金额{amount:,.0f}占总仓位{pos_pct:.1%}，"
                        f"超过上限{self.max_position_pct:.0%}，已拒绝"
                    ),
                )

        return RiskCheckResult(
            passed=True, level="none", rule="", action="allow", message="通过全部风控检查"
        )

    def check_holding(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        holding_days: int,
        peak_price: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        持仓风控检查（每日调用）

        Returns:
            RiskCheckResult，action 为:
            - 'force_sell': 次日开盘强制平仓
            - 'reduce': 减仓50%
            - 'warn': 给出警告，不强制
            - 'hold': 继续持有
        """
        if entry_price <= 0:
            return RiskCheckResult(
                passed=True, level="none", rule="", action="hold", message=""
            )

        pnl_pct = (current_price - entry_price) / entry_price

        # 涨跌停检测（跌停不强制卖出，等次日）
        if pnl_pct <= -0.099:
            return RiskCheckResult(
                passed=True,
                level="stock",
                rule="跌停过滤",
                action="hold",
                message=f"[{symbol}] 跌停板，不在跌停时强制卖出，等待次日",
            )

        # 固定止损
        if pnl_pct <= -self.stop_loss_pct:
            self._record_stop_loss(symbol)
            return RiskCheckResult(
                passed=False,
                level="stock",
                rule="固定止损",
                action="force_sell",
                message=(
                    f"[{symbol}] 亏损{pnl_pct:.1%}，触发固定止损({self.stop_loss_pct:.0%})，"
                    "次日开盘强制卖出"
                ),
            )

        # 移动止损
        if peak_price and peak_price > 0:
            drawdown = (current_price - peak_price) / peak_price
            if drawdown <= -self.trailing_stop_pct:
                return RiskCheckResult(
                    passed=False,
                    level="stock",
                    rule="移动止损",
                    action="reduce",
                    message=(
                        f"[{symbol}] 从最高点{peak_price:.2f}回撤{drawdown:.1%}，"
                        f"触发移动止损，次日减仓50%"
                    ),
                )

        # 时间止损
        if holding_days >= self.time_stop_days and pnl_pct < self.time_stop_min_return:
            return RiskCheckResult(
                passed=False,
                level="stock",
                rule="时间止损",
                action="force_sell",
                message=(
                    f"[{symbol}] 持仓{holding_days}日，收益{pnl_pct:.1%} < "
                    f"{self.time_stop_min_return:.0%}，触发时间止损"
                ),
            )

        return RiskCheckResult(
            passed=True, level="none", rule="", action="hold",
            message=f"[{symbol}] 持仓正常，当前盈亏{pnl_pct:.1%}"
        )

    def check_trade_price(self, symbol: str, price_change_pct: float) -> RiskCheckResult:
        """
        涨跌停过滤（买入时：涨停不追；卖出时：跌停顺延）
        """
        if price_change_pct >= 0.099:
            return RiskCheckResult(
                passed=False,
                level="stock",
                rule="涨停过滤",
                action="block",
                message=f"[{symbol}] 涨停板，不追涨，信号顺延至下一日",
            )
        return RiskCheckResult(
            passed=True, level="none", rule="", action="allow", message=""
        )

    # ── 第二层：策略级风控 ────────────────────────────────────

    def _check_strategy_level(self) -> RiskCheckResult:
        """策略级风控检查"""
        # 熔断器检查（连续亏损）
        if self._circuit_breaker:
            if self._circuit_breaker_until and datetime.now() < self._circuit_breaker_until:
                remaining = self._circuit_breaker_until - datetime.now()
                return RiskCheckResult(
                    passed=False,
                    level="strategy",
                    rule="连续亏损熔断",
                    action="block",
                    message=f"连续亏损熔断触发，暂停交易，剩余{remaining.seconds//3600}小时",
                )
            else:
                self._circuit_breaker = False

        # 日亏损熔断
        if self._daily_loss >= self.daily_loss_limit:
            return RiskCheckResult(
                passed=False,
                level="strategy",
                rule="日亏损熔断",
                action="block",
                message=f"当日亏损{self._daily_loss:.1%}已触发熔断({self.daily_loss_limit:.0%})，停止新开仓",
            )

        # 防御模式（允许买入，但发出警告）
        if self._defensive_mode:
            return RiskCheckResult(
                passed=False,
                level="strategy",
                rule="最大回撤防御",
                action="block",
                message=f"策略回撤超过{self.max_drawdown_def:.0%}，已进入防御模式，仅允许卖出操作",
            )

        return RiskCheckResult(
            passed=True, level="none", rule="", action="allow", message=""
        )

    def update_strategy_pnl(self, trade_pnl: float, portfolio_value: float, peak_value: float):
        """
        更新策略盈亏状态（每笔交易后调用）

        Args:
            trade_pnl: 本次交易盈亏金额（负数=亏损）
            portfolio_value: 当前组合市值
            peak_value: 历史最高市值
        """
        # 更新日内亏损
        if trade_pnl < 0:
            self._daily_loss += abs(trade_pnl) / max(portfolio_value, 1)
            self._consecutive_losses += 1

            # 连续亏损熔断
            if self._consecutive_losses >= self.consec_loss_count:
                self._circuit_breaker = True
                self._circuit_breaker_until = datetime.now() + timedelta(hours=24)
                logger.warning(
                    f"连续亏损{self._consecutive_losses}笔，触发熔断，暂停24小时"
                )
        else:
            self._consecutive_losses = 0

        # 最大回撤防御模式
        if peak_value > 0:
            drawdown = (portfolio_value - peak_value) / peak_value
            if drawdown <= -self.max_drawdown_def:
                self._defensive_mode = True
                logger.warning(f"策略回撤{drawdown:.1%}，进入防御模式")
            elif drawdown > -self.max_drawdown_def * 0.5:
                self._defensive_mode = False

    def reset_daily_stats(self):
        """每日开盘前重置日内统计"""
        self._daily_loss = 0.0
        logger.info("日内风控计数器已重置")

    # ── 第三层：系统级风控 ────────────────────────────────────

    def _check_system_level(self, symbol: str, stock_name: str = "") -> RiskCheckResult:
        """系统级风控检查"""
        # ST股票过滤
        if self.exclude_st and ("ST" in stock_name.upper() or "*ST" in stock_name):
            return RiskCheckResult(
                passed=False,
                level="system",
                rule="ST过滤",
                action="block",
                message=f"[{symbol}] 为ST/*ST股票，不生成买入信号",
            )

        # 黑名单检查
        if self._is_blacklisted(symbol):
            return RiskCheckResult(
                passed=False,
                level="system",
                rule="黑名单",
                action="block",
                message=f"[{symbol}] 在黑名单中，暂停交易",
            )

        return RiskCheckResult(
            passed=True, level="none", rule="", action="allow", message=""
        )

    def _record_stop_loss(self, symbol: str):
        """记录止损次数，超过阈值加入黑名单"""
        self._stop_counts[symbol] = self._stop_counts.get(symbol, 0) + 1
        count = self._stop_counts[symbol]

        # 保存到数据库
        self._save_stop_count(symbol, count)

        if count >= self.blacklist_trigger:
            self._add_to_blacklist(
                symbol,
                f"连续触发{count}次止损",
                days=self.blacklist_days,
            )

    def _is_blacklisted(self, symbol: str) -> bool:
        """检查黑名单"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT expire_date FROM blacklist WHERE symbol=?", (symbol,)
            )
            row = cursor.fetchone()
            conn.close()
            if not row:
                return False
            expire = datetime.strptime(row[0], "%Y-%m-%d").date()
            return datetime.now().date() <= expire
        except Exception:
            return False

    def _add_to_blacklist(self, symbol: str, reason: str, days: int = 30):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            expire = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
            cursor.execute(
                """INSERT OR REPLACE INTO blacklist (symbol, reason, added_date, expire_date)
                   VALUES (?, ?, ?, ?)""",
                (symbol, reason, datetime.now().strftime("%Y-%m-%d"), expire),
            )
            conn.commit()
            conn.close()
            logger.warning(f"[{symbol}] 加入黑名单: {reason}，到期: {expire}")
        except Exception as e:
            logger.error(f"黑名单写入失败: {e}")

    def _save_stop_count(self, symbol: str, count: int):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS stop_loss_counts
                   (symbol TEXT PRIMARY KEY, count INTEGER, last_time TEXT)"""
            )
            cursor.execute(
                "INSERT OR REPLACE INTO stop_loss_counts VALUES (?, ?, ?)",
                (symbol, count, datetime.now().isoformat()),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _init_state_db(self):
        """初始化风控状态数据库表"""
        try:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS stop_loss_counts
                   (symbol TEXT PRIMARY KEY, count INTEGER, last_time TEXT)"""
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_risk_summary(self) -> dict:
        """获取当前风控状态摘要（用于UI展示）"""
        return {
            "daily_loss_pct": round(self._daily_loss, 4),
            "daily_loss_limit": self.daily_loss_limit,
            "circuit_breaker": self._circuit_breaker,
            "circuit_breaker_until": (
                self._circuit_breaker_until.isoformat()
                if self._circuit_breaker_until else None
            ),
            "defensive_mode": self._defensive_mode,
            "consecutive_losses": self._consecutive_losses,
        }

    def get_blacklist(self) -> pd.DataFrame:
        """获取黑名单列表"""
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM blacklist ORDER BY added_date DESC", conn)
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()
