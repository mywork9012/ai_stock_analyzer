# core/data_manager.py
"""
数据获取与缓存管理模块
主数据源: baostock（免费、稳定、专为A股设计）
备用数据源: AkShare
存储: Parquet（行情/因子）+ SQLite（元数据）
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent.parent / "data"
PARQUET_DIR = DATA_DIR / "parquet"
DB_PATH     = DATA_DIR / "database" / "metadata.db"


# ── 工具函数 ─────────────────────────────────────────────────

def _bs_code(symbol: str) -> str:
    """'600519' → 'sh.600519'，'000858' → 'sz.000858'"""
    if symbol.startswith(("6", "9")):
        return f"sh.{symbol}"
    elif symbol.startswith(("0", "2", "3")):
        return f"sz.{symbol}"
    elif symbol.startswith("8"):
        return f"bj.{symbol}"
    return f"sh.{symbol}"


def _fmt_date(d: str) -> str:
    """'20220101' → '2022-01-01'"""
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


# ── baostock 会话单例 ────────────────────────────────────────

class _BsSession:
    _logged_in = False

    @classmethod
    def login(cls):
        if cls._logged_in:
            return
        try:
            import baostock as bs
            ret = bs.login()
            if ret.error_code == "0":
                cls._logged_in = True
                logger.info("baostock 登录成功")
            else:
                logger.error(f"baostock 登录失败: {ret.error_msg}")
                cls._logged_in = False
        except ImportError:
            logger.error("baostock 未安装，请执行: pip install baostock")
        except Exception as e:
            logger.error(f"baostock 登录异常: {e}")
            cls._logged_in = False

    @classmethod
    def relogin(cls):
        """强制重新登录（会话断开时调用）"""
        cls._logged_in = False
        cls.login()

    @classmethod
    def logout(cls):
        if cls._logged_in:
            try:
                import baostock as bs
                bs.logout()
            except Exception:
                pass
            cls._logged_in = False


# ── 主类 ─────────────────────────────────────────────────────

class StockDataFetcher:
    """
    股票数据获取器

    用法:
        fetcher = StockDataFetcher(config)
        df = fetcher.fetch_daily("600519")
        df_fund = fetcher.fetch_fundamental("600519")
    """

    def __init__(self, config: dict):
        self.config        = config
        data_cfg           = config.get("data", {})
        self.cache_days    = data_cfg.get("cache_days", 7)
        self.history_years = data_cfg.get("history_years", 3)
        adjust             = data_cfg.get("adjust_type", "hfq")
        # baostock: 1=后复权, 2=前复权, 3=不复权
        self.adjustflag    = "1" if adjust == "hfq" else "2"
        self._init_db()
        _BsSession.login()

    # ── 初始化 SQLite ────────────────────────────────────────

    def _init_db(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_registry (
                symbol    TEXT NOT NULL,
                data_type TEXT NOT NULL,
                last_date TEXT,
                fetch_time TEXT,
                row_count  INTEGER,
                quality    TEXT DEFAULT 'ok',
                PRIMARY KEY (symbol, data_type)
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blacklist (
                symbol      TEXT PRIMARY KEY,
                reason      TEXT,
                added_date  TEXT,
                expire_date TEXT
            )""")
        conn.commit()
        conn.close()

    # ── 日线数据 ─────────────────────────────────────────────

    def fetch_daily(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        获取 A 股日线数据（后复权）

        Returns:
            DataFrame 列: date / open / high / low / close /
                          volume / amount / pct_change / turnover
        """
        today = datetime.now()
        end_date   = end_date   or today.strftime("%Y%m%d")
        start_date = start_date or (
            today - timedelta(days=self.history_years * 365 + 60)
        ).strftime("%Y%m%d")

        # 读缓存
        if not force_refresh:
            cached = self._load_parquet(symbol, "daily")
            if cached is not None and self._cache_valid(symbol, "daily"):
                logger.info(f"[{symbol}] 命中缓存 {len(cached)} 行")
                return self._date_filter(cached, start_date, end_date)

        # baostock 优先
        df = self._bs_daily(symbol, start_date, end_date)

        # 失败降级 AkShare
        if df is None or df.empty:
            logger.warning(f"[{symbol}] baostock 失败，尝试 AkShare")
            df = self._ak_daily(symbol, start_date, end_date)

        if df is None or df.empty:
            logger.error(f"[{symbol}] 所有数据源均失败")
            return pd.DataFrame()

        df = self._clean(df)
        self._save_parquet(df, symbol, "daily")
        self._reg_update(symbol, "daily", df)
        return df

    def _bs_daily(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """baostock 日线（自动重连，最多重试2次）"""
        for attempt in range(2):
            result = self._bs_daily_once(symbol, start, end)
            if result is not None:
                return result
            if attempt == 0:
                logger.warning(f"[{symbol}] baostock 第1次失败，重新登录后重试")
                _BsSession.relogin()
        return None

    def _bs_daily_once(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """baostock 日线单次尝试"""
        try:
            import baostock as bs
            _BsSession.login()

            rs = bs.query_history_k_data_plus(
                _bs_code(symbol),
                "date,open,high,low,close,volume,amount,turn,pctChg,tradestatus",
                start_date=_fmt_date(start),
                end_date=_fmt_date(end),
                frequency="d",
                adjustflag=self.adjustflag,
            )
            if rs.error_code != "0":
                logger.error(f"[{symbol}] baostock 错误: {rs.error_msg}")
                return None

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                return None

            df = pd.DataFrame(rows, columns=rs.fields)

            # 只保留正常交易日
            if "tradestatus" in df.columns:
                df = df[df["tradestatus"] == "1"].copy()

            df = df.rename(columns={"turn": "turnover", "pctChg": "pct_change"})

            num_cols = ["open", "high", "low", "close", "volume", "amount",
                        "turnover", "pct_change"]
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            logger.info(f"[{symbol}] baostock 取得 {len(df)} 行")
            return df

        except Exception as e:
            logger.error(f"[{symbol}] baostock 异常: {e}")
            return None

    def _ak_daily(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """AkShare 备用日线"""
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust="hfq",
            )
            if df is None or df.empty:
                return None
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "涨跌幅": "pct_change", "换手率": "turnover",
            })
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.error(f"[{symbol}] AkShare 异常: {e}")
            return None

    # ── 基本面数据 ───────────────────────────────────────────

    def fetch_daily_raw(self, symbol: str) -> pd.DataFrame:
        """
        获取不复权日线数据，专用于价格展示（与行情软件数值一致）。
        模型训练继续使用 fetch_daily()（后复权，保证技术指标连续性）。
        """
        try:
            import baostock as bs
            _BsSession.login()
            end   = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(
                _bs_code(symbol),
                "date,open,high,low,close,volume,amount,turn,pctChg,tradestatus",
                start_date=start, end_date=end,
                frequency="d", adjustflag="3",
            )
            if rs.error_code != "0":
                return pd.DataFrame()
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=rs.fields)
            if "tradestatus" in df.columns:
                df = df[df["tradestatus"] == "1"].copy()
            df = df.rename(columns={"turn": "turnover", "pctChg": "pct_change"})
            for c in ["open", "high", "low", "close", "volume", "amount",
                      "turnover", "pct_change"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["close"].notna() & (df["close"] > 0)]
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"[{symbol}] 不复权数据获取失败，将降级使用复权数据: {e}")
            return pd.DataFrame()

    def fetch_fundamental(
        self, symbol: str, force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取估值 + 盈利能力指标
        来源: baostock query_valuation_data + query_profit_data
        """
        if not force_refresh:
            cached = self._load_parquet(symbol, "fundamental")
            if cached is not None and self._cache_valid(
                symbol, "fundamental", max_age_days=30
            ):
                return cached

        df = self._bs_fundamental(symbol)
        if df is not None and not df.empty:
            self._save_parquet(df, symbol, "fundamental")
            self._reg_update(symbol, "fundamental", df)
            return df
        return pd.DataFrame()

    def _bs_fundamental(self, symbol: str) -> Optional[pd.DataFrame]:
        """baostock 获取季度财务数据"""
        try:
            import baostock as bs
            _BsSession.login()

            bs_code  = _bs_code(symbol)
            cur_year = datetime.now().year
            all_rows = []

            for year in range(cur_year - 3, cur_year + 1):
                for q in range(1, 5):
                    # 估值
                    rs_val = bs.query_valuation_data(
                        code=bs_code, year=str(year), quarter=str(q)
                    )
                    val_rows = []
                    while rs_val.next():
                        val_rows.append(rs_val.get_row_data())

                    # 盈利
                    rs_pft = bs.query_profit_data(
                        code=bs_code, year=str(year), quarter=str(q)
                    )
                    pft_rows = []
                    while rs_pft.next():
                        pft_rows.append(rs_pft.get_row_data())

                    if val_rows and pft_rows:
                        df_v = pd.DataFrame(val_rows, columns=rs_val.fields)
                        df_p = pd.DataFrame(pft_rows, columns=rs_pft.fields)
                        merged = pd.merge(
                            df_v, df_p,
                            on=["code", "pubDate", "statDate"], how="outer"
                        )
                        all_rows.append(merged)
                    elif val_rows:
                        all_rows.append(pd.DataFrame(val_rows, columns=rs_val.fields))

            if not all_rows:
                return None

            df = pd.concat(all_rows, ignore_index=True)
            df = df.rename(columns={
                "pubDate":          "date",
                "peTTM":            "pe_ttm",
                "pbMRQ":            "pb",
                "psTTM":            "ps_ttm",
                "roeAvg":           "roe_ttm",
                "netProfitMargin":  "net_margin",
                "grossProfitMargin":"gross_margin",
            })

            keep = ["date"] + [
                c for c in ["pe_ttm","pb","ps_ttm","roe_ttm",
                             "net_margin","gross_margin"]
                if c in df.columns
            ]
            df = df[keep].drop_duplicates(subset=["date"])
            for c in keep[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            logger.info(f"[{symbol}] 基本面数据 {len(df)} 行")
            return df

        except Exception as e:
            logger.error(f"[{symbol}] baostock 基本面异常: {e}")
            return None

    # ── 指数数据 ─────────────────────────────────────────────

    def get_index_data(self, index_code: str = "000300") -> pd.DataFrame:
        """获取指数日线（用于回测基准）"""
        try:
            import baostock as bs
            _BsSession.login()

            # 沪市指数用 sh，其余用 sz
            prefix = "sh" if index_code.startswith(("0", "9")) else "sz"
            bs_idx = f"{prefix}.{index_code}"

            today  = datetime.now()
            sd     = (today - timedelta(days=self.history_years * 365 + 60)).strftime("%Y-%m-%d")
            ed     = today.strftime("%Y-%m-%d")

            rs = bs.query_history_k_data_plus(
                bs_idx, "date,close",
                start_date=sd, end_date=ed,
                frequency="d", adjustflag="3",
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=rs.fields)
            df["date"]  = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            return df.dropna().sort_values("date").reset_index(drop=True)

        except Exception as e:
            logger.error(f"指数 {index_code} 失败: {e}")
            return pd.DataFrame()

    # ── 股票列表（搜索框）───────────────────────────────────

    def fetch_stock_list(self) -> pd.DataFrame:
        cache = PARQUET_DIR / "stock_list.parquet"
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)

        if cache.exists():
            age = (datetime.now() - datetime.fromtimestamp(
                cache.stat().st_mtime
            )).days
            if age < 7:
                return pd.read_parquet(cache)

        # baostock
        try:
            import baostock as bs
            _BsSession.login()
            rs = bs.query_stock_basic()
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=rs.fields)
                df["symbol"] = df["code"].str.split(".").str[1]
                df = df.rename(columns={"code_name": "name"})[["symbol","name"]]
                df = df.dropna().drop_duplicates("symbol")
                df.to_parquet(cache, index=False)
                return df
        except Exception as e:
            logger.warning(f"baostock 股票列表失败: {e}")

        # AkShare 备用
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            df.columns = ["symbol", "name"]
            df.to_parquet(cache, index=False)
            return df
        except Exception as e:
            logger.error(f"股票列表全部失败: {e}")
            return pd.DataFrame(columns=["symbol", "name"])

    # ── 内部工具 ─────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        for c in ["open","high","low","close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["close"].notna() & (df["close"] > 0)]
        df[["open","high","low","close"]] = df[["open","high","low","close"]].ffill()
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        return df.reset_index(drop=True)

    def _parquet_path(self, sym: str, dtype: str) -> Path:
        return PARQUET_DIR / f"{sym}_{dtype}.parquet"

    def _load_parquet(self, sym: str, dtype: str) -> Optional[pd.DataFrame]:
        p = self._parquet_path(sym, dtype)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
        return None

    def _save_parquet(self, df: pd.DataFrame, sym: str, dtype: str):
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(self._parquet_path(sym, dtype), index=False)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _cache_valid(
        self, sym: str, dtype: str, max_age_days: Optional[int] = None
    ) -> bool:
        max_age = max_age_days or self.cache_days
        try:
            conn = sqlite3.connect(DB_PATH)
            row  = conn.execute(
                "SELECT fetch_time, last_date FROM data_registry "
                "WHERE symbol=? AND data_type=?", (sym, dtype)
            ).fetchone()
            conn.close()
        except Exception:
            return False

        if not row or not row[0]:
            return False

        age = (datetime.now() - datetime.fromisoformat(row[0])).days
        if age < max_age:
            return True

        if dtype == "daily" and row[1]:
            today = datetime.now().date()
            if today.weekday() >= 5:
                last = datetime.strptime(row[1], "%Y-%m-%d").date()
                if last >= today - timedelta(days=2):
                    return True
        return False

    def _reg_update(self, sym: str, dtype: str, df: pd.DataFrame):
        last = str(df["date"].max().date()) if not df.empty and "date" in df.columns else ""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT OR REPLACE INTO data_registry "
                "(symbol,data_type,last_date,fetch_time,row_count,quality) "
                "VALUES(?,?,?,?,?,?)",
                (sym, dtype, last, datetime.now().isoformat(), len(df), "ok"),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"注册表更新失败: {e}")

    def _date_filter(self, df: pd.DataFrame, sd: str, ed: str) -> pd.DataFrame:
        if df.empty or "date" not in df.columns:
            return df
        return df[
            (df["date"] >= pd.to_datetime(sd)) &
            (df["date"] <= pd.to_datetime(ed))
        ].reset_index(drop=True)

    # ── 元数据查询 ───────────────────────────────────────────

    def get_registry_info(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM data_registry ORDER BY symbol", conn)
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()

    def is_blacklisted(self, symbol: str) -> bool:
        try:
            conn = sqlite3.connect(DB_PATH)
            row  = conn.execute(
                "SELECT expire_date FROM blacklist WHERE symbol=?", (symbol,)
            ).fetchone()
            conn.close()
            if not row:
                return False
            return datetime.now().date() <= datetime.strptime(
                row[0], "%Y-%m-%d"
            ).date()
        except Exception:
            return False

    def add_to_blacklist(self, symbol: str, reason: str, days: int = 30):
        try:
            expire = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT OR REPLACE INTO blacklist "
                "(symbol,reason,added_date,expire_date) VALUES(?,?,?,?)",
                (symbol, reason, datetime.now().strftime("%Y-%m-%d"), expire),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"黑名单写入失败: {e}")
