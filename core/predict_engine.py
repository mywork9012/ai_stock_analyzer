# core/predict_engine.py
"""
AI预测引擎
- 加载已训练的LSTM模型，输出次日上涨概率（≤1秒）
- 历史预测记录持久化 + 自动回填实际涨跌结果
- 首次使用时，自动对历史数据做回测预测，快速积累准确率样本
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from models.lstm_model import LSTMModel
from models.trainer import ModelTrainer

logger = logging.getLogger(__name__)

PRED_DIR = Path(__file__).parent.parent / "data" / "parquet"


class LSTMPredictor:
    """
    LSTM预测引擎

    关键设计：
    - predict()       同步推理，≤1秒
    - backfill_history() 首次调用时，对最近N日历史数据批量预测，
                       快速积累准确率样本（解决"记录不足"问题）
    - _save_prediction() 每次预测后保存记录，次日自动回填 actual_up
    """

    def __init__(self, config: dict):
        self.config   = config
        self.seq_len  = config.get("model", {}).get("sequence_length", 60)
        self._cache: dict = {}   # symbol → (model, scaler, meta)

    # ── 主推理入口 ────────────────────────────────────────────

    def predict(self, symbol: str, df_factors: pd.DataFrame) -> dict:
        """
        预测次日上涨概率（同步，≤1秒）

        Returns:
            {success, probability, probability_pct, confidence,
             degraded, model_meta, predict_time}
        """
        model, scaler, meta = self._load(symbol)
        if model is None:
            return {
                "success": False,
                "reason": "no_model",
                "message": f"股票 {symbol} 尚未训练模型，请在【系统配置→模型训练】执行训练",
                "probability": None,
            }

        feature_cols = meta.get("feature_cols", [])
        X = self._make_input(df_factors, feature_cols, scaler)
        if X is None:
            return {"success": False, "reason": "input_error",
                    "message": "特征准备失败", "probability": None}

        try:
            model.eval()
            with torch.no_grad():
                prob = model(torch.FloatTensor(X).unsqueeze(0)).item()
        except Exception as e:
            logger.error(f"[{symbol}] 推理异常: {e}")
            return {"success": False, "reason": "inference_error",
                    "message": str(e), "probability": None}

        if np.isnan(prob) or not (0 <= prob <= 1):
            logger.warning(f"[{symbol}] 输出异常({prob:.4f})，降级为观望")
            return {"success": True, "probability": 0.5,
                    "confidence": "low", "degraded": True,
                    "message": "模型输出异常，已降级为观望", "model_meta": meta}

        # 保存今日预测，并回填昨日 actual_up
        self._save_and_update(symbol, prob, df_factors)

        return {
            "success":       True,
            "probability":   round(prob, 4),
            "probability_pct": round(prob * 100, 1),
            "confidence":    self._confidence(symbol, prob),
            "degraded":      False,
            "model_meta":    meta,
            "predict_time":  datetime.now().isoformat(),
        }

    # ── 历史回填（解决"记录不足"问题）────────────────────────

    def backfill_history(
        self,
        symbol: str,
        df_factors: pd.DataFrame,
        days: int = 60,
    ) -> int:
        """
        对历史数据批量预测并保存，快速积累准确率样本。
        每次训练后、或首次查看准确率时调用一次即可。

        Args:
            symbol:     股票代码
            df_factors: 完整因子数据（需超过 seq_len + days 行）
            days:       回填天数，默认60个交易日

        Returns:
            成功回填的记录数
        """
        model, scaler, meta = self._load(symbol)
        if model is None:
            return 0

        feature_cols = meta.get("feature_cols", [])
        pred_file    = PRED_DIR / f"{symbol}_predictions.parquet"

        # 已有记录，找出缺失的日期
        existing_dates: set = set()
        if pred_file.exists():
            try:
                ex = pd.read_parquet(pred_file)
                existing_dates = set(ex["date"].tolist())
            except Exception:
                pass

        df = df_factors.sort_values("date").reset_index(drop=True)
        # 只处理最近 days 个交易日
        target_rows = df.tail(days + self.seq_len)

        records = []
        for i in range(self.seq_len, len(target_rows)):
            row_date = str(target_rows["date"].iloc[i].date())
            if row_date in existing_dates:
                continue   # 跳过已有记录

            df_window = target_rows.iloc[:i + 1]
            X = self._make_input(df_window, feature_cols, scaler)
            if X is None:
                continue

            try:
                model.eval()
                with torch.no_grad():
                    prob = model(torch.FloatTensor(X).unsqueeze(0)).item()
            except Exception:
                continue

            if np.isnan(prob) or not (0 <= prob <= 1):
                continue

            # 对于历史数据，actual_up 可以直接计算（已知次日收盘）
            actual_up = None
            if i + 1 < len(target_rows):
                c_now  = target_rows["close"].iloc[i]
                c_next = target_rows["close"].iloc[i + 1]
                if c_now and c_next:
                    actual_up = 1 if c_next > c_now else 0

            records.append({
                "date":         row_date,
                "probability":  round(prob, 4),
                "predict_time": row_date + "T00:00:00",
                "actual_up":    actual_up,
            })

        if not records:
            return 0

        df_new = pd.DataFrame(records)

        if pred_file.exists():
            try:
                df_ex = pd.read_parquet(pred_file)
                df_new = pd.concat([df_ex, df_new], ignore_index=True)
                df_new = df_new.drop_duplicates(subset=["date"], keep="last")
            except Exception:
                pass

        df_new = df_new.sort_values("date").reset_index(drop=True)
        df_new.to_parquet(pred_file, index=False)
        logger.info(f"[{symbol}] 历史回填 {len(records)} 条预测记录")
        return len(records)

    # ── 历史准确率统计 ────────────────────────────────────────

    def get_accuracy_stats(self, symbol: str, days: int = 30) -> dict:
        """
        计算最近N日预测准确率。

        Returns:
            {accuracy, total, correct, recent_predictions, has_enough_data}
        """
        history = self._load_history(symbol)

        if history.empty:
            return {"accuracy": None, "total": 0, "correct": 0,
                    "has_enough_data": False,
                    "hint": "尚无预测记录，请先执行预测或在【系统配置→模型训练】后点击【回填历史准确率】"}

        # 只统计有实际结果的记录
        history = history.sort_values("date").tail(days)
        history = history.dropna(subset=["actual_up"])

        if len(history) < 5:
            return {"accuracy": None, "total": len(history), "correct": 0,
                    "has_enough_data": False,
                    "hint": f"有效记录仅 {len(history)} 条（需≥5条）。可在【AI预测页】点击【回填历史准确率】快速积累。"}

        history["pred_label"] = (history["probability"] > 0.5).astype(int)
        history["actual_up"]  = history["actual_up"].astype(int)
        correct = int((history["pred_label"] == history["actual_up"]).sum())
        total   = len(history)

        return {
            "accuracy":           round(correct / total, 4),
            "total":              total,
            "correct":            correct,
            "has_enough_data":    True,
            "hint":               "",
            "recent_predictions": history.tail(10).to_dict("records"),
        }

    # ── 预测记录保存与回填 ────────────────────────────────────

    def _save_and_update(
        self, symbol: str, prob: float, df_factors: pd.DataFrame
    ):
        """保存今日预测，同时回填昨日 actual_up"""
        pred_file = PRED_DIR / f"{symbol}_predictions.parquet"
        today     = datetime.now().strftime("%Y-%m-%d")

        new_rec = pd.DataFrame([{
            "date":        today,
            "probability": round(prob, 4),
            "predict_time": datetime.now().isoformat(),
            "actual_up":   None,
        }])

        if pred_file.exists():
            try:
                existing = pd.read_parquet(pred_file)

                # 回填：用今日 df_factors 的最新涨跌，更新昨日预测的 actual_up
                if "close" in df_factors.columns and "date" in df_factors.columns \
                        and len(df_factors) >= 2:
                    latest_date  = str(df_factors["date"].iloc[-1].date())
                    latest_close = df_factors["close"].iloc[-1]
                    prev_close   = df_factors["close"].iloc[-2]
                    if latest_close and prev_close and prev_close != 0:
                        actual = 1 if latest_close > prev_close else 0
                        mask = existing["date"] == latest_date
                        if mask.any():
                            existing.loc[mask, "actual_up"] = actual

                combined = pd.concat([existing, new_rec], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="last")
                combined.to_parquet(pred_file, index=False)
                return
            except Exception as e:
                logger.warning(f"[{symbol}] 预测记录更新失败: {e}")

        new_rec.to_parquet(pred_file, index=False)

    def _load_history(self, symbol: str) -> pd.DataFrame:
        p = PRED_DIR / f"{symbol}_predictions.parquet"
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
        return pd.DataFrame()

    # ── 模型加载 ─────────────────────────────────────────────

    def _load(self, symbol: str) -> Tuple[Optional[LSTMModel], Optional[object], Optional[dict]]:
        if symbol not in self._cache:
            m, s, meta = ModelTrainer.load_model(symbol)
            if m is not None:
                self._cache[symbol] = (m, s, meta)
            else:
                return None, None, None
        return self._cache[symbol]

    def invalidate_cache(self, symbol: str):
        self._cache.pop(symbol, None)

    # ── 特征准备 ─────────────────────────────────────────────

    def _make_input(
        self, df: pd.DataFrame, feature_cols: list, scaler
    ) -> Optional[np.ndarray]:
        if df.empty or len(df) < self.seq_len:
            return None
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            return None

        vals = df[available].tail(self.seq_len).ffill().bfill().fillna(0).values

        # 补零对齐训练时的特征维度
        if len(available) < len(feature_cols):
            padded = np.zeros((self.seq_len, len(feature_cols)), dtype=np.float32)
            padded[:, :len(available)] = vals
            vals = padded

        if scaler is not None:
            try:
                vals = scaler.transform(vals)
            except Exception:
                pass

        return vals.astype(np.float32)

    # ── 置信度 ────────────────────────────────────────────────

    def _confidence(self, symbol: str, prob: float) -> str:
        dist = abs(prob - 0.5)
        acc  = (self.get_accuracy_stats(symbol, 20).get("accuracy") or 0.5)
        if dist >= 0.15 and acc >= 0.55:
            return "high"
        if dist >= 0.08 or acc >= 0.52:
            return "medium"
        return "low"
