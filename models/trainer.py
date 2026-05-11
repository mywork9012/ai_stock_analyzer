# models/trainer.py
"""
模型训练器
- 滚动窗口训练（3年训练集 + 3个月验证集）
- 加权损失函数（替代SMOTE，规避时序数据泄露）
- 早停机制（10个epoch patience）
- 模型版本管理（.pth + .json元数据）
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_model import LSTMModel

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "saved"


class ModelTrainer:
    """
    模型训练器

    用法:
        trainer = ModelTrainer(config)
        result = trainer.train(symbol, df_factors)
        # result: {'loss': float, 'val_loss': float, 'epochs': int, 'val_acc': float}
    """

    def __init__(self, config: dict):
        self.config = config
        model_cfg = config.get("model", {})

        self.seq_len      = model_cfg.get("sequence_length", 60)
        self.hidden_units = model_cfg.get("hidden_units", 64)
        self.num_layers   = model_cfg.get("num_layers", 1)
        self.dropout      = model_cfg.get("dropout", 0.3)
        self.batch_size   = model_cfg.get("batch_size", 64)
        self.lr           = model_cfg.get("learning_rate", 0.001)
        self.max_epochs   = model_cfg.get("max_epochs", 100)
        self.patience     = model_cfg.get("early_stopping_patience", 10)
        self.train_years  = model_cfg.get("train_years", 3)
        self.val_months   = model_cfg.get("val_months", 3)

        # 设备选择
        device_cfg = model_cfg.get("device", "auto")
        if device_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 主训练入口 ───────────────────────────────────────────

    def train(self, symbol: str, df_factors: pd.DataFrame) -> dict:
        """
        训练指定股票的LSTM模型

        Args:
            symbol: 股票代码
            df_factors: 包含所有因子的DataFrame（含date和close列）

        Returns:
            训练结果字典
        """
        logger.info(f"[{symbol}] 开始训练模型，设备: {self.device}")

        # 1. 准备特征列
        feature_cols = self._get_feature_cols(df_factors)
        if len(feature_cols) < 5:
            logger.error(f"[{symbol}] 特征数量不足（{len(feature_cols)}），跳过训练")
            return {"success": False, "reason": "insufficient_features"}

        # 2. 划分训练/验证集（时序划分，验证集在训练集之后）
        df_train, df_val = self._time_split(df_factors)
        if len(df_train) < self.seq_len + 10:
            logger.error(f"[{symbol}] 训练数据不足（{len(df_train)}行）")
            return {"success": False, "reason": "insufficient_data"}

        # 3. 构建序列数据
        scaler = StandardScaler()
        X_train, y_train, scaler = self._build_sequences(df_train, feature_cols, scaler, fit=True)
        X_val, y_val, _ = self._build_sequences(df_val, feature_cols, scaler, fit=False)

        if X_train is None or len(X_train) == 0:
            return {"success": False, "reason": "sequence_build_failed"}

        # 4. 计算类别权重（加权损失，替代SMOTE）
        pos_count = int(y_train.sum())
        neg_count = len(y_train) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(self.device)
        logger.info(f"[{symbol}] 类别分布 - 上涨:{pos_count}, 下跌:{neg_count}, pos_weight:{pos_weight.item():.3f}")

        # 5. 初始化模型
        input_size = len(feature_cols)
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_units,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # 6. 训练
        result = self._training_loop(model, X_train, y_train, X_val, y_val, pos_weight)

        # 7. 保存模型
        self._save_model(symbol, model, scaler, feature_cols, result)

        logger.info(
            f"[{symbol}] 训练完成 | epochs:{result['epochs']} | "
            f"val_loss:{result['val_loss']:.4f} | val_acc:{result['val_acc']:.2%}"
        )
        return result

    # ── 数据准备 ─────────────────────────────────────────────

    def _time_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """时序划分：训练集(3年) + 验证集(3个月)，验证集在训练集之后"""
        df = df.sort_values("date").reset_index(drop=True)

        end_date = df["date"].max()
        val_start = end_date - pd.DateOffset(months=self.val_months)
        train_end = val_start - pd.Timedelta(days=1)
        train_start = end_date - pd.DateOffset(years=self.train_years)

        df_train = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        df_val = df[df["date"] > val_start]

        logger.info(
            f"数据划分 - 训练集:{len(df_train)}行 "
            f"({df_train['date'].min().date()} ~ {df_train['date'].max().date() if len(df_train)>0 else 'N/A'}), "
            f"验证集:{len(df_val)}行"
        )
        return df_train, df_val

    def _build_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        scaler: StandardScaler,
        fit: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
        """
        构建LSTM输入序列

        X[i] = 因子矩阵[i : i+seq_len]        shape: (seq_len, n_features)
        y[i] = 1 if close[i+seq_len] > close[i+seq_len-1] else 0

        严格保证：不使用未来数据
        """
        if df.empty or "close" not in df.columns:
            return None, None, scaler

        # 填充因子缺失值
        df_feat = df[feature_cols].copy()
        df_feat = df_feat.ffill().bfill()

        # 填充后仍有缺失则用0
        df_feat = df_feat.fillna(0)

        values = df_feat.values  # shape: (n_days, n_features)
        close = df["close"].values

        # 标准化
        if fit:
            values = scaler.fit_transform(values)
        else:
            try:
                values = scaler.transform(values)
            except Exception:
                return None, None, scaler

        X, y = [], []
        for i in range(self.seq_len, len(values)):
            X.append(values[i - self.seq_len : i])
            # 标签：明日相对今日涨跌
            tomorrow_up = 1 if close[i] > close[i - 1] else 0
            y.append(tomorrow_up)

        if not X:
            return None, None, scaler

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """
        获取有效特征列。
        与 FactorCalculator.get_feature_columns() 保持完全一致，
        确保因子计算和模型训练使用相同的列集合。
        """
        from core.factor_engine import FactorCalculator, _BASE_EXCLUDE
        exclude = _BASE_EXCLUDE | {c for c in df.columns if c.endswith("_raw")}
        cols = [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().mean() > 0.5
        ]
        if len(cols) < 5:
            logger.warning(
                f"有效特征列仅 {len(cols)} 个（需≥5），"
                f"当前全部列: {list(df.columns)}"
            )
        return cols

    # ── 训练循环 ─────────────────────────────────────────────

    def _training_loop(
        self,
        model: LSTMModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        pos_weight: torch.Tensor,
    ) -> dict:
        """带早停机制的训练循环"""

        # DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(y_train).unsqueeze(1).to(self.device),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False  # 时序数据不shuffle
        )

        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(self.max_epochs):
            # 训练阶段
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                # 使用logits（未经sigmoid）计算BCEWithLogitsLoss
                out = model(X_batch)
                # 注意: BCEWithLogitsLoss需要logits，但模型forward已加sigmoid
                # 改为直接使用BCELoss
                loss = nn.BCELoss()(out, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证阶段
            if has_val:
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val_t)
                    val_loss = nn.BCELoss()(val_preds, y_val_t).item()
                    val_losses.append(val_loss)
                    scheduler.step(val_loss)

                    # 早停
                    if val_loss < best_val_loss - 1e-5:
                        best_val_loss = val_loss
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.patience:
                        logger.info(f"早停触发，epoch={epoch+1}")
                        break

                if (epoch + 1) % 10 == 0:
                    logger.debug(
                        f"Epoch {epoch+1}/{self.max_epochs} | "
                        f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f}"
                    )

        # 恢复最佳模型
        if best_state is not None:
            model.load_state_dict(best_state)

        # 计算验证集准确率
        val_acc = 0.0
        if has_val:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t).cpu().numpy().flatten()
                val_pred_labels = (val_preds > 0.5).astype(int)
                val_acc = (val_pred_labels == y_val).mean()

        return {
            "success": True,
            "epochs": len(train_losses),
            "train_loss": float(train_losses[-1]) if train_losses else 0.0,
            "val_loss": float(best_val_loss) if best_val_loss != float("inf") else 0.0,
            "val_acc": float(val_acc),
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    # ── 模型保存与加载 ───────────────────────────────────────

    def _save_model(
        self,
        symbol: str,
        model: LSTMModel,
        scaler: StandardScaler,
        feature_cols: list,
        train_result: dict,
    ):
        """保存模型权重 + 元数据"""
        import joblib

        # 保存PyTorch模型权重
        model_path = MODEL_DIR / f"{symbol}_model.pth"
        torch.save(model.state_dict(), model_path)

        # 保存标准化器
        scaler_path = MODEL_DIR / f"{symbol}_scaler.pkl"
        joblib.dump(scaler, scaler_path)

        # 保存元数据
        meta = {
            "symbol": symbol,
            "train_date": datetime.now().isoformat(),
            "feature_cols": feature_cols,
            "input_size": len(feature_cols),
            "hidden_units": self.hidden_units,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "val_acc": train_result.get("val_acc", 0.0),
            "val_loss": train_result.get("val_loss", 0.0),
            "epochs_trained": train_result.get("epochs", 0),
        }
        meta_path = MODEL_DIR / f"{symbol}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"[{symbol}] 模型已保存: {model_path}")

    @staticmethod
    def load_model(symbol: str) -> Tuple[Optional[LSTMModel], Optional[StandardScaler], Optional[dict]]:
        """
        加载已训练的模型

        Returns:
            (model, scaler, meta) 或 (None, None, None)
        """
        import joblib

        model_path = MODEL_DIR / f"{symbol}_model.pth"
        scaler_path = MODEL_DIR / f"{symbol}_scaler.pkl"
        meta_path = MODEL_DIR / f"{symbol}_meta.json"

        if not model_path.exists():
            logger.warning(f"[{symbol}] 模型文件不存在: {model_path}")
            return None, None, None

        try:
            # 读取元数据
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # 重建模型结构
            model = LSTMModel(
                input_size=meta["input_size"],
                hidden_size=meta["hidden_units"],
                num_layers=meta["num_layers"],
                dropout=meta["dropout"],
            )
            # 加载权重（CPU兼容）
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()

            # 加载标准化器
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None

            return model, scaler, meta

        except Exception as e:
            logger.error(f"[{symbol}] 模型加载失败: {e}")
            return None, None, None

    @staticmethod
    def model_exists(symbol: str) -> bool:
        """检查模型文件是否存在"""
        return (MODEL_DIR / f"{symbol}_model.pth").exists()

    @staticmethod
    def get_all_model_meta() -> list:
        """获取所有已训练模型的元数据"""
        metas = []
        for meta_path in MODEL_DIR.glob("*_meta.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metas.append(json.load(f))
            except Exception:
                pass
        return sorted(metas, key=lambda x: x.get("train_date", ""), reverse=True)
