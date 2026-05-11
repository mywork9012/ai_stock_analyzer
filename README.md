# AI股票量化分析软件 V1.0

> 基于 BiLSTM + Attention 的 A 股量化分析工具，融合技术面、基本面因子，自动生成具备三层风控的交易信号，并提供完整回测验证与可视化界面。

---

## 目录

- [快速开始](#快速开始)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [首次使用](#首次使用)
- [界面说明](#界面说明)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [版本说明](#版本说明)
- [免责声明](#免责声明)

---

## 快速开始

```bash
# 1. 克隆/下载项目
cd ai_stock_analyzer

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动程序
streamlit run main.py
```

浏览器自动打开 `http://localhost:8501`，**从零安装到看到界面约5~10分钟**。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | **3.10 或以上** |
| 操作系统 | Windows 10+ / macOS 12+ / Ubuntu 20.04+ |
| 内存 | 建议 8GB+（模型训练需要） |
| 磁盘 | 至少 2GB 可用空间（数据缓存） |
| 网络 | 需要访问 AkShare / Tushare 数据接口 |

---

## 安装步骤

### 第一步：安装 Python

前往 https://www.python.org/downloads/ 下载并安装 Python 3.10+。

安装时勾选 **"Add Python to PATH"**。

验证安装：
```bash
python --version   # 应显示 Python 3.10.x 或更高
pip --version
```

### 第二步：创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活（Windows）
venv\Scripts\activate

# 激活（macOS/Linux）
source venv/bin/activate
```

### 第三步：安装依赖

```bash
pip install -r requirements.txt
```

> **安装时间**：约5~15分钟（取决于网速）。PyTorch 较大，约500MB。

#### 国内网络加速（可选）

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 常见安装问题

**问题：`pandas-ta` 安装报错**
```bash
pip install pandas-ta --pre
```

**问题：`torch` 安装很慢**
```bash
# 仅安装CPU版PyTorch（无GPU机器推荐）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**问题：`vectorbt` 依赖冲突**
```bash
pip install vectorbt --no-deps
pip install numba scipy
```

### 第四步：启动程序

```bash
streamlit run main.py
```

---

## 首次使用

### 1. 配置股票池

打开程序后进入 **「⚙️ 系统配置」→「股票池」**，添加你想跟踪的股票（最多30只）。

也可以直接编辑 `config.yaml` 中的 `watchlist` 节：

```yaml
watchlist:
  "600519": "贵州茅台"
  "000858": "五粮液"
  "300750": "宁德时代"
  # 继续添加...
```

### 2. 训练模型

进入 **「⚙️ 系统配置」→「模型训练」**，点击「训练此股票」。

- **单只股票**：约2~5分钟（CPU模式）
- **30只全量**：约1~4小时，建议周末执行

> 首次使用必须训练模型，否则预测页面会提示"尚未训练模型"。

### 3. 查看信号

回到 **「🏠 首页」**，点击「刷新今日数据」，即可看到所有股票的今日交易信号。

---

## 界面说明

### 🏠 首页·总览

- **信号统计**：今日买入/观望/卖出信号数量汇总
- **信号明细表**：所有股票的信号详情，支持按类型筛选
- **风控横幅**：显示熔断/防御模式等风控状态

### 📈 单股分析

- **K线图**：支持叠加 SMA/EMA/布林带等指标，可勾选/取消
- **RSI图**：显示超买超卖状态
- **基本面指标**：PE/PB/ROE等
- **因子热力图**：近30日各因子值分布

### 🤖 AI预测

- **概率仪表盘**：显示次日上涨概率（0-100%）
- **信号徽章**：买入/观望/卖出信号及建议仓位
- **历史准确率**：近30日预测准确率统计
- **AI解释**：点击「查看AI解释」查看Top-5因子贡献度

### 📡 信号中心

- 按信号类型/置信度/波动状态多维筛选
- 展示信号触发原因和建议仓位
- 风控黑名单查看

### 📋 回测报告

- **参数配置**：自定义起止日期、初始资金、交易成本
- **绩效指标**：年化收益/Sharpe/MDD/Calmar/胜率
- **资金曲线**：含回撤阴影区
- **月度热力图**：各月收益分布
- **交易明细**：可导出CSV

### ⚙️ 系统配置

- **股票池管理**：添加/删除股票
- **模型训练**：单只/批量训练，查看训练状态
- **数据源配置**：切换AkShare/Tushare，填入Token
- **风控参数**：调整止损比例、仓位上限等
- **数据注册表**：查看数据获取状态，清除缓存

---

## 配置说明

所有配置集中在 `config.yaml`，主要参数如下：

```yaml
# 股票池（最多30只）
watchlist:
  "600519": "贵州茅台"

# 数据配置
data:
  primary_source: "akshare"     # 主数据源
  tushare_token: ""             # 可选：填入提升数据质量
  cache_days: 7                 # 缓存有效期

# 模型配置
model:
  type: "lstm"                  # V1.0轻量版
  sequence_length: 60           # 输入时间步（天）
  hidden_units: 64
  max_epochs: 100

# 信号配置
signal:
  base_buy_threshold: 0.55     # 买入概率阈值
  dynamic_threshold: true      # 是否启用动态阈值

# 风控配置
risk:
  stop_loss_pct: 0.08          # 固定止损8%
  max_position_pct: 0.20       # 单只最大仓位20%
  daily_loss_limit: 0.03       # 日亏损熔断3%
```

---

## 常见问题

**Q：启动后浏览器没有自动打开？**

手动访问 `http://localhost:8501`

**Q：AkShare 数据获取失败/超时？**

AkShare 依赖公开数据接口，偶尔会有限速或接口变更。解决方案：
1. 稍等几分钟后重试
2. 配置 Tushare Pro Token 作为备用源（[申请地址](https://tushare.pro/register)）

**Q：模型训练很慢？**

V1.0 使用轻量单层LSTM，单只股票训练约2~5分钟（CPU）。如果更慢：
- 检查 `config.yaml` 中 `max_epochs` 是否过高（默认100）
- 尝试降低 `sequence_length`（默认60）

**Q：预测准确率不高？**

- 确保训练数据完整（至少2年）
- 尝试先在「单股分析」页查看技术面是否符合常规
- AI预测仅为参考，需结合自身判断

**Q：回测结果显示"降级回测引擎"？**

```bash
pip install vectorbt
```
如安装失败（常见于 Windows），系统会自动使用内置简化引擎，结果仍可参考。

**Q：想修改交易信号阈值？**

在 `config.yaml` 中修改 `signal.thresholds` 节，或在「系统配置→风控参数」页面调整。

---

## 版本说明

| 版本 | 状态 | 主要功能 |
|------|------|----------|
| **V1.0（当前）** | ✅ 可用 | 数据获取+技术/基本面因子+单层LSTM+回测+完整界面 |
| V1.5 | 🔄 规划中 | BiLSTM+Attention+FinBERT舆情+完整风控三层 |
| V2.0 | 🔄 规划中 | SHAP异步解释+底仓T+0+回测报告增强 |
| V2.5 | 🔄 规划中 | 多策略并行+因子热插拔+模型版本管理 |

---

## 项目结构

```
ai_stock_analyzer/
├── main.py                    # Streamlit 主入口
├── config.yaml                # 全局配置文件（在此修改参数）
├── requirements.txt           # 依赖清单
├── README.md                  # 本文档
│
├── core/                      # 核心业务模块
│   ├── data_manager.py        # 数据获取与缓存
│   ├── factor_engine.py       # 因子计算
│   ├── predict_engine.py      # AI预测
│   ├── signal_generator.py    # 信号生成
│   ├── risk_manager.py        # 风控管理
│   └── backtest_engine.py     # 回测引擎
│
├── models/                    # AI模型
│   ├── lstm_model.py          # LSTM模型定义
│   ├── trainer.py             # 训练器
│   └── saved/                 # 已训练模型文件
│
├── ui/                        # 界面层
│   ├── pages/                 # 各功能页面
│   └── components/            # 可复用组件
│
└── data/                      # 本地数据（自动生成）
    ├── parquet/               # 行情/因子缓存
    ├── database/              # SQLite元数据
    └── shap_cache/            # SHAP结果缓存
```

---

## 免责声明

> ⚠️ 本软件仅提供基于历史数据的 AI 预测参考，**不构成任何投资建议**。
>
> 股票投资存在风险，历史回测表现不代表未来收益。用户需独立做出投资决策并承担相应后果。
>
> 开发者对因使用本软件产生的任何投资损失不承担责任。
