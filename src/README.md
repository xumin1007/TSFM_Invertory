# src/f2d — forecast-to-decision 实现

```bash
PYTHONPATH=src python -m f2d.run_zhao          # Zhao 端到端（约 2 分钟，首次含 Excel 解析约 3 分钟）
PYTHONPATH=src python -m f2d.run_zhao --no-cache   # 重新解析原始 Excel
```

## 模块

| 模块 | 职责 | 对应文档 |
| --- | --- | --- |
| `conventions.py` | 全局常量。不得被调用方覆盖 | `00_global_conventions.md` §1 |
| `config.py` | 加载 `configs/*.yaml` + 四条一致性断言 | `configs/README.md` |
| `metrics.py` | pinball、NPL、覆盖率、WAPE/WPE、y_min 阶梯 | §4–§5 |
| `postprocess.py` | 非有限值 → 非负截断 → 单调重排 | §2.3 |
| `encoding.py` | 冻结词表、`to_vocab`、`VocabStore` | §7 |
| `calibration.py` | 残差分位数校准器（L0/L1/L2 回退） | §6 |
| `checks.py` | 验收契约 JSON、退出码、复现哈希 | §9 |
| `datasets/zhao.py` | Zhao 原始审计 + 事前特征面板 | spec 02 |
| `models/gbdt.py` | LightGBM 分位数回归 | `06_model_hyperparams.md` §6.1 |
| `models/baselines.py` | 点预测基线 | §7 |
| `run_zhao.py` | 端到端编排 | spec 02 |

## 已通过的等价性验证

- **NPL 与阿里题 judge 逐位一致**（随机 500 样本，差 = 0.00e+00）
- **三个模型族的类别编码逐元素相同**（pandas / sklearn / chronos 均为 `[2,0,1,4]`）
- **y_min 阶梯**：整数需求 → 1.0；FreshRetailNet 类连续量 → 0.1

## 尚未实现

`datasets/` 只有 zhao；TSFM（Chronos-2）、统计基线（Croston/TSB）、
neuralforecast 深度基线、决策层（`07_decision_layer.md`）均未接入。
`run_zhao.py` 的编排结构可直接复用到其他数据集。
