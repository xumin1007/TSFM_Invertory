# 模型身份、超参数与算力预算

## 0. 范围与优先级

本文件固定**每个 `model_id` 的库、版本、完整超参、随机种子与算力预算**。机制与公式见 `docs/00_global_conventions.md`；切分边界与逐数据集常量见 `configs/*.yaml`。三者不重复定义同一事项，冲突时以 `00_global_conventions.md` 为准。

本文件中的所有实测数字均在下述环境上取得，写入 `run_manifest.json` 时须记录实际环境：

```
platform      Darwin arm64 (Apple Silicon)，MPS 可用，无 CUDA
python        3.12.2
torch         2.13.0        transformers  5.14.1
chronos-forecasting 2.3.1   neuralforecast 3.2.0
lightgbm      4.6.0         scikit-learn  1.5.1
peft          0.19.1        accelerate    1.14.0
numpy         1.26.4        pandas        2.1.4
measured_on   2026-08-22
```

---

## 1. TSFM 身份确认

`pyproject.toml` 钉的是 `chronos-forecasting>=2.3,<3`；本地实装 **2.3.1**。该版本包含三个互不兼容的族：

| Pipeline | 输出类型 | 有 `fit()` | 支持协变量 | 本项目使用 |
| --- | --- | --- | --- | --- |
| `ChronosPipeline`（初代 T5） | 采样路径 | ✗ | ✗ | 否 |
| `ChronosBoltPipeline` | 分位数 | ✗ | ✗ | 否 |
| **`Chronos2Pipeline`** | **分位数** | **✓** | **✓（含已知未来）** | **是** |

选定 `Chronos2Pipeline` 的三条实测依据：

1. **它是包内唯一提供 `fit()` 的族。** 本研究要求 zero-shot 与微调的对照，另两族不具备微调入口。
2. **`forecast_type = ForecastType.QUANTILES`，`predict_quantiles(quantile_levels=[...])` 直接产出分位数。** 按 `00_global_conventions.md` §6.1，TSFM 因此走 `quantile_source="native"`，**不经残差校准器**。
3. **`predict_df(df, future_df, ...)` 与 `preprocess.from_data_frame(known_covariates_names=...)` 支持已知未来协变量与多目标**，可与 GBDT 使用同一特征集，使二者的比较不被"谁能看到协变量"污染。

本地 HF 缓存另有 `Salesforce/moirai-1.1-R-base`（uni2ts 2.0.0）与 `google/timesfm-2.5-200m-pytorch`（timesfm 2.0.2）。二者**不进入主比较**（`pyproject.toml` 未钉、接口与协变量能力未审计），登记为第二阶段可选扩展。

---

## 2. 尺寸选择：`amazon/chronos-2`（119.5M），而非 `chronos-2-small`（27.9M）

两个 checkpoint 均已在本地 HF 缓存中，实测规格：

| | `autogluon/chronos-2-small` | `amazon/chronos-2` |
| --- | --- | --- |
| 参数量 | 27.9M | **119.5M** |
| `default_context_length` | 2048 | 2048 |
| `model_context_length` | 8192 | 8192 |
| `model_prediction_length` | 1024 | 1024 |
| 原生分位数网格 | 13 点：`.01 .05 .1 .2 .3 .4 .5 .6 .7 .8 .9 .95 .99` | 21 点：`.01 .05 .1 .15 .2 … .8 **.85** .9 .95 .99` |
| **含 $\tau=0.85$** | **✗（需在 .8/.9 间插值）** | **✓ 原生** |
| 含 §5.2 校准网格 `{.05,.10,…,.95}` 全部 19 点 | ✗（仅 9 点原生） | **✓ 全部原生** |
| 吞吐（MPS，512 序列 ctx365 h7） | 892 series/s | 268 series/s |
| 吞吐（CPU，同上） | 547 series/s | 167 series/s |

**决策依据**：本项目的全部评价接口是 $(q_{.50}, q_{.85})$。base 版把 $\tau=0.85$ 作为**训练过的**分位数直接输出，small 版必须在 $0.8$ 与 $0.9$ 之间插值——这会给 TSFM 的上尾表现引入一层与模型能力无关的插值误差，而上尾正是本研究的核心观察量。同理，`00_global_conventions.md` §5.2 的 19 点校准曲线在 base 上全部原生，在 small 上过半需插值。代价是可接受的：268 series/s 对应全项目 zero-shot 推理约 1 小时（§9）。

`chronos-2-small` 保留为**尺寸敏感性臂**（§3 的 `chronos2s-zs`），用于回答"TSFM 的增益是否只来自参数规模"。它成本极低（约主臂的 1/3 时间），但其 $q_{.85}$ 须标注为插值值，不与 base 的原生值并列声称同等可靠。

---

## 3. TSFM 对比矩阵（zero-shot × 微调）

四个必跑臂 + 两个登记消融。所有臂共享同一 split、同一特征集、同一后处理（`00_global_conventions.md` §2.3）。

| `model_id` | checkpoint | 模式 | 协变量 | 目的 |
| --- | --- | --- | --- | --- |
| `chronos2-zs` | `amazon/chronos-2` | zero-shot | 有 | **主 zero-shot 臂** |
| `chronos2-ft-lora` | `amazon/chronos-2` | LoRA 微调 | 有 | **主微调臂**（低成本、抗过拟合） |
| `chronos2-ft-full` | `amazon/chronos-2` | 全参微调 | 有 | 微调上限；与 LoRA 对照 |
| `chronos2s-zs` | `autogluon/chronos-2-small` | zero-shot | 有 | 尺寸敏感性 |
| `chronos2-zs-univar` | `amazon/chronos-2` | zero-shot | **无** | 登记消融：隔离协变量贡献 |
| `chronos2-ft-lora-te` | `amazon/chronos-2` | LoRA 微调 | 有 | 登记消融：`use_target_encoding=True`（见 §4.3），编码方式敏感性 |

主比较回答三个问题，均在**冻结测试窗**上、以 NPL 报告：

1. `chronos2-zs` vs GBDT — 零样本 TSFM 能否在无训练成本下达到结构化基线水平；
2. `chronos2-ft-lora` / `chronos2-ft-full` vs `chronos2-zs` — 微调的增量，以及全参是否值得其额外成本；
3. `chronos2-zs` vs `chronos2s-zs` — 增益中有多少来自参数规模。

差异必须以 `00_global_conventions.md` §9.5 的**配对 bootstrap** 报告 $\Delta\text{NPL}$ 的 CI，不比较各自 CI 是否重叠。

---

## 4. TSFM 通用配置（全部臂共用）

### 4.1 加载与推理

```python
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="mps",          # 无 MPS 时 "cpu"；有 CUDA 时 "cuda"
    torch_dtype=torch.float32, # 固定 float32；bfloat16 仅作登记的速度敏感性
)
q, _ = pipe.predict_quantiles(
    inputs, prediction_length=H,
    quantile_levels=[0.50, 0.85],   # 与评价接口一致
    context_length=CTX, batch_size=256,
)
```

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `torch_dtype` | `float32` | 固定，保证跨设备可复现；`bfloat16` 只作速度敏感性，不进主结果 |
| `context_length` | `configs.<ds>.history.history_window_days`（Zhao 用月数） | 不用模型默认 2048，改为与其他模型同一历史窗，保证公平 |
| `prediction_length` | `configs.<ds>.calendar.horizon_days`（Zhao 为 1 月） | |
| `batch_size` | 256 | 仅影响速度，不影响结果 |
| `quantile_levels` | `[0.50, 0.85]`；校准曲线另跑 `{0.05,…,0.95}` | |
| `cross_learning` | `False` | 开启会让同批序列互相影响，破坏逐序列可归因性；登记为消融 |
| `limit_prediction_length` | `False` | 全部视界均远小于 1024 |

### 4.2 协变量映射

用 `chronos.chronos2.preprocess.from_data_frame` 构造输入，`known_covariates_names` 取自各数据集配置的 `future_known_covariates.allowed`：

| 数据集 | `known_covariates_names` | 说明 |
| --- | --- | --- |
| FreshRetailNet (LT / 50K) | `holiday_flag, discount, activity_flag` | 天气实况在 `forbidden`，不得传入 |
| Zhao | 无已知未来协变量 | 月初快照属历史协变量，非未来已知 |
| OSA | `units_under_promotion` | 须确认促销计划在 origin 时点可知；未确认前置空 |
| Lokad | 无 | |
| Mendeley | 无 | |

历史协变量（库存状态、在订量等）通过 `from_data_frame` 的非 known 列传入。**特征集必须与 GBDT 完全一致**，差异写入 `run_manifest.json` 的 `feature_list`。

### 4.3 类别编码：`use_target_encoding=False`

`from_data_frame` / `from_list_of_dicts` 的 `use_target_encoding` **默认为 `True`**，仅对 pandas `category` dtype 的协变量生效（数值列走另一路径）。其机制（`preprocess._target_encode`）为

$$\text{encoded}[\text{item},\text{cat}]=\frac{\text{smooth}\cdot\text{item\_mean}+\text{category\_sum}}{\text{smooth}+\text{category\_count}},\quad \text{smooth}=1$$

统计口径是**逐序列、且只取 context 段的 target**。

#### 它不属于 §7.2 禁令的射程

已通过读源码确认：传入 `_target_encode` 的 `target` 形状为 `(n_targets, total_context_rows)`，**预测窗的目标值不在该数组中**；future 行的编码值由 context 统计导出；统计不跨序列汇总。因此相对预测窗**不构成泄漏**。

`00_global_conventions.md` §7.2 禁的是经典的**跨行 label-mean 编码**（某行自身标签参与其自身编码值，或在含验证/测试的全量上拟合）。Chronos-2 的这个编码不满足该特征。**§7.2 的禁令不适用于此**，二者不冲突。

#### 但本项目仍显式传 `use_target_encoding=False`

理由是**退化性**，不是泄漏。实测（两条序列，静态类别 A/B，目标均值 10/50）：

| | series 0 (cat=A, mean=10) | series 1 (cat=B, mean=50) |
| --- | --- | --- |
| `use_target_encoding=True` | 编码值 **10.0** | 编码值 **50.0** |
| `use_target_encoding=False` | 编码值 **0**（A 的 ordinal 码） | 编码值 **1**（B 的 ordinal 码） |

对**序列内为常量的静态类别**，$\text{category\_sum}/\text{category\_count}\equiv\text{item\_mean}$，故编码值塌为该序列均值，类别身份被完全抹除：同类别不同水平的序列取不同值，不同类别相同水平的序列取相同值。

三条理由：

1. **我们的类别几乎全是静态属性**（`third_category_id`、`part_category`、`warehouse_tier`、`Product Category` 等），按上表一律退化。该编码只对序列内随时间变化的类别列有意义，本项目几乎没有这类列。
2. **稀疏序列上更差**：Mendeley 平均 2.5 行/序列、Zhao 28,626 SKU × 10 月，`category_count` 趋近 0，编码值被平滑项支配，等于加入一路与目标序列冗余的噪声通道。
3. **可比性**：GBDT 用 ordinal + 冻结词表 + `__UNK__`；TSFM 使用同一口径，可使 TSFM vs GBDT 的差异不混入编码方式这一因子。

#### 反向考量与消融安排

`predict_df` 内部调用 `from_data_frame` 时**不传该参数**，即走默认 `True`——这是库的 canonical 路径，改用 ordinal 存在偏离模型训练分布的风险。该风险**以实验回答，不以判断回答**：

`chronos2-ft-lora-te` 臂保留 `use_target_encoding=True`，性质为**编码方式敏感性**（非泄漏审计）。若该臂显著优于 `chronos2-ft-lora`，说明 Chronos-2 确实期望目标编码输入，届时须把主臂切换过去并在报告中说明该次切换及其证据。

#### 数据密度不改变该结论，只强化它

`use_target_encoding` 只对**序列内随时间变化**的类别列才可能有意义。实测各数据集所有类别列的「序列内取值数 > 1 的序列占比」：

| 数据集 | 类别列 | 变化占比 |
| --- | --- | --- |
| fresh_lt / fresh_50k | `city_id`, `store_id`, `*_category_id`, `product_id` | 0（全静态） |
| zhao | `category`, `subcategory`, `brand_ID`, `operation_mode` | 0 |
| osa | `product_category` | 0.0000 |
| lokad | `Category`, `Ref`, `Loc` | 0.0000 |
| mendeley | `Product Category` / `Segment` / `Supplier` / `Sales Channel` / `Store Type` | 0.0000 |
| mendeley | **`Sales Type`** | **0.167** |
| mendeley | **`Stock Status`** | **0.122** |

全项目**只有 Mendeley 的两列**是时变的。而恰恰在 Mendeley 上，即便只看这些时变序列：观测数中位仅 3（恒定序列为 2），**`category_count` 中位数 = 1，63% 的 (序列, 类别) 组只有一行**。代入平滑式（`smooth=1`）：

$$\text{encoded}=\frac{1\cdot\text{item\_mean}+y_{\text{that row}}}{1+1}=\frac{\text{item\_mean}+y_{\text{that row}}}{2}$$

即 63% 的情形下编码值是「该行自身目标与序列均值的平均」——不是类别表示，而是目标的噪声副本。

**结论：`use_target_encoding=False` 在六个数据集上一致适用。** 密度差异不构成例外，反而使唯一可能的例外（Mendeley 时变列）成为最差的情形。

#### 未见类别的处理已统一（原「未统一项」已解除）

早期版本记录了一处分叉：Chronos-2 的 ordinal 路径把未见类别映射为 `NaN` sentinel，GBDT 侧映射为 `__UNK__`。**该分叉可完全消除**，且已在 `00_global_conventions.md` §7.2 固定为统一机制。

关键事实（实测）：Chronos-2 的 sentinel 分支判定依据是 pandas `CategoricalDtype.categories` **声明**，不是数据中实际出现过的取值。只要 past 与 future 两个 DataFrame 都把 `categories` 设为完整冻结词表（含 `__UNK__` / `__MISSING__`），该分支不可达，三个模型族产出**逐元素相同**的整数码。

因此 `feature_list` 的备注要求改为：记录所用 `VOCAB` 的哈希，并断言 past/future 两侧 `categories` 一致；不再需要声明「两模型处理不同」。

### 4.4 分位数路由与后处理

`quantile_source = "native"`（base 版 $\tau\in\{.5,.85\}$ 均为训练过的分位点）。`chronos2s-zs` 记 `"native_interpolated"`。

输出仍须走 `00_global_conventions.md` §2.3 的完整后处理：非有限值检出 → 非负截断 → 单调重排，并记录 `crossing_rate`。

---

## 5. 微调协议

### 5.1 时间隔离（最高优先，可断言）

`fit()` 的 `inputs` 只能包含**训练窗内**的序列片段，即每条序列在 `configs.<ds>.splits.train.origins[1] + horizon - 1` 处截断。`validation_inputs` 取验证窗，用于 checkpoint 选择。测试窗数据在微调的任何阶段都不得出现。

必须固化断言：`max(timestamp in fit inputs) < min(validation origin)`，以及 `max(timestamp in validation_inputs) < min(test origin)`。

### 5.2 重拟合规则

**每个数据集微调一次**，在训练窗末完成，随后冻结用于全部验证与测试 origin。不在每个 origin 重拟合——成本不成比例，且会让"微调增益"与"重拟合频率"混淆。

唯一例外：**Zhao 的 prequential 协议**允许在 2019-10 起点前用截至 9 月的实际历史重拟合一次。该次重拟合必须使用与首次完全相同的超参，且禁止依据 9 月得分做任何选择（`configs/zhao.yaml: splits.prequential_rule`）。

### 5.3 超参

```python
ft = pipe.fit(
    inputs=train_inputs, validation_inputs=val_inputs,
    prediction_length=H, context_length=CTX,
    finetune_mode=MODE,          # "lora" | "full"
    learning_rate=LR, num_steps=1000, batch_size=32,
    min_past=MIN_PAST, output_dir=..., finetuned_ckpt_name="ft",
    remove_printer_callback=True, disable_data_parallel=True,
)
```

| 参数 | `chronos2-ft-lora` | `chronos2-ft-full` | 依据 |
| --- | --- | --- | --- |
| `finetune_mode` | `"lora"` | `"full"` | |
| `learning_rate` | `1e-5` | `1e-6` | 库文档明示：LoRA 建议用更高 LR（如 1e-5），全参默认 1e-6 |
| `num_steps` | 1000 | 1000 | 库默认；实际 checkpoint 由 `validation_inputs` 选择 |
| `batch_size` | 32 | 32 | 库默认 256 是**序列数**（含目标与协变量）；本项目多协变量，降至 32 以控显存并使实测预算成立 |
| `lora_config` | 库默认 | — | 首轮不调；调整须登记为新臂 |
| `min_past` | $\max(2H,\,28)$ | 同 | 保证每个训练样本有足够历史 |
| `context_length` | 同 §4.1 | 同 | |
| `random seed` | 42 | 42 | 方差报告用 `{42,43,44}` |

`num_steps=1000` 是**固定预算**，不按数据集调整；这样"微调增益"在各数据集间的比较不被训练步数差异干扰。若某数据集的验证损失在 1000 步内明显未收敛，须记录该事实并作为结论的限定条件，而不是私自加步数。

---

### 5.4 实测（Zhao，验证窗，2026-08-23）

**对 §5.1 的一处刻意偏离。** §5.1 规定 `validation_inputs` 取**验证窗**做
checkpoint 选择——那个设计假定最终只在测试窗报数。本轮在验证窗上报数，若
再用验证窗选 checkpoint，报出的微调增益就是乐观偏倚的。故改为从**训练窗
末尾**切出内部验证段（`2019-06-24 ~ 06-30`，即最后 $H=7$ 天），微调输入
截至 `2019-06-23`，验证窗在微调的任何阶段都不出现。断言已固化。

代价：内部验证段（6 月末）与真实验证窗（7–8 月）分布可能不同，checkpoint
选择因此不是对目标分布最优的。这个代价必须付——反之会污染结论。

| 臂 | NPL | `cov50_pos` | `cov85_pos` | $\Delta$NPL vs `zs` | 95% CI | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| `chronos2-ft-full` | **0.28618** | 0.4007 | 0.7764 | $-0.00322$ | $[-0.0055,-0.0016]$ | 显著 |
| `chronos2-ft-lora` | 0.28812 | 0.3802 | 0.7695 | $-0.00129$ | $[-0.0030,+0.0001]$ | **不显著** |
| `chronos2-zs` | 0.28940 | 0.4365 | 0.8011 | — | — | — |

**① 微调增益极小。** 全参显著但只有 1.1% 相对改善；对照零样本相对经验基线的
优势是 15.3%，微调只贡献了其 **6%**。LoRA 完全不显著。**Chronos-2 在本数据集
上的价值几乎全部来自预训练，不来自领域适应。**

**② 固定预算下未收敛，如实登记（§5.3 要求）。** `num_steps=1000` 在
`batch_size=32`、1868 条序列上仅约 **1 个 epoch**。`eval_loss` 轨迹：

| | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| LoRA | 2.062 | 2.059 | 2.056 | 2.053 | 2.052 | 2.052 |
| 全参 | 2.057 | 2.048 | 2.039 | 2.036 | 2.036 | 2.036 |

LoRA 到预算末仍单调下降，属**欠训练**；全参在 0.6 epoch 处取得最优
（`checkpoint-600`，2.034）后回升，checkpoint 选择确实起了作用。按 §5.3，
不加步数，把「LoRA 结论以 1 epoch 预算为限定条件」写入结论。

**两个控制臂**（为拆开决策层的视界混淆而训，见 07 §6.2 结论 ④）：

| 臂 | 微调 $H$ | 训练截止 | 序列数 | 最优 ckpt | NPL | $\Delta$ vs `zs` | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `chronos2-ft-full` | 7 | 06-23 | 1868 | 600 | 0.28618 | $-0.00322$ | 显著 |
| `chronos2-ft-full-short` | 7 | 05-29 | 1853 | — | 0.28794 | $-0.00146$ | **不显著** |
| `chronos2-ft-full-h32` | 32 | 05-29 | 1650 | 300 | 0.28659 | $-0.00281$ | 显著 |

$H=32$ 臂的 `eval_loss` 在 0.3 epoch 即平台（4.96→4.93），且因内部验证目标段
占满 32 天，训练输入被砍到 05-29。**微调增益对训练数据量敏感**：仅仅少一个月
（`-short`）就使 NPL 增益从显著变为不显著。

**③ 微调降低了尾部覆盖。** 两个微调臂的 `cov85_pos` 都低于零样本
（0.770 / 0.776 vs 0.801），`cov50_pos` 同向下降。即微调把分布整体**下移**：
NPL 略有改善，而 $\tau=0.85$ 的名义覆盖变差。这一权衡在决策层被放大，见
`07_decision_layer.md` §6.2 结论 ④。

---

## 6. 结构化 GBDT

### 6.1 LightGBM（主）

`model_id`: `lgbm-q50`, `lgbm-q85`（按 `00_global_conventions.md` §2.1 分别训练）

```python
params = dict(
    objective="quantile", alpha=TAU,       # 0.50 / 0.85
    n_estimators=2000, learning_rate=0.05,
    num_leaves=63, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.1, lambda_l2=1.0, max_bin=255,
    seed=42, deterministic=True, force_row_wise=True, verbose=-1,
)
# 早停在验证窗上进行，测试窗不参与
early_stopping_rounds = 100
```

- 类别特征：先经 `to_vocab()` 归一化，再 `pd.Categorical(..., categories=VOCAB)`（`VOCAB` 前两位为 `__UNK__` / `__MISSING__`），以 `categorical_feature` 传入（`00_global_conventions.md` §7.2）；
- 缺失值：原样传入 NaN，不填充（§8.1）；
- 样本权重：`sample_weight = s_i`，按 §2.2 的 `train_weight_scheme`。

### 6.2 HistGradientBoostingRegressor（备）

`model_id`: `hgb-q50`, `hgb-q85`

```python
HistGradientBoostingRegressor(
    loss="quantile", quantile=TAU,
    max_iter=MAX_ITER,          # 由验证窗显式外循环选定
    learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20,
    l2_regularization=1.0, max_bins=255,
    early_stopping=False,       # 关键：见下
    categorical_features=CAT_MASK, random_state=42,
)
```

**必须设 `early_stopping=False`。** 已核对 sklearn 1.5.1 的实际行为：

- `early_stopping` 默认为 `'auto'`，即**样本数 > 10,000 时自动开启**。本项目全部数据集都远超该阈值，所以这不是假设风险，是默认就会发生的行为；
- 开启后走 `train_test_split(X, y, sample_weight, test_size=validation_fraction, random_state=...)`，即**随机划分**，没有暴露 `shuffle=False` 或任何时序切分选项；
- 默认 `validation_fraction=0.1`、`n_iter_no_change=10`。

危害的精确形态：随机划分会把**同一序列时间上相邻的行**分到两侧。需求有强自相关，$t$ 与 $t+1$ 近似重复样本，于是内部验证集被训练集的近邻污染，早停判据系统性偏乐观 → **停得过晚 → 过拟合**。注意这**不是测试集泄漏**（测试窗从未进入 `fit`），而是训练窗内部的选择信号有偏。

替代方案与实现（用 `warm_start` 避免重复训练）：

```python
m = HistGradientBoostingRegressor(..., early_stopping=False, warm_start=True, max_iter=0)
for it in [200, 400, 800, 1600]:
    m.set_params(max_iter=it); m.fit(X_train, y_train, sample_weight=s)
    val_npl[it] = npl(y_val, m.predict(X_val))   # 在显式验证窗上评估
best_iter = min(val_npl, key=val_npl.get)        # 选定后冻结，测试窗不参与
```

`warm_start=True` 使整条 `max_iter` 曲线只需一次增量训练即可得到，不是 4 次独立拟合。

**边界规则**：若选中值落在网格端点 `1600`，说明网格过窄，必须向上扩展（`{3200, 6400}`）并重跑，同时把该次扩展记入 `hpo_log.csv`；不得直接采用端点值。

该改动使 HGB 与 LightGBM 处在同一选择口径下（后者以显式验证窗做 `early_stopping_rounds`），且用 100% 训练数据而非 90%，在数据利用与选择有效性两方面都优于内置早停。

### 6.3 超参搜索协议

固定网格，**共 12 组**，在验证窗上按 NPL（§4.1）选择，选定后冻结：

| 参数 | 候选 |
| --- | --- |
| `learning_rate` | `{0.03, 0.05, 0.1}` |
| `num_leaves` / `max_leaf_nodes` | `{31, 63}` |
| `min_child_samples` / `min_samples_leaf` | `{20, 60}` |

其余参数固定为 §6.1 的值。**两个 $\tau$ 共用同一组超参**（按二者 NPL 之和选择），避免搜索预算翻倍并防止 $q_{.50}$ 与 $q_{.85}$ 被不同容量的模型产生。搜索日志写入 `artifacts/<ds>/hpo_log.csv`，含全部 12 组的验证 NPL。

---

## 7. 统计基线

全部为点预测，按 `00_global_conventions.md` §6 经残差校准器转为 $(q_{.50},q_{.85})$，`quantile_source="calibrated"`。

| `model_id` | 参数 | 适用 |
| --- | --- | --- |
| `naive-last` | 无 | 全部 |
| `roll-mean-{1,3,6}` | 窗长取自 `configs.<ds>.history.lag_*` | 全部 |
| `seasonal-naive` | 周期：日频 7、周频 52、月频 12 | **仅当 `capabilities.annual_seasonality = true`**：fresh_lt、osa、lokad。fresh_50k、zhao、mendeley 禁用 |
| `croston-sba` | **无可调超参**（见下） | osa、mendeley |
| `croston-opt` | 内部优化 $\alpha$ | osa、mendeley |
| `tsb` | $\alpha_d,\alpha_p\in\{0.1,0.2,0.3\}$（9 组），验证窗选定 | osa、mendeley |
| `ssa` | 窗长 $L$、成分数 $r$ 待来源实现转写 | 仅 FreshRetailNet 来源复现 |

实现来源：**`statsforecast 2.1.1`**（已安装，须钉入 `pyproject.toml`）。安装副作用：`statsmodels` 由 0.14.2 升至 0.14.6，并引入 `fugue/triad/adagio`——须在环境记录中登记。

```python
from statsforecast.models import CrostonSBA, CrostonOptimized, TSB, SeasonalNaive, WindowAverage
```

**已核对的 API 事实（与文档早期草稿不同，以此为准）**：

- `CrostonSBA(alias=..., prediction_intervals=...)` —— **没有 $\alpha$ 参数**。源码中 `_croston_classic` 对需求量与间隔的 SES 均硬编码 $\alpha=0.1$，`_croston_sba` 即在其上乘 $0.95$（恰为 $1-\alpha/2$）。因此 SBA **不参与超参搜索**；若要 $\alpha$ 自适应，改用 `CrostonOptimized`（内部优化），登记为独立 `model_id`。
- `TSB(alpha_d, alpha_p, ...)` —— 两个 $\alpha$ 是**必填位置参数**，无默认值，故 9 组网格成立。
- **零需求回退**：`_croston_classic` 在序列无任何非零需求时（`yd.size == 0`）直接退化为 `_naive`。该回退在 Mendeley 上会大面积触发（§7.1），必须计数并写入 `fallback_level`，不得静默。
- 三个模型均支持 `prediction_intervals=ConformalIntervals(n_windows, h, method)`，可自行产出区间。**本项目不使用它**，统一走 `00_global_conventions.md` §6 的残差校准器——这是为跨模型口径一致而做的刻意选择，须在报告中声明，避免被读作疏漏。

### 7.1 序列密度对统计基线与校准器的实际约束

实测每序列观测数（全量，非某一 split）：

| 数据集 | 序列数 | 周期数 | obs/序列中位数 | 密度 | ≥30 obs 占比 | 训练窗 origin 数 |
| --- | --- | --- | --- | --- | --- | --- |
| fresh_lt | 22,939 | 770 | 236 | 0.31 | 100% | 105 |
| fresh_50k | 50,000 | 90 | 90 | 1.00 | 100% | 7 |
| osa | 100 | 854 | 409 | 0.48 | 100% | 87 |
| lokad | 333 | 52 | 52 | 1.00 | 100% | 30 |
| zhao | 21,468 | 10 | 8 | 0.80 | 0% | 6 |
| mendeley | 50,447 | 326 | **2** | **0.01** | **0.1%** | 35 |

两点直接后果：

1. **校准器 L0（逐序列）层在多数数据集上不可达。** 残差池的大小由**该序列已实现的过去 origin 数**决定，而非观测数。以 `CALIB_N_MIN = 30` 衡量：fresh_50k（7 个训练 origin）、zhao（6 个）**结构性永远达不到 L0**；lokad（30 个）仅在训练窗末勉强达到；mendeley 虽有 35 个 origin，但多数序列的目标不可观测而被排除，有效残差远少于 30。**只有 fresh_lt 与 osa 能常态使用 L0。** 这不是缺陷，是 §6.3 回退层级正常发挥作用；但 `calib_level` 的分布必须逐数据集报告，否则会误以为逐序列校准在全项目生效。
2. **Mendeley 的间歇需求基线大面积退化。** 中位 2 个观测意味着 Croston 的需求间隔几乎无从估计，零需求回退将频繁触发。Mendeley 上的 Croston/TSB 结果须与 `fallback_level != "none"` 的占比一并报告，不得单独作为"间歇需求方法表现"的证据。

---

## 8. 深度基线（FreshRetailNet 来源复现，协议 A）

库：`neuralforecast 3.2.0`（已实装，四个模型均可用）。分位数直接由 `MQLoss(quantiles=[0.5, 0.85])` 产出，`quantile_source="native"`，不经校准器。

统一参数（四模型共用，参数名已实测核对）：

| 参数 | 取值 |
| --- | --- |
| `h` | `configs.<ds>.calendar.horizon_days` |
| `input_size` | `configs.<ds>.history.history_window_days` |
| `loss` | `MQLoss(quantiles=[0.5, 0.85])` |
| `valid_loss` | 同 `loss` |
| `max_steps` | 3000 |
| `val_check_steps` | 100 |
| `early_stop_patience_steps` | 5 |
| `learning_rate` | 1e-3 |
| `batch_size` | 32 |
| `scaler_type` | `"standard"` |
| `random_seed` | 42 |
| `futr_exog_list` | 同 §4.2 的 known covariates |
| `hist_exog_list` | 同 GBDT 的历史特征 |

架构专属参数（`TimesNet.top_k/num_kernels/conv_hidden_size`、`iTransformer.n_heads/e_layers/d_ff`、`DLinear.moving_avg_window`、`TFT.n_head/n_rnn_layers`）**一律保留 neuralforecast 默认值**，并在论文中声明为"库默认，非来源论文调参结果"。若要主张与来源论文数值可比，须先从来源实现转写其架构配置——本轮不做，因此 §8 的结果只作**代表性复现**，不声称复现来源论文的具体数字（与 `ALIBABA_CASE_INSIGHTS_NEXT_STEPS.md` §4.2.2 的定位一致）。

`iTransformer` 需要 `n_series`，取该数据集的序列数；序列数极大时（Fresh 50K 有 50000 条）须先做子采样或分组，方案在运行前登记。

---

## 9. 算力预算（实测外推）

### 9.1 TSFM 推理

按实测吞吐（MPS，ctx365 h7）：base **268 series/s**，small **892 series/s**。

| 数据集 | 序列数 | val origins | test origins | 推理量 | base 预估 |
| --- | --- | --- | --- | --- | --- |
| fresh_lt | 22,939 / 10,000(eval) | 4 | 1 | ~101.8k | ~6.3 min |
| fresh_50k | 50,000 | 2 | 2 | 200k | ~12.4 min |
| zhao | 28,626 | 2 | 2 | ~114.5k | ~7.1 min |
| osa | 100 | 17 | 17 | 3.4k | ~13 s |
| lokad | 333 | 9 | 12 | ~7.0k | ~26 s |
| mendeley | 50,447 | 5 | 6 | ~555k | **~34.5 min** |
| **合计（单臂）** | | | | | **~61 min** |

序列数为实测值。Zhao 的上下文仅约 10 个月（远短于基准的 365 步），实际吞吐会高于表中按 ctx365 外推的保守值。

四个 TSFM 臂的推理合计约 2.5–3 小时（small 臂约为 base 的 1/3）。Mendeley 是最重的一项，因其 50,447 条序列大多极稀疏——若冷启动占比过高，可考虑对零历史序列直接走 §7.3 回退而不调用模型，此优化须登记后统一应用。

### 9.2 TSFM 微调

实测（MPS，bs32，20 步计时外推至 1000 步）：

| checkpoint | 模式 | s/step | 1000 步 |
| --- | --- | --- | --- |
| `amazon/chronos-2` | LoRA | 0.55 | **~9 min** |
| `amazon/chronos-2` | full | 0.70 | **~12 min** |
| `chronos-2-small` | LoRA | 0.17 | ~3 min |
| `chronos-2-small` | full | 0.21 | ~3 min |

6 个数据集配置 × 2 个微调臂 ≈ **2–2.5 小时**（Zhao 的 prequential 额外一次重拟合另计 ~12 min）。

### 9.3 运行时限与失败处理

| 阶段 | 软上限 | 超限处理 |
| --- | --- | --- |
| 单次 TSFM 推理（单臂单数据集） | 60 min | 记 `TIMEOUT_SOFT`，继续但在报告标注 |
| 单次微调 | 45 min | 同上 |
| GBDT 单组超参训练 | 15 min | 同上 |
| 深度基线单模型训练 | 120 min | 超限则降 `max_steps` 并**对全部四个深度模型统一降**，禁止只降其中一个 |

无硬性 120 秒约束（那是原始笔试题的条件，不适用于本研究）。但每次运行的 wall-clock 必须写入 `run_manifest.json`，供成本—收益讨论使用。

---

## 10. `run_manifest.json` 契约

每次模型运行落一个，字段为强制项：

```json
{
  "model_id": "chronos2-ft-lora",
  "dataset": "fresh_lt", "dataset_version": "LT",
  "run_id": "...", "code_version": "...", "config_hash": "...",
  "library": {"name": "chronos-forecasting", "version": "2.3.1"},
  "checkpoint": "amazon/chronos-2", "checkpoint_params": 119500000,
  "quantile_source": "native",
  "hyperparams": {...},
  "feature_list": [...],
  "train_weight_scheme": "w",
  "seed": 42, "device": "mps", "torch_dtype": "float32",
  "train_end": "2025-06-09", "calibration_train_end": null,
  "vocab_frozen_on": "train",
  "wall_clock_sec": 0, "n_train_rows": 0, "n_pred_rows": 0,
  "leakage_assertions_passed": true
}
```

`quantile_source` 取值须与 `00_global_conventions.md` §6.1 的路由表一致：GBDT 与 neuralforecast 记 `"native"`（直接训练 $\tau$），Chronos-2 base 记 `"native"`，`chronos-2-small` 记 `"native_interpolated"`，统计基线记 `"calibrated"`。

---

## 11. 未决事项

| 事项 | 影响 | 处理 |
| --- | --- | --- |
| ~~`statsforecast` 未安装~~ | 已解除 | 已安装 2.1.1 并钉入 `pyproject.toml`；API 事实已核对并更正 §7（`CrostonSBA` 无 $\alpha$ 参数） |
| ~~`use_target_encoding` 泄漏性~~ | 已解除 | 读源码 + 数值验证确认不泄漏，改为退化性问题；主臂用 `False`，`-te` 臂作编码敏感性（§4.3） |
| 深度基线架构参数用库默认 | §8 结果不可声称复现来源论文数字 | 已在 §8 声明定位；如需可比，另立转写任务 |
| `iTransformer` 在 50,000 序列上的 `n_series` 处理 | fresh_50k 可能不可行 | 子采样或分组方案须运行前登记 |
| Moirai / TimesFM 已在本地缓存但未钉版本 | 不进主比较 | 第二阶段可选扩展 |
