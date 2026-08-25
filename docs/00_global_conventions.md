# 全局约定：目标函数、业务权重、评价指标、校准器、编码与缺失值

## 0. 本文件的地位

本文件是 `docs/implementation_specs/01–05` 的**共用底座**。五份数据集规格中所有未定义的跨数据集常量、公式和机制，一律以本文件为准；数据集规格只允许声明本文件明确留给它的**逐数据集取值**（在下文各表中以「逐数据集」标注），不得另行定义机制。

优先级：本文件 > `ALIBABA_CASE_INSIGHTS_NEXT_STEPS.md`（下称主 MD） > 数据集规格。若与主 MD 冲突，以本文件为准，并在主 MD 中记录该次修订。**已知需回写主 MD 的冲突见 §4.4。**

本文件不定义：模型超参数（见待建 `docs/06_model_hyperparams.md`）、切分边界与路径常量（见待建 `configs/*.yaml`）、决策层成本网格与策略公式（见待建 `docs/07_decision_layer.md`）。

---

## 1. 符号与全局常量

| 符号 | 含义 |
| --- | --- |
| $i$ | 一个预测行，即 `(series_id, origin, target_window)` 三元组 |
| $y_i$ | 该行的观测目标值（在缺货截断场景下为下界，见主 MD §5.3） |
| $q_{i,\tau}$ | 模型对该行第 $\tau$ 分位数的预测 |
| $w_i$ | 评价用业务权重（§3） |
| $s_i$ | 训练用样本权重（§2） |
| $\mathcal{S}$ | 一个评价场景或切片（整体、低可用性、冷启动等） |
| $\gamma$ | 上尾分位点，全局默认 $\gamma=0.85$ |

全局常量（**不得逐数据集覆盖**）：

```
TAU_MAIN        = 0.50
TAU_UPPER       = 0.85          # γ
EPS_DENOM       = 1e-12         # NPL 分母数值下限
CALIB_N_MIN     = 30            # 校准器分组最小残差数
QUANTILE_METHOD = "linear"      # numpy.quantile method（type-7），全项目统一
BOOTSTRAP_B     = 1000
BOOTSTRAP_CI    = 0.95
SEED_BASE       = 42
FLOAT_TOL       = 1e-6          # 复现比对容差与哈希舍入位数（6 位小数）
```

---

## 2. 预测层训练目标函数（解除 B1）

### 2.1 目标函数

对分位点 $\tau$，参数为 $\theta$ 的模型在训练集 $\mathcal{T}$ 上最小化加权 pinball loss：

$$
\ell_\tau(y,q)=\max\{\tau\,(y-q),\;(\tau-1)(y-q)\}
$$

$$
\mathcal{L}_\tau(\theta)=\frac{\sum_{i\in\mathcal{T}} s_i\,\ell_\tau\!\big(y_i,\;q_\tau(x_i;\theta)\big)}{\sum_{i\in\mathcal{T}} s_i}
$$

$\tau\in\{0.50,\,0.85\}$ **分别训练两个独立模型**，不使用共享树结构的多输出头。原因：LightGBM/HGB 的 quantile objective 本身是逐 $\tau$ 的；共享结构会把两个分位数的拟合耦合，使「$q_{.85}$ 是否受 $q_{.50}$ 拖累」不可分离。

### 2.2 训练权重 $s_i$ 与评价权重 $w_i$ 的关系（必须显式声明）

**默认 $s_i = w_i$**（训练口径与评价口径对齐）。

唯一登记的备选方案是 $s_i \equiv 1$（训练不加权）。规则：

1. 两种方案在**验证窗**上比较，按 §4 的 NPL 选择，选定后在冻结测试前锁死；
2. 选中的方案必须**对所有可训练模型统一应用**，不允许逐模型挑选；
3. 每个 `run_manifest.json` 必须记录 `train_weight_scheme ∈ {"w", "unit", "n/a"}`。

不可训练模型（季节朴素、滚动历史、Croston-SBA、TSB、零样本 TSFM）按其自身准则拟合，$s_i$ 不适用，记 `"n/a"`。这不构成不公平比较：它们的分位数由 §6 校准器产生，而校准器的残差分位数估计对 $w_i$ 不敏感（校准器不加权，见 §6.2）。

### 2.3 输出后处理（所有模型、所有数据集统一）

按固定顺序执行，**不得跳过、不得改序**：

1. **非有限值替换**：`NaN/±inf` → 记为预测失败，写入失败清单并触发 §7.3 回退；不得静默填 0；
2. **非负截断**：$q \leftarrow \max(q,\,0)$；
3. **整数取整**（仅当 `metric.integer_units = true`，且**仅用于分布重建路径**，见下）；
4. **单调重排**：$\big(\tilde q_{.50},\tilde q_{.85}\big)=\big(\min(q_{.50},q_{.85}),\;\max(q_{.50},q_{.85})\big)$。

#### 第 3 步的适用范围：重建路径必需，评价路径**不做**

整数取整**不是**通用的精度改进。实测（Zhao 日级→周级，`roll-mean-4` + 残差校准器，测试窗）：

| 取整方式 | NPL | $\Delta$NPL vs 不取整 | 判定 |
| --- | --- | --- | --- |
| 不取整 | 0.37570 | — | — |
| 四舍五入 | 0.37565 | $-0.00005$ | 显著但量级可忽略 |
| 向下取整 | 0.37440 | $-0.00130$ | 显著 |
| 向上取整 | 0.37728 | $+0.00158$ | 显著**恶化** |

四舍五入对 NPL 的改善（$-0.00005$）**远小于该 split 的 MDE**，实质为零。向下取整看似更好，但那反映的是**该基线本身系统性偏高**（`cov_50_pos = 0.4845`），不是取整规则的性质——换一个不偏的模型，方向可能反转。**故不得把任何取整方向当作通用优化。**

取整的真实作用在**分布重建**：整数计数目标的分位函数取值本就在整数支撑上，而模型（尤其 TSFM）会在零附近输出 $0.005\!-\!0.275$ 这类小正值，掩盖零原子。实测 Chronos-2 在合成零膨胀序列上的隐含 $P(0)$：

| 处理 | 隐含 $P(0)$（真值 0.30 / 0.50 / 0.68 / 0.85） | 平均绝对误差 |
| --- | --- | --- |
| 仅非负截断 | 0.05 / 0.05 / 0.15 / 0.40 | **0.420** |
| 截断 + 四舍五入 | 0.15 / 0.50 / 0.70 / 0.80 | **0.055** |
| 再拼接 context 经验零率 | 0.25 / 0.50 / 0.70 / 0.85 | **0.017** |

**非负截断对零原子完全无效**（截断前后隐含 $P(0)$ 一致）——负值只出现在最低几档，而那几档 $\tau$ 本就很小。取整才是主要修复。

**但外部取整不是分布重建路径的必要步骤。** `quantile_grid_to_pmf` 自 2026-08-23
起以 `support="integer"` 显式承担整数支撑的落点语义（$q(\tau)=v \Rightarrow
F(\lfloor v \rfloor) \ge \tau$），零原子在**重建内部**即已恢复。Zhao 日级→周级
实测（Chronos-2 零样本，验证窗，`vmax=300`，midpoint）：

| `QuantileRepair.round_integer` | NPL |
| --- | --- |
| `True` | 0.32107 |
| `False` | 0.32525 |

差值 $-0.004$，与该 split 的 MDE 同量级。**修正前**该差值是 $-0.040$——那不是
取整的收益，是重建函数当时隐式**上取整**（`v <= g`）造成每期 $+0.5$ 偏移，
外部 `round` 在抵消它。同一缺陷使显式 `ceil` 的 $\Delta$NPL 恰为 $+0.00000$，
这正是暴露该 bug 的证据。

**因此规则是**：

| 路径 | 是否取整 |
| --- | --- |
| 送入 `f2d.aggregation` 做分布重建/卷积 | 由 `support` 参数承担；外部取整**可选**，但须对所有被比较模型一致 |
| 直接进入 §4 的 NPL 评价 | **不做**（改善可忽略，且取整方向的收益依赖模型偏倚，会引入不可比因素） |

`support` 的取值由目标性质决定，**不是调参项**：整数计数目标一律
`"integer"`；`"continuous"` 仅用于连续目标（用就近取整，使离散化无偏，
否则每期下移 0.5）。两条路径的取整状态与 `support` 取值必须记入
`run_manifest.json` 的 `rounding_applied` / `pmf_support`。

「拼接 context 经验零率」（`f2d.models.chronos.QuantileRepair(splice_zero_atom=True)`）默认**关闭**：它把零发生率从模型手中拿走交给经验估计，改变了被检验的命题。若启用，必须对所有模型一致启用并在报告中声明。

第 3 步采用**排序重排**而非单侧压平（即不采用「若 $q_{.50}>q_{.85}$ 则令 $q_{.50}=q_{.85}$」）。理由：排序重排是 Chernozhukov 等人的 quantile rearrangement，对任何真实单调分位曲线不会增加损失，且对两个 $\tau$ 项对称；单侧压平会系统性地偏袒其中一项，在交叉率不为零时污染 $q_{.50}$ 与 $q_{.85}$ 的相对比较。

交叉率 `crossing_rate` 必须逐模型逐切片报告，它本身是模型质量诊断。

---

## 3. 业务权重 $w_i$ 的逐数据集实例化（解除 B2）

### 3.1 权重取值

沿用三档 $w\in\{1.0,\,1.5,\,2.5\}$，对应低/中/高业务价值档，与售后备件题的 `{standard, important, critical}` 同尺度，使跨数据集的 NPL 量级可比。

### 3.2 分档规则

| 数据集 | 价值字段 | series 级聚合 | 分档 | 结果 |
| --- | --- | --- | --- | --- |
| FreshRetailNet-50K / LT | **无成本字段** | — | 不分档 | $w_i\equiv 1.0$ |
| Zhao SKU×月 | `sup-0002` 的 `unit_cost` | 训练窗内 `order_date` 落在训练窗的订单 `unit_cost` 中位数 | 训练窗三分位 | 1.0 / 1.5 / 2.5 |
| OSA-Data | **无成本字段** | — | 不分档 | $w_i\equiv 1.0$ |
| Mendeley | 库存表 `Stock Unit Cost Price` | 训练窗内该 `(Store, Product No)` 的中位数 | 训练窗三分位 | 1.0 / 1.5 / 2.5 |
| Lokad | `Catalog.tsv` 的 `BuyPrice` | 静态，按 `Ref` | 训练窗三分位 | 1.0 / 1.5 / 2.5 |

三分位切点（bin edges）**只在训练窗计算一次**，随后冻结，原样应用到验证窗与测试窗。

### 3.3 硬性约束

- **禁止**在验证/测试窗重算切点；
- **禁止**用 `SellPrice`、毛利或销量派生权重（这会把标签信息带入权重）；
- 冷启动 series 或成本字段缺失 → $w_i = 1.0$，并在预测表 `weight_source` 列记为 `"default_missing_cost"`，其余记 `"tercile"` 或 `"flat"`；
- FreshRetailNet 与 OSA 的 $w\equiv1$ 是**数据能力的事实**，不是遗漏。二者的 NPL 因此退化为无权重形式，报告中必须注明，不得与 Zhao/Mendeley/Lokad 的加权 NPL 并列为同一排名。

### 3.4 登记的敏感性方案（非默认，需显式开启）

OSA 可按 `vendor_leadtime_info.csv` 的 `LEAD_TIME_ON_ORDER` 三分位分档（提前期越长、缺货代价越高）。该方案只作为敏感性报告，不进入主结果，且必须与 $w\equiv1$ 的主结果并列呈现。

---

## 4. 主指标：加权归一化双分位 pinball loss（解除 B3）

### 4.1 定义

对场景/切片 $\mathcal{S}$：

$$
\operatorname{NPL}_{.50,\gamma}(\mathcal{S})=
\frac{\displaystyle\sum_{i\in\mathcal{S}} w_i\Big[\ell_{.50}\big(y_i,q_{i,.50}\big)+\ell_{\gamma}\big(y_i,q_{i,\gamma}\big)\Big]}
{\displaystyle\max\Big(2\sum_{i\in\mathcal{S}} w_i\,\max\big(y_i,\;y_{\min}\big),\;\;\varepsilon\Big)}
$$

其中 $\gamma=0.85$，$\varepsilon=10^{-12}$，$y_{\min}$ 为逐数据集常量（§4.3）。

这与售后备件题 judge 的形式一致：**地板 $\max(y_i,y_{\min})$ 在求和内部、逐样本生效**。

### 4.2 为什么必须是逐样本地板

主 MD §5.3 原式的分母是 $2\sum_i w_i\max(\bar y_w,\varepsilon)$，其中 $\bar y_w$ 是场景**标量**加权均值。展开后该分母在 $\bar y_w>\varepsilon$ 时等于 $2\sum_i w_i y_i$，对单行零需求毫无保护，只在整个场景全零时才触发 $\varepsilon$。

这在本项目是**实际会发生的失效**，不是理论顾虑：

- Mendeley 与 OSA 明确要跑 Croston-SBA / TSB（主 MD §4.2.1），间歇需求意味着大量 $y_i=0$；
- 主 MD §5.4 要求按低可用性、冷启动等切片**分别**报告 NPL，某些切片可能接近全零，此时旧式分母趋于 0，指标爆炸或需要临时决策。

逐样本地板下，全零切片的分母为 $2\sum_i w_i\,y_{\min}>0$，指标有限且有意义。

### 4.3 $y_{\min}$ 的逐数据集取值与确定规则

$y_{\min}$ 是**目标变量量纲下的最小业务单位**，不可跨数据集照搬。已实测：FreshRetailNet 的 `sale_amount` 为连续量（样本中位数 0.7，均值 1.10，91% 为非整数），若沿用 $y_{\min}=1$ 将使超过半数训练行被地板值支配，NPL 退化为几乎与预测无关的常数。

确定规则分**两条路径**，按目标量纲二选一（确定性、仅用训练窗、禁止调优）：

**路径 A —— 整数计数目标**（Zhao 月销量、OSA 日销量、Mendeley 净销量、Lokad 周销量）：

> $y_{\min}=1.0$，即**一个最小业务单位**。**不适用 10% 上限。**

理由：整数件的「1」有业务含义，不是任意尺度。若目标本身零膨胀（Zhao 零占比 0.428、OSA 0.485），地板在这些行上 bind 正是它存在的目的——防止分母塌陷。此时 bind_rate 高是数据性质，不是配置错误。仍须记录 bind_rate 供解释。

**路径 B —— 连续量目标**（FreshRetailNet `sale_amount`）：

> 在阶梯 $\{1.0,\,0.5,\,0.1,\,0.05,\,0.01\}$ 中，取**满足训练窗地板触发率 $\le 10\%$ 的最大值**。
> 地板触发率 $\text{bind\_rate}=\dfrac{|\{i\in\mathcal{T}: y_i<y_{\min}\}|}{|\mathcal{T}|}$。

理由：连续量的地板尺度本无业务锚点，必须由数据定，10% 上限防止地板主导分母。

**已实测的失效案例**：对 OSA 周级目标误用路径 B，阶梯全程无解（零占比 0.2258 > 0.10），函数一路跌到 $y_{\min}=0.01$ 并返回违规的 bind_rate。这不是数据异常，是**路径选错**。实现须按目标量纲显式选路，不得对整数目标调用阶梯规则。

**$y_{\min}$ 依赖目标定义，不只依赖数据集。** 同一数据集的日级目标与 7 日求和目标量纲不同，须各算一次：FreshRetailNet 日级 $y_{\min}=0.1$（bind 0.056），7 日求和 $y_{\min}=1.0$（bind 0.020）。切换目标粒度时必须重算，不可沿用。

| 数据集 | 目标量纲 | $y_{\min}$ | 状态 |
| --- | --- | --- | --- |
| Zhao SKU×月 | 月销量，整数件 | 1.0 | 已定 |
| OSA-Data | 日销量，整数件 | 1.0 | 已定 |
| Mendeley | 日净销量，整数件 | 1.0 | 已定 |
| Lokad | 周销量，整数件 | 1.0 | 已定 |
| FreshRetailNet | `sale_amount`，连续 | **由 §4.3 规则在原始审计阶段确定** | 待 `raw_audit.json` 输出 bind_rate 后按阶梯选定 |

该常量一经选定即写入 `configs/<dataset>.yaml`，冻结，**不得在看到任何测试窗结果后修改**。每次指标计算必须把实际使用的 $y_{\min}$ 与 bind_rate 一并写入 `metrics_*.csv`。

### 4.4 使用约束

1. **NPL 不可分解。** 每个切片用**该切片自己的分母**计算。因此整体 NPL $\ne$ 各切片 NPL 的加权平均，禁止用切片值平均出总值，也禁止反向拆解。
2. **跨数据集不可直接排名。** $w_i$ 方案与 $y_{\min}$ 不同（§3.2、§4.3），NPL 只在同一数据集同一场景内比较模型。
3. **需回写主 MD**：§5.3 的 NPL 公式须替换为本节 §4.1 式，并补上 $\varepsilon=10^{-12}$、$y_{\min}$ 表与「不可分解」声明。此为已知冲突，未回写前主 MD §5.3 视为失效。

---

## 5. 配套指标定义（解除 C4，取代五份规格引用的 `metrics_definition.md`）

所有指标均在**同一预测表、同一切片掩码**上计算，并与样本数 $n$ 一同报告。

### 5.1 经验覆盖率

$$
\operatorname{cov}_\tau(\mathcal{S})=\frac{1}{|\mathcal{S}|}\sum_{i\in\mathcal{S}}\mathbb{1}\big[y_i\le q_{i,\tau}\big],
\qquad
\operatorname{gap}_\tau=\operatorname{cov}_\tau-\tau
$$

默认**不加权**（覆盖率是分布性质）。同时输出加权版本 `cov_tau_w` $=\sum w_i\mathbb{1}[\cdot]/\sum w_i$。两列都要，不可只报其一。

#### 零膨胀下覆盖率退化，必须同时报告正需求子集

§2.3 的后处理保证 $q_{i,\tau}\ge 0$，因此**任何 $y_i=0$ 的行都自动满足 $y_i\le q_{i,\tau}$**，无论模型好坏。整体覆盖率于是被零占比机械抬高：

$$\operatorname{cov}_\tau=\pi_0\cdot 1+(1-\pi_0)\cdot\operatorname{cov}_\tau^{+},\qquad \pi_0=\Pr(y=0)$$

已在 Zhao 实测验证：测试窗 $\pi_0=0.428$，$y=0$ 子集的 $\operatorname{cov}_{.50}=\operatorname{cov}_{.85}=1.000$，$y>0$ 子集的 $\operatorname{cov}_{.50}=0.568$、$\operatorname{cov}_{.85}=0.879$；整体 $0.428+0.572\times0.568=0.753$，与直接计算的 $0.7528$ 吻合。此时整体覆盖率无法区分校准良好与校准糟糕的模型。

因此**必须同时输出**：

| 列 | 定义 |
| --- | --- |
| `cov_50` / `cov_85` | 全体样本，供与名义 $\tau$ 对照 |
| `cov_50_pos` / `cov_85_pos` | **仅 $y_i>0$ 子集**，零膨胀下唯一有信息的覆盖率 |
| `zero_share` | $\pi_0$，用于解释前两者的差距 |

校准结论以 `cov_*_pos` 为准；`zero_share > 0.10` 时，禁止单独引用 `cov_50`/`cov_85` 主张校准质量。Zhao（0.43）、Mendeley、OSA 均属此列。

#### 零占比超过 $\tau$ 时，该分位数的接口彻底退化

若 $\pi_0=\Pr(y=0)>\tau$，则真实的 $\tau$ 分位数**恒等于 0**，恒预测 0 就是该损失下的**贝叶斯最优解**。此时 $q_\tau$ 不携带任何关于需求的信息，任何模型比较都是在比较"谁更接近 0"。

**已实测证实**（Mendeley 生命期≥100天子总体，周级 $\pi_0=0.906$）：加入 `always-zero` 退化对照后，它在验证窗 **NPL 最低（0.0694）**，优于 `roll-mean-4`(0.0759)、`naive-last`(0.0781)、`croston-sba`(0.1186)；配对 bootstrap 显示三个真实模型都**显著劣于** always-zero。而 `always-zero` 的 `cov_50_pos = 0.0000`——它从不覆盖任何正需求，却赢得指标。这不是指标漏洞，是 $\pi_0>\tau$ 下的正确答案。

**逐数据集判定**（周级目标）：

| 数据集 | $\pi_0$ | $q_{.50}$ 有信息？ | $q_{.85}$ 有信息？ |
| --- | --- | --- | --- |
| fresh_lt | 0.011 | ✅ | ✅ |
| zhao | 0.428 | ⚠️ 临界 | ✅ |
| osa | 0.485 | ⚠️ 临界（0.485 < 0.50 仅差 0.015） | ✅ |
| **mendeley** | **0.906** | ❌ **恒为 0** | ❌ **恒为 0** |

**强制规则**：

1. 每个数据集每个切片必须报告 $\pi_0$，并与 $\tau$ 对照；
2. $\pi_0>\tau$ 时，该 $\tau$ 的比较结果**不得作为模型能力证据**，须标 `QUANTILE_DEGENERATE`；
3. $\pi_0$ 接近 $\tau$（差值 < 0.05）时标 `QUANTILE_MARGINAL`，结论须附此标记——OSA 的 $q_{.50}$ 即属此列；
4. **每次模型比较必须包含 `always-zero` 退化对照。** 它成本为零，却是检出接口退化的唯一可靠手段。若任何真实模型未能显著优于它，该比较无效。

若数据集在目标 $\tau$ 上退化，可选做法是改用 $\tau>\pi_0$ 的分位数（Mendeley 需 $q_{.95}$ 以上），或改变预测目标（如"给定发生时的需求量"与"距下次需求的时间"分开建模）。任一改动都须登记为新的场景，不得与主接口结果混report。

### 5.2 校准曲线

主比较只产出两个分位数，**无法**由此画出校准曲线。因此：

- 校准曲线是**补充诊断**，不进入主结果；
- 分位点网格固定为 $\mathcal{T}_{\text{cal}}=\{0.05,0.10,\dots,0.95\}$（步长 0.05，共 19 点）；
- 仅对能低成本产出该网格的模型运行：GBDT（逐 $\tau$ 重训）、输出样本或完整分布的 TSFM、经 §6 校准器的点预测基线；
- 标量摘要：$\operatorname{CalErr}=\frac{1}{|\mathcal{T}_{\text{cal}}|}\sum_{\tau}\big|\operatorname{cov}_\tau-\tau\big|$；
- 输出 `calibration_curve.csv`：`dataset, model_id, split, slice_id, tau, cov_tau, n`。

### 5.3 点预测指标

分位数模型的点预测统一取 $\hat y_i = q_{i,.50}$（显式声明，不使用均值）。

$$
\operatorname{WAPE}(\mathcal{S})=\frac{\sum_{i\in\mathcal S}\big|y_i-\hat y_i\big|}{\max\big(\sum_{i\in\mathcal S}|y_i|,\;\varepsilon\big)}
\qquad
\operatorname{WPE}(\mathcal{S})=\frac{\sum_{i\in\mathcal S}\big(\hat y_i-y_i\big)}{\max\big(\sum_{i\in\mathcal S}y_i,\;\varepsilon\big)}
$$

WPE 的符号约定：**正值 = 高估**。

该约定已与 FreshRetailNet 来源论文核对，**一致**（论文 eq. 4–5）：

$$
\operatorname{WAPE}=\frac{\sum_t\big|d_t-y_t\big|}{\sum_t y_t},
\qquad
\operatorname{WPE}=\frac{\sum_t\big(d_t-y_t\big)}{\sum_t y_t}
$$

其中 $d_t$ 为模型输出（恢复需求或预测），$y_t$ 为真值。论文正文以 WPE $=-7.37\%$ 描述「systematic under-estimation」、以 $+2.58\%$ 描述其改善，确认负值 = 低估、正值 = 高估。

两点与本项目 §5.3 式的细微差异，按下述处理：

1. 来源 WAPE 分母是 $\sum_t y_t$（不取绝对值）。本项目式取 $\sum|y_i|$，在 $y\ge 0$ 的所有数据集上等价；Mendeley 允许负净销量（退货），故 Mendeley 的 WAPE **以本项目式为准**，并在 `metrics_notes.md` 注明该处不与来源可比。
2. 来源指标的**计算范围**是其任务定义的一部分，不可省略：MNAR 恢复实验只在合成删失区间上计算（以非缺货期为真值）；预测实验「exclusively during operational periods without stockouts」，即只在无缺货营业期计算。协议 A 必须复用该资格掩码并保存，协议 B 的 NPL 则在全部合格行上计算。二者的 WAPE 数值因此不可直接并列。

### 5.4 $\rho_{DS}$（Decoupling Score，需求—缺货解耦分数）

**状态：已转写（来源：FreshRetailNet-50K 论文 §5.1 eq. 6，`Reference/FreshRetailNet-50K.pdf`）。**

#### 5.4.1 来源定义（逐项转写，不得改写）

$$
\rho_{DS}=\sum_{i\in P} w_i\cdot \operatorname{Pearson}\big(SR_i,\;d_i\big),
\qquad
w_i=\frac{\mu_i}{\sum_{j\in P}\mu_j}
$$

| 符号 | 来源原文含义 |
| --- | --- |
| $P$ | 全部 store–product 对（"all store-product pairs"） |
| $SR_i$ | 该对的 stockout ratio（缺货率） |
| $d_i$ | 该对的 recovered demand（恢复后需求） |
| $\mu_i$ | 该对的 mean sales（平均销量） |
| $w_i$ | 按 $\mu_i$ 归一化的权重 |

**注意：此处的 $w_i$ 是来源论文 $\rho_{DS}$ 内部的销量权重，与本文件 §3 的业务权重 $w_i$ 是不同的量。** 实现中前者命名 `rho_ds_weight`，后者 `w_i`，禁止复用同一变量名。

#### 5.4.2 方向性

**$|\rho_{DS}|$ 越接近 0 越好。** 论文明确以 TimesNet 的 $0.07$「vs. raw sales' $-0.57$」说明「successful elimination of spurious demand and stockout linkages」，并指出仍保持显著负相关的模型「recovered demand is still under the implicit influence of supply constraints」。

来源锚点值（论文 Table 2，用于实现自检；本项目复现值与之量级差异过大即视为实现错误）：

| | Raw Sale | TimesNet | ImputeFormer | SAITS | iTransformer | GPVAE | CSDI | DLinear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\rho_{DS}$ | −0.57 | **0.07** | −0.26 | −0.50 | −0.22 | −0.15 | −0.27 | −0.16 |

报告时**同时输出** `rho_DS`（带符号）与 `abs_rho_DS`；排名依据后者。Raw Sale 在来源 Table 2 中只有 $\rho_{DS}$、无 WAPE/WPE（MNAR 实验中原始销量即被删失的输入，无「恢复」可评），本项目须保持同一处理，不得为 Raw 基线补算 WAPE/WPE 后与其他模型并列。

#### 5.4.3 来源未定义、由本项目声明的实现细节

以下四项论文**没有**给出，属本项目决策，必须在 `metrics_notes.md` 中标注为「our choice, not source」，且一经固定不得因结果调整：

1. **Pearson 的时间粒度**：$\operatorname{Pearson}(SR_i,d_i)$ 在每个 store–product 对**内部沿时间**计算。本项目固定为**日级**序列（$SR_i$ 天然是日聚合量），即对每个对，取其日缺货率序列与日恢复需求序列的 Pearson 相关。
2. **$SR_i$ 的构造**：$SR_{i,t}=\dfrac{\text{该对第 }t\text{ 日营业时段内的缺货小时数}}{\text{该日营业小时数}}$，营业时段取 6:00–22:00（共 16 小时），分子直接使用 `stock_hour6_22_cnt`（见 §5.4.4 的极性说明）。
3. **$\mu_i$ 的口径**：取该对在**评价窗内的观测销量均值**（非恢复需求均值），与论文 "mean sales" 的字面一致。
4. **退化对的处理**：$SR_i$ 或 $d_i$ 在评价窗内方差为零时 Pearson 无定义（例如从不缺货的对）。该对**排除**出求和，并在剩余对上**重新归一化** $w_i$；被排除的对数与其销量占比必须写入 `recovery_metrics.csv` 的 `rho_ds_excluded_pairs`、`rho_ds_excluded_mu_share`。禁止以 0 填充无定义的 Pearson（那会把「从不缺货」错误地记为「完美解耦」）。

#### 5.4.4 实现前必须修正的数据极性（已实测）

来源论文 eq. 1 定义 $s_t=1$ 表示**有货**、$s_t=0$ 表示缺货，并据此给出恢复约束 eq. 2：

$$
d = y\odot s + \hat d\odot(1-s)
$$

即：**有货小时保留观测销量，仅在缺货小时用模型估计填充。** 这是来源任务的硬约束，所有恢复模型的输出必须满足，spec 01 步骤 6–7 须据此校验。

但数据集列 `hours_stock_status` 的极性与论文的 $s$ **相反**。实测（首个 shard 前 5000 行）：

- `hours_stock_status == 1` 的小时平均销量 $0.0047$；`== 0` 的小时平均销量 $0.0554$，相差一个数量级；
- `stock_hour6_22_cnt` 与 `sum(hours_stock_status[6:22])` 的相关系数为 $1.000$，而该列在 spec 01 中已被描述为「缺货小时数」。

结论：**`hours_stock_status == 1` 表示缺货**。因此实现中必须取

```
s_paper       = 1 - hours_stock_status     # 论文口径：1 = 有货
is_stockout   = hours_stock_status         # 数据集口径：1 = 缺货
```

**这修正了 spec 01 步骤 4 的一处错误**（原文写 `is_available = hours_stock_status[hour]`，极性反了）。该错误若不修正，会同时反转缺货掩码、低可用性切片定义、eq. 2 的填充位置和 $\rho_{DS}$ 的符号。极性断言必须作为 spec 01 原始审计的一条 assertion 固化：`mean(hours_sale[hours_stock_status==1]) < mean(hours_sale[hours_stock_status==0])`。

### 5.5 零分母规则（全项目统一）

任一指标的分母 $<\varepsilon$ 时：

- 指标值写 `NaN`（**不写 0，不丢行**）；
- 同行写 `reason_code = "ZERO_DENOM"`；
- 同行仍写 `n`、`n_positive`（$y_i>0$ 的样本数）与该切片的 $\sum w_i$。

空切片（$n=0$）写 `reason_code = "EMPTY_SLICE"`，同样保留行，不得省略。

---

## 6. 残差分位数校准器（解除 B4）

### 6.1 适用范围

| 模型类别 | 分位数来源 |
| --- | --- |
| LightGBM / HGB | 直接按 §2.1 训练 $\tau=.50,.85$，**不经校准器** |
| 季节朴素、上一可观测月、1/3/6 月滚动历史、移动平均 | 点预测 → 校准器 |
| Croston-SBA、TSB | 点预测 → 校准器 |
| Lokad 既有 `WeeklyForecast`、Workshop #4 低维模型点预测 | 点预测 → 校准器 |
| TSFM 输出样本或完整分布 | 直接取经验分位数（`QUANTILE_METHOD`），**不经校准器** |
| TSFM 仅输出点预测 | 点预测 → 校准器 |

同一模型不得在不同数据集间切换路径；路径写入 `run_manifest.json` 的 `quantile_source ∈ {"native","empirical","calibrated"}`。

### 6.2 算法

**残差定义（加性）**：对已实现的历史预测行 $j$，$r_j = y_j - \hat y_j$。

采用加性而非乘性残差：间歇需求下 $\hat y_j=0$ 频繁出现，乘性残差 $y_j/\hat y_j$ 无定义；加性残差的尺度适配由 §6.3 的分组承担。此为**已登记的拒绝项**，不得在实现中临时改为乘性。

**估计式**：

$$
q_{i,\tau}=\hat y_i + \hat r_{\,g(i),\,\tau},
\qquad
\hat r_{g,\tau}=\operatorname{Quantile}_\tau\big(\{r_j : j\in \mathcal{R}_g(\text{origin}_i)\}\big)
$$

`Quantile` 一律使用 `numpy.quantile(..., method="linear")`。校准器**不加权**（$w_i$ 不进入残差分位数估计），保证 §2.2 中不可训练模型与可训练模型的可比性。

**残差池 $\mathcal{R}_g(\text{origin})$ 的三条硬约束**：

1. **时点隔离**：只纳入 `target_end < origin` 的残差行。这是可断言的泄漏检查，必须在代码中 assert，不可仅靠约定；
2. **视界匹配**：只纳入与当前预测**相同预测视界**的残差（7 日对 7 日、1 月对 1 月）。跨视界混用被禁止；
3. **扩展窗**：采用 expanding window（纳入所有满足前两条的历史残差），在**每个 origin 重新拟合**。不采用固定长度滚动窗，理由是 Zhao 仅 10 个月，丢弃历史的代价过高。

### 6.3 分组回退层级

$g(i)$ 按以下层级取**满足 $n\ge$ `CALIB_N_MIN` (=30) 的最具体一层**：

| 层 | 分组键 | 逐数据集的 category 列 |
| --- | --- | --- |
| L0 | `series_id` | — |
| L1 | `series_id` 所属类别 | Fresh: `third_category_id`；Zhao: `category`；OSA: `product_category`；Mendeley: `Product Category`；Lokad: `Category` |
| L2 | 全局（该数据集该 split 内） | — |

L2 若仍 $n<30$，则该次运行**失败退出**（不是回退到常数），因为这意味着校准窗过短，属于切分配置错误。

实际使用的层级逐行写入预测表 `calib_level ∈ {"L0","L1","L2"}`，供事后审计。

**L0 的可达性逐数据集不同，须预期而非视为异常。** 残差池大小由该序列**已实现的过去 origin 数**决定，不是观测数。实测（详见 `docs/06_model_hyperparams.md` §7.1）：fresh_50k（7 个训练 origin）与 zhao（6 个）**结构性永远达不到 $n\ge30$**，恒走 L1/L2；lokad（30 个）仅训练窗末勉强达到；mendeley 虽有 35 个 origin，但多数序列目标不可观测被排除，有效残差远少于 30。**仅 fresh_lt（105）与 osa（87）能常态使用 L0。**

因此 `calib_level` 的分布必须**逐数据集逐模型报告**，否则会误以为逐序列校准在全项目生效。这不改变 §6.3 的机制，只是要求把回退层级当作结果的一部分呈现。

### 6.4 退化与后处理

- 组内残差全等（如恒零序列且预测恒零）→ $\hat r_{g,.50}=\hat r_{g,.85}$，产出 $q_{.50}=q_{.85}$。这是**合法结果**，不得视为错误或人为加宽；
- 校准器输出仍须走 §2.3 的完整后处理（非有限值 → 非负截断 → 单调重排）。

---

## 7. 类别编码与冷启动回退（解除 C5）

### 7.1 词表冻结与两个保留槽

所有类别特征使用**仅由训练窗构建的冻结词表**，前两个索引为保留槽：

```
VOCAB = ["__UNK__", "__MISSING__"] + sorted(训练窗出现过的取值)
索引:      0             1              2, 3, 4, ...
```

**`__UNK__` 与 `__MISSING__` 必须分开，禁止合并为同一个码。** 二者是不同的信息：

| 槽 | 含义 | 诊断价值 |
| --- | --- | --- |
| `__UNK__` (0) | 字段**有记录**，但取值未在训练窗出现 | 冷启动信号；`is_cold_start` 切片的直接依据 |
| `__MISSING__` (1) | 字段**无记录** | 数据质量信号，与冷启动无关 |

合并会让 §5.4/§9.4 的冷启动切片对比出现无法归因的混淆源。词表连同其构建截止时点写入 `artifacts/<ds>/vocab.json`。

### 7.2 编码机制：由本项目做映射，不依赖任何库的 unknown 分支

**归一化函数（唯一实现，三个模型族共用）**：

```python
def to_vocab(s: pd.Series, vocab: list[str]) -> np.ndarray:
    real = vocab[2:]                      # 真实取值，不含两个保留槽
    return np.where(s.isna(), "__MISSING__",
           np.where(s.isin(real), s, "__UNK__"))
```

先归一化，**再**交给各库编码。这样任何库的 unknown/sentinel 分支都不会被触发。

| 模型 | 机制 |
| --- | --- |
| LightGBM | `pd.Categorical(to_vocab(x), categories=VOCAB)`，以 `categorical_feature` 传入 |
| HGB | `OrdinalEncoder(categories=[np.array(VOCAB, dtype=object)])`，输入已是归一化值，**不设 `handle_unknown`/`unknown_value`** |
| Chronos-2 | `pd.Categorical(to_vocab(x), categories=VOCAB)`，past 与 future 两个 DataFrame 使用**同一** `categories`；`use_target_encoding=False` |

**已实测**（`['A', 'Z'(未见), None(缺失), 'C']`，`VOCAB=['__UNK__','__MISSING__','A','B','C']`）：三条路径产出**完全相同**的编码 `[2, 0, 1, 4]`，Chronos-2 的 future 侧得到 `[0, 1]` 且**不出现 NaN sentinel**。

两条必须避开的陷阱：

1. **`OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=0)` 会直接抛 `ValueError`** —— sklearn 要求 `unknown_value` 落在 `[0, n_categories)` 之外，而 `__UNK__` 在词表内即占用了 0。本文件早期版本写的就是这个形式，已作废。
2. **Chronos-2 的 sentinel→NaN 分支看的是 pandas `categories` 声明，不是数据中实际出现过的取值。** 只要 past 与 future 的 `CategoricalDtype.categories` 都设为完整冻结词表，该分支不可达。反之若只用数据推断 categories，未见值会变成 NaN，与 GBDT 侧的行为分叉。

**硬禁止：跨行 label-mean 编码。** 即以类别为分组、在**多行样本**上对标签求均值或其他统计量作为特征，且其拟合范围包含被编码行自身、或越过 split 边界。这是经典 target encoding 的泄漏形态；标准解药（K-fold out-of-fold）本项目一律不采用，直接禁用。

**禁令边界**（防止误伤）。判定标准是两条**同时**成立：① 跨行汇总了标签；② 拟合范围越过预测时点或 split 边界。以下三者不在射程内：

1. **序列内、仅用 context 的目标派生编码。** 例如 Chronos-2 `preprocess` 的 `use_target_encoding`：统计逐序列进行、只取 context 段目标，预测窗目标不在输入数组中。已通过读源码与数值验证确认不泄漏。本项目出于**退化性**关闭它（静态类别会塌为序列均值），那是建模选择，不是合规要求 —— 详见 `docs/06_model_hyperparams.md` §4.3。
2. 目标序列自身的滞后/滚动特征（其可用性由 §8.2 与各 spec 的时点隔离规则单独约束）。
3. **频次编码**：只用训练窗的出现次数、不触及标签者，属登记备选，需显式开启并对所有模型统一应用。

### 7.3 冷启动与预测失败的回退

正常情况下，冷启动 series 仍有静态特征，GBDT/TSFM 能给出预测，**无需回退**。回退只在两种情形触发：

1. 逐序列统计基线（Croston、季节朴素等）在零历史序列上无法产出；
2. §2.3 第 1 步检出非有限值。

回退按以下顺序取**第一个可用**的值：

| 级别 | 回退值 |
| --- | --- |
| F1 | 该 series 所属类别在训练窗内目标值的经验 $\tau$ 分位数 |
| F2 | 训练窗全局目标值的经验 $\tau$ 分位数 |

每次回退在预测表写 `fallback_level ∈ {"none","F1","F2"}` 与 `fallback_reason`。**禁止静默回退**；`fallback_level != "none"` 的行数必须出现在该次运行的验收 JSON 中。

---

## 8. 缺失值策略（解除 C6）

### 8.1 数值特征

- LightGBM / HGB：**不做任何填充**，原样传入 NaN（二者均原生处理缺失并将其作为可分裂方向）。禁止用 0、均值或中位数预填；
- 不默认创建 `{col}_is_missing` 指示列（GBDT 已能在 NaN 上分裂）。创建指示列属登记备选，需对**所有模型统一**开启或关闭；
- 任何 imputer/scaler/encoder 只允许在训练窗拟合，禁止在含验证/测试的数据上 `fit`。此为可断言项。

### 8.2 目标序列的日历缺口

面板中某 `(series, period)` 无交易记录时：

- **仅当**该数据集的原始审计确证「无记录 = 该期需求为零」时补 0（例如 Zhao：该 SKU×月存在于月初库存快照表，见 spec 02 步骤 2）；
- 否则保持缺失，并将以该期为目标的 origin **排除**出预测集，写 `exclusion_reason = "TARGET_UNOBSERVED"`；
- 排除行数必须出现在验收 JSON，禁止静默丢弃。

### 8.3 不接受 NaN 的模型

TSFM 与统计基线要求连续历史。其输入序列的缺口按 §8.2 的同一判定处理：可补零的补零，不可补零的则该序列在该 origin 上不参与，并记录原因。**禁止**为迁就模型而对不可补零的缺口填 0。

---

## 9. 可复现性与验收契约（解除 C7 / C8）

### 9.1 随机种子

基础种子 `SEED_BASE = 42`。每次运行在 `run_manifest.json` 记录实际使用的整数种子。需要方差报告时使用 `{42, 43, 44}` 三个种子，报告中位数与极差，不得只报最好的一次。

### 9.2 浮点比对与复现哈希

预测表的复现哈希定义为：按主键升序排序 → 列顺序按 §9.4 固定 → 所有浮点列以 `"%.6f"` 格式化 → 写为 UTF-8、`\n` 换行、无索引的 CSV → 取 `sha256`。

两次运行「一致」的判据：哈希相同，或所有浮点列的逐元素绝对差 $\le$ `FLOAT_TOL` $=10^{-6}$。这使五份规格中「同一 seed 重跑哈希一致」成为可执行断言。

### 9.3 验收契约

每个步骤输出 `artifacts/<dataset>/checks/<step_id>.json`：

```json
{
  "step_id": "...", "dataset": "...", "status": "PASS|FAIL|BLOCKED",
  "assertions": [{"name": "...", "passed": true, "detail": "..."}],
  "n_rows": 0, "n_excluded": 0, "n_fallback": 0,
  "config_hash": "...", "code_version": "...", "seed": 42
}
```

退出码约定，消解五份规格中「零缺失**或**有失败清单」这类运行时二义：

| 退出码 | 含义 | 下游 |
| --- | --- | --- |
| 0 | `PASS`，全部断言通过 | 继续 |
| 1 | `FAIL`，断言失败或出现未声明的异常 | 中止整条链路 |
| 2 | `BLOCKED`，某个**已声明的数据门**未通过（OSA 提前期连接、Mendeley 库存语义、$\rho_{DS}$ 未转写） | 跳过依赖该门的下游步骤，其余继续；报告中该项标 BLOCKED，不标 FAIL |

「缺失预测率为零」是 `PASS` 条件；存在缺失但已全部登记入失败清单且触发了 §7.3 回退，是 `PASS` **附带** `n_fallback > 0`，必须在报告正文出现。二者不再是可自由选择的分支。

### 9.4 预测表统一 schema

五份规格各自声明的预测表列名以本表为准，逐数据集只允许**追加**列，不得改名或删列：

```
dataset, dataset_version, series_id, origin, target_start, target_end,
split, model_id, run_id, seed,
q50, q85, y_obs, w_i, weight_source,
quantile_source, calib_level, fallback_level, fallback_reason,
is_cold_start, slice_flags, model_fit_end
```

`series_id` 为字符串主键（Fresh: `store_id + "_" + product_id`；Zhao: `sku_ID`；OSA: `store_id + "_" + sku`；Mendeley: `Store + "_" + Product No`；Lokad: `Sku`）。主键 `(dataset, series_id, origin, target_start, model_id, run_id)` 必须唯一。

### 9.5 不确定性区间

按 **series 聚类的 bootstrap**：以 series 为重抽样单元，$B=1000$，百分位法 95% CI，种子 42。模型间比较使用**配对**重抽样（同一组重抽样索引施加于所有模型），报告 $\Delta\text{NPL}$ 的 CI，而非各自 CI 的重叠情况。

Zhao 仅两个测试月（主 MD §2.5），其 CI 必须与「不主张稳健长期时间外泛化」的声明一并出现。

**每次模型比较必须同时报告该 split／切片的最小可分辨效应（MDE）**，定义为 $\Delta$NPL 的 CI 半宽中位数。没有 MDE 的比较无法判断"改进"是否有意义。实现见 `f2d.uncertainty.min_detectable_effect`。

**MDE 必须在两个「相近」模型之间估计，不得用退化对照作基准。** CI 半宽随效应量级增大而增大，因此以 `always-zero` 这类远离真实模型的对照为基准会**系统性高估** MDE。

已实测的偏差幅度（Zhao 日级、周级目标）：

| 基准选择 | validation 相对 MDE | test 相对 MDE |
| --- | --- | --- |
| 以 `always-zero` 为基准（$\Delta\approx-0.36$） | 6.11% | 13.18% |
| 以 `naive-last` 为基准（$\Delta\approx-0.002$） | **3.72%** | **4.02%** |

高估了 64%–228%。规则：

1. **报告 MDE 时**，基准取一个与被比较模型量级相当的真实基线（如 `naive-last`）；
2. **检验接口退化时**，才用 `always-zero`（§5.1），此时关心的是符号与显著性，不是 CI 宽度；
3. 两者是**不同用途的两次 bootstrap**，须分别报告，不得混用同一次结果。

**更进一步：MDE 取决于差异的「结构」，不只是量级。** 两个真实模型的差异是**异质**的（部分序列变好、部分变差），这种异质性正是 CI 宽度的来源。因此：

| 对照类型 | 例子 | 偏差方向 |
| --- | --- | --- |
| 效应大且异质 | `roll-mean-3` vs `naive-last`（$\Delta=0.019$） | **高估**（Zhao 月级 test 测得 17.93%） |
| 效应小但**完全同质** | `base` vs `base×1.02` | **低估**（同一设置测得 0.41%） |
| 效应小且异质（**正确**） | 两个真实模型，如 `lgbm` vs 其消融变体（$\Delta=0.0005$） | 可用（测得 1.16%） |

同质扰动对所有序列施加相同变换，重抽样时 $\Delta$NPL 几乎不变，CI 被人为压窄。**它只能作为分辨率下限（floor），不能当作 MDE。**

因此报告规范：**MDE 必须由一对真实模型给出**，并同时注明该对的 $\Delta$。若只有同质扰动的结果，须标为 `RESOLUTION_FLOOR` 而非 MDE。真正的 MDE 在实际跑完模型对之后才能确定——这不是可以事先估准的量，只能给出区间与依据。

**MDE 还依赖目标的聚合粒度。** 同一数据集，聚合越粗则方差越低、MDE 越小：Zhao 月度目标相对 MDE 1.09%（validation），同一数据日级建模、周级目标则为 3.72%。因此比较不同粒度方案时，**MDE 必须在各自的目标粒度上分别测**，不可跨粒度引用。

**切片级 bootstrap 在切片内部按 series 重抽样**（与 §4.4 的 NPL 不可分解性质一致），不得先在全体上抽样再取切片。

**必须检查切片的独立单元数，而非行数。** 实测教训：Zhao 的冷启动切片 `n_series == n_rows`（1027 == 1027）——`months_observed==0` 只在序列首次出现的那一个月成立，故每条序列恰好贡献一行，聚类 bootstrap 毫无收益。该切片测试窗 MDE = 0.0393 而基数约 0.40，即 **±10% 相对分辨率**，无法支撑任何冷启动结论。若只看"1,027 行"而不看"1,027 个独立单元"，会严重高估该切片的证据强度。

**配对要求同一组重抽样索引施于所有变体。** 实测教训：若在变体间连续消耗同一个 RNG（`for v: for b: rng.choice(...)`），各变体会拿到不同的抽样集，变体之间不再可比。正确做法是预抽样一次后复用。`f2d.uncertainty.paired_bootstrap` 对行序一致性做硬断言——行序错位时该函数仍会返回数字，但那个数字无意义，故必须抛错而非静默通过。

---

## 10. 本文件明确未定义的事项

以下不属于本文件范围，在对应文件落地前，相关步骤记为未就绪：

| 事项 | 归属 | 阻塞的规格 |
| --- | --- | --- |
（本文件的所有外部依赖均已落地。）

已解除（均于 2026-08-22）：

- 服务目标 $\alpha$、保护期换算、五个策略公式、成本参数与 Pareto 协议已落 `docs/07_decision_layer.md`。$\alpha_{\text{primary}}=0.95$，短缺成本由 newsvendor 临界比自 $\alpha$ 导出而非独立编造。

- $\rho_{DS}$ 的来源定义已转写，见 §5.4；spec 01 的协议 A 不再因该项阻塞。
- 各数据集切分边界、历史窗、已知未来协变量、回放窗与 $y_{\min}$ 已落 `configs/*.yaml`（6 个文件，边界均经一致性校验）。
- 全部 `model_id` 的库、版本、超参、TSFM 身份与算力预算已落 `docs/06_model_hyperparams.md`。TSFM 定为 `amazon/chronos-2`（Chronos-2，119.5M），zero-shot 与 LoRA/全参微调三臂对照。

仍未解除的依赖：`statsforecast` 未安装，Croston-SBA 与 TSB 无实现来源，OSA 与 Mendeley 的间歇需求对照不完整（见 `06_model_hyperparams.md` §7）。
