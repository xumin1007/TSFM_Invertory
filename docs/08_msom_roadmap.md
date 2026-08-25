# §8 MSOM 投稿路线图

目标：把当前单数据集技术报告升级为 MSOM 可投稿论文。六项补充按依赖关系和优先级排列。

---

## WS-1 第二个数据集的决策层（P0）

**动机。** 单数据集无法回答"这是 Zhao 的特殊性质还是一般性结论"。MSOM 审稿人第一反应。

**选择。** 项目已有 OSA（仿真/完整需求）和 Lokad（多 DC 网络）的数据接口。优先 OSA：

| 属性 | Zhao | OSA |
| --- | --- | --- |
| 需求观测 | 截断（观测销量） | 完整（仿真真值） |
| 需求分布 | 间歇（零占比 68%） | 连续（快消） |
| 提前期 | 1 天（P1≈P2≈P3） | 声明情景，可设 7–14 天 |
| 策略可区分 | 弱（m≈1.03） | 强（m 可达 2–3） |

OSA 的提前期足够长，使 P1/P2/P3 产生实际差异，这在 Zhao 上被压平了。

**产出。**
- `src/f2d/run_osa_daily.py` — 预测层（chronos2-zs, gbdt-lean, emp-daily）
- `src/f2d/run_osa_decision.py` — 决策层（层 B + Pareto 扫描）
- 与 Zhao 的交叉验证表："预测≠决策"在两个数据集上同时成立？

**前置。** 无。可立即开始。

**完成标志。** 两个数据集的 Pareto 前沿并列展示，结论一致性有定量判定。

---

## WS-2 层 C 连续回放（P0）

**动机。** 当前层 B 每月从观测快照重置，不跨月连接。真实库存是连续演化的——上月超配的库存压低本月订货，欠配的缺货会累积。MSOM 审稿人会认为月级重置是重大简化，成本的绝对量级不可信。

**范围。**
- 日级循环递推（`07_decision_layer.md` §7.1 已固定公式）
- 守恒断言（§7.2 的 5 条）
- 支持 order-up-to 和 (s,S) 两类策略
- 支持 lost-sales 和 backorder 两种缺货模式（Zhao = lost-sales）

**对 (s,S) 的依赖。** 层 B 下 $\kappa_K=0$ 时 $(s,S)$ 退化为 order-up-to，无新信息。层 C 允许 $\kappa_K>0$，(s,S) 才有意义。故 (s,S) 的评价必须在层 C 上做。

**产出。**
- `src/f2d/simulation.py` — 日级递推引擎 + 守恒断言
- `src/f2d/run_zhao_layerc.py` — Zhao 上的连续回放
- 层 B vs 层 C 的排序一致性检验（如果排序翻转，层 B 的结论需要限定）

**前置。** 无。可与 WS-1 并行。

**完成标志。** 层 C 结果的臂间排序与层 B 一致（或明确报告不一致处），且 (s,S) 在 $\kappa_K>0$ 下有独立的结论。

---

## WS-3 Decision-aware fine-tuning（P1）

**动机。** 当前发现"微调用服务换成本，净效果不显著"。MSOM 审稿人会问：既然你说 pinball loss 不够，为什么不用决策损失？这是把论文从"发现问题"升级为"解决问题"的关键。

**方法。** 三个候选，按实现难度排序：

1. **Newsvendor cost 作为 fine-tuning loss。** 最直接：$L = h \cdot (S - y)^+ + p \cdot (y - S)^+$，其中 $S = F^{-1}(\alpha)$ 从模型输出的分位数取。需要把 $F^{-1}$ 的梯度传回去（直通估计器或重参数化）。
2. **SPO+ loss（Elmachtoub & Grigas 2022）。** 用决策的最优性损失作为代理，凸且有一致性保证。需要实现 SPO+ 的梯度。
3. **End-to-end differentiable inventory optimization。** 把整个 newsvendor 决策嵌入计算图。最完整但工程量最大。Bertsimas & Kallus (2020) 的 data-driven newsvendor

**建议路径。** 先做方法 1（工程量小，概念直接），与标准 pinball loss 微调对比。如果能打破"服务换成本"的僵局，就是论文的核心方法贡献。

**产出。**
- `src/f2d/losses.py` — newsvendor cost loss + SPO+ loss
- `src/f2d/run_zhao_finetune_decision.py` — decision-aware 微调
- 对照实验：pinball vs newsvendor-cost vs SPO+ 在预测层和决策层的表现

**前置。** WS-1 或 WS-2 的数据准备完成（需要在多数据集上验证方法有效性）。

**完成标志。** 至少一种 decision-aware loss 在决策层上显著优于 pinball loss 微调，且不以服务水平退化为代价。

---

## WS-4 经济显著性与敏感性分析（P1）

**动机。** 统计显著 ≠ 经济显著。MSOM 审稿人和 practitioner 读者需要看到：省了多少钱、值不值得换模型。

**内容。**

### 4a 年化经济影响
- 单期成本差 × SKU 数 × 12 个月 → 企业级年化节省
- 部署 TCO 对比：TSFM 零样本（推理成本 only）vs GBDT pipeline（特征工程 + 训练 + 监控 + 数据质量）
- ROI 估算：切换模型的一次性成本 vs 年化节省

### 4b 敏感性扫描
- $\kappa_h \in \{0.10, 0.15, 0.20, 0.25, 0.30\}$：持有成本率变化是否翻转排序
- 提前期 $L \in \{1, 3, 7, 14\}$ 天：模拟不同提前期结构（Zhao 只有 L=1，需要在 OSA 上测或做半合成实验）
- 服务目标 $\alpha \in \{0.85, \dots, 0.999\}$：已有 Pareto 扫描数据，补充经济解读
- 预测视界：日级 vs 周级聚合目标对决策层的影响

### 4c Practical guideline
- 决策树/流程图：企业在什么条件下应该选 TSFM vs GBDT vs 经验法
- 边界条件：需求间歇度、SKU 数、数据可得性的交互影响

**前置。** WS-1 完成后可用多数据集做。WS-4a/4b 可在当前数据上先做初版。

**完成标志。** 一张表列出各因素变化时排序是否翻转，附企业级年化数字。

---

## WS-5 截断需求修正（P2）

**动机。** Zhao 的观测销量 $S = \min(D, \text{inventory})$。当前做法是标注为下界并限定表述（§10），但 MSOM 社区对 censored demand estimation 有成熟方法（Huh & Rusmevichientong 2009, Shi et al. 2016, Tang et al. 2023），审稿人会期望至少做敏感性检查。

**文献定位。** 三条相关研究线：

- **Nonparametric inventory with censored demand（Shi, Chen & Duenyas, OR）：** 提出 DDM 算法——不估计需求分布，直接在截断数据上用梯度下降优化 order-up-to 水平，regret $O(1/\sqrt{T})$。这是截断需求下 nonparametric 方法的理论最优速率。**关键差异**：DDM 假设 iid 需求，我们的 Zhao 数据明确违反（drift ratio 0.878，§6.2.0 已实测）。非平稳性恰好是 TSFM 的优势区。
- **Offline pricing under censored demand（Tang, Qi, Fang & Shi）：** 用因果推断（potential outcome framework）+ survival analysis 从截断销量恢复真实需求函数。他们量化了"忽略截断"的代价：定价偏低 30%+，利润损失 5%。方法论上的 Kaplan-Meier 修正可以直接借鉴到库存场景。
- **在线学习线（Huh & Rusmevichientong 2009, Ban 2020）：** 从截断数据 online 学习最优库存策略。与我们的 offline 设置互补。

**方法。** 三条互补路径，按论证力度排序：

1. **在非截断数据集上验证（最干净）。** OSA 有完整需求真值。如果 Zhao 和 OSA 的结论一致，截断不是驱动因素——不需要估计截断需求，只需要对照。
2. **DDM 作为 baseline（Shi et al.）。** 在 Zhao 截断数据上实现 DDM：直接从截断销量学习 order-up-to 水平，不经过需求估计。与 TSFM 的"先预测再决策"路径做对比。如果 TSFM 零样本优于 DDM，说明预训练知识在截断+非平稳场景下有增量价值——这是一个有理论对标（$O(1/\sqrt{T})$ regret bound）的定量结论。
3. **Survival analysis 修正（Tang et al.）。** 用 Kaplan-Meier 类方法从 $S = \min(D, Y)$ 恢复 $D$ 的分布，重跑决策层看排序是否变化。作为 robustness check 放附录。

**建议路径。** 路径 1（OSA 对照）是主论证。路径 2（DDM baseline）是加分项——如果做了，论文的理论深度显著提升，因为它把"TSFM vs 经典方法"从纯实证比较升级到"有理论对标的实证比较"。路径 3 降级为附录。

**产出。**
- 截断 vs 非截断数据集的结论对照表
- `src/f2d/baselines/ddm.py` — Shi et al. DDM 算法实现（nonparametric gradient descent on censored data）
- （可选）`src/f2d/censored.py` — Kaplan-Meier 截断修正
- 文献引用清单：Huh & Rusmevichientong (2009), Shi, Chen & Duenyas (2016), Tang, Qi, Fang & Shi (2023), Ban (2020)

**前置。** WS-1（OSA 提供非截断对照）。DDM baseline 可在 Zhao 上先做，不依赖 WS-1。

**完成标志。** 能对审稿人说："(a) 在非截断数据集上结论不变；(b) TSFM 零样本在截断+非平稳数据上优于理论最优速率的 nonparametric 方法（DDM），说明预训练知识的增量价值不来自忽略截断"。

---

## WS-6 零 context 冷启动（P2）

**动机。** 当前冷启动定义为 context < 90 天，但 MSOM 更关心的是**新 SKU 引入决策**——上架一个新品，历史数据为零，初始库存怎么定。TSFM 的预训练知识在此场景下理论价值最大。

**实验设计。**
- **零样本实验**：context = 0，TSFM 只依赖品类信息（通过 prompt 或 cross-series transfer），GBDT 完全无法出价。
- **Few-shot 梯度**：context = 7, 14, 30, 60, 90 天，画出性能随历史长度的衰减曲线。
- **类比法对照**：实务中常用"找相似 SKU 的历史"作为新品预测。实现一个 k-NN 类比基线。

**产出。**
- 性能 vs context 长度的衰减曲线（prediction + decision layer）
- TSFM vs 类比法在零/少 context 下的对比
- 新 SKU 首期补货的实际建议

**前置。** WS-1（需要在多数据集上验证）。

**完成标志。** 能给出"新 SKU 上架后，用 TSFM 零样本比用类比法便宜 X%"的定量结论。

---

## 依赖关系与时间线

```
WS-1 (OSA 决策层) ──┬──→ WS-3 (decision-aware FT)
                     ├──→ WS-4 (经济显著性, 多数据集版)
                     ├──→ WS-5 (截断修正, 用 OSA 对照)
                     └──→ WS-6 (零 context 冷启动)

WS-2 (层 C 连续回放) ──→ WS-3 (需要层 C 评价 (s,S))

WS-4a/4b (单数据集版) ──→ 可立即开始
```

建议执行顺序：**WS-1 + WS-2 并行 → WS-3 → WS-4 → WS-5/WS-6**。

WS-1 和 WS-2 是 P0，没有它们论文的 external validity 和 operational realism 都站不住。WS-3 是 novelty 的来源。WS-4–6 是 robustness 和深度。
