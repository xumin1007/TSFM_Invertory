# Zhao SKU×月：需求分位数与单期补货代理实现规格

## 0. 目标与边界

该数据集是主真实数据预测—决策接口。一个样本是 `(sku_ID, month t)`；在月初信息集预测该自然月累计 `observed_sales_next_month` 的 q50/q85。它的库存是月度快照，不能重建为日库存，也不能把销售称为潜在真实需求。决策实验是每月重置到观测快照的**单期代理**，不是连续因果回放。

## 1. 读取的数据与原始审计

| 输入 | 文件/工作表 | 必读字段 |
| --- | --- | --- |
| 月初库存 | `Zhao/nav21957-sup-0001-supinfo01.xlsx`, `sheet` | `date, sku_ID, category, subcategory, unit, beginning_inventory, on-order_inventory, stock_value` |
| 采购订单 | `...sup-0002-supinfo02.xlsx`, `sheet1` | `sku_ID, order_date, arrival_date, quantity, unit_cost, normal_lead_time` 与类别字段 |
| 销售明细 | `...sup-0003-supinfo03.xlsx`, `Jan`–`Oct` | `date, sku_ID, quantity`（July 的 `quantiity` 统一为 `quantity`）、价格/品牌字段；统一 `sales_revenue` 拼写变体 |
| SKU 属性 | `...sup-0004-supinfo04.xlsx`, `Sheet1` | `sku_ID, brand_ID, introduction_year, stop_year, operation_mode` |
| 陈列状态 | `...sup-0005-supinfo05.xlsx`, `Sheet1` | `date, sku_ID, facing_number, shelf_capacity` |

步骤 1：将所有月份转为月初时间戳，保留原行来源和原字段名，输出 `artifacts/zhao/raw_audit.json`、`raw_schema.md`。记录 2019-01 至 2019-10 的每表行数、`sku_ID` 覆盖、主键重复、缺失率、负数量和采购日期异常。

步骤 2：输出 `artifacts/zhao/sales_monthly.parquet`：每行 `(sku_ID, month)`，`observed_sales_next_month=sum(quantity)`，同时保存销售行数、收入、价格的可审计聚合。对缺少销售的 SKU×月，只有在月初库存快照表中存在该 SKU×月时才补零；不能把“没有交易记录”无条件解释为所有 SKU 的零需求。

验收：销售聚合量等于原明细 `quantity` 总和；月初库存 `(sku_ID, month)` 唯一；销售、库存、陈列和静态属性的连接覆盖率被报告，而不是因 inner join 静默删样本。

## 2. 事前特征面板与信息隔离

步骤 3：输出 `artifacts/zhao/panel_monthly.parquet`。主键 `(sku_ID, month)`，目标为该月的 `observed_sales_next_month`。特征只包括：

- 月初 `beginning_inventory, on-order_inventory, stock_value, facing_number, shelf_capacity`；
- 静态/截至 t 已知的 SKU、类别、品牌、生命周期字段；
- 仅来自 `t-1` 及更早销售的 1/3/6 月滞后及滚动特征；
- `arrival_date < t` 的 1/3/6 月实际收货量、`months_since_last_receipt`；
- `order_date < t` 的订单，按 `promised_eta=order_date+normal_lead_time` 得到预测月、下月和逾期在订量；
- 只用已在 t 前完成订单计算的 ETA 误差滚动中位数/p85。

`arrival_date` 是事后实际到货结果：不能直接作为 t 后订单的特征，不能按实际未来到货日期做 ETA 分桶。`on-order_inventory` 是首选在订快照；订单重建未交付量仅输出审计差异，绝不替代快照。

**静态属性同样受 origin 截断约束（已实测的泄漏）。** `sup-0004` 的 `stop_year` 编码的是停产**事件年份**，不是恒久属性。实测 562 个 SKU 的 `stop_year = 2019`（另有 140 个 > 2019），落在数据期内；在 2019 年任一 origin 上「该 SKU 将于 2019 年停产」不可观察。面板中 **3,653 行（1.5%）** 受影响，且该值与目标强相关（已停产行目标均值 2.4、未停产 43.3，相差 18 倍），属会抬高分数的真实泄漏。

因此 `stop_year` 必须按 origin 截断，并派生为可解释特征：

```
known            = stop_year < year(origin)      # 严格早于当年才可观察
stop_year_obs    = where(known, stop_year, NaN)
is_discontinued  = known
years_since_stop = where(known, year(origin) - stop_year, NaN)
```

原始 `stop_year` 列**不得**直接进入特征集。`introduction_year` 同理需 `<= year(origin)` 才可用，但因新 SKU 在进入月初快照前本就没有面板行，实测无越界样本。

一般规则：**任何编码事件日期的静态属性都须按 origin 截断**，判据与 §2.4.1 的「origin 时点可观察」一致——字段是否静态不决定它是否可用，取值何时确定才决定。

步骤 4：输出 `artifacts/zhao/feature_lineage.csv`：`feature, source_table, availability_rule, transformation, max_source_month, missing_rule`。对每个面板行输出 `max_feature_date`。

验收：任一特征的 `max_feature_date < month`；标签不进入任何编码/标准化器；`on-order_inventory` 与订单重建量的差异分布报告；未见 SKU/类别在验证和测试映射至训练窗定义的未知值。

## 3. 固定切分、模型与分位数

步骤 5：输出 `artifacts/zhao/split_manifest.parquet`。固定：1–6 月为初始训练/上下文；7–8 月滚动验证；9–10 月冻结 prequential 测试。10 月允许使用已结束的 9 月实际历史重新拟合，但不允许依据 9 月得分改变模型、特征、校准、成本或策略。

步骤 6：运行上一可观测月、1/3/6 月滚动历史、LightGBM/HGB 分位数和 TSFM 分位数。每个模型输出 `artifacts/zhao/predictions.parquet`：

`sku_ID, month, split, model_id, q50, q85, observed_sales_next_month, is_cold_start, train_end_month, calibration_train_end`。

点预测基线/TSFM 点输出以训练窗拟合、验证窗冻结的残差分布转换为 q50/q85。GBDT 直接训练两个分位数目标。每一模型须保存 `run_manifest.json`（seed、训练行、特征清单、编码器拟合截止月、参数）。

验收：测试预测覆盖每个合格 `(sku_ID, month)`，无重复、无 NaN、q50≥0、q85≥q50；7 月特征不读 7 月销售，9–10 月不驱动模型选择；所有模型共享 split manifest。

## 4. 单期补货代理、表和验收

步骤 7：在验证窗选择固定的 order-up-to、ETA 覆盖、安全库存近似、lead-time-demand convolution、`(s,S)` 候选映射。对每个 `(sku_ID, month)` 从当月观测 `beginning_inventory` 与 `on-order_inventory` 重置，输出 `artifacts/zhao/decision_proxy.parquet`：

`sku_ID, month, model_id, policy_id, initial_on_hand, open_purchase_qty, eta_bucket_qty, target_stock, order_qty, observed_sales, ending_inventory_proxy, observed_shortage_lower_bound, holding_cost_scenario, feasible`。

持有成本只使用预先固定的敏感性 `a∈{.10,.20,.30}` 与 `h=a*unit_cost/12`；不伪造短缺罚金、欠单、月内到货顺序或真实服务率。

步骤 8：输出 `artifacts/zhao/metrics_prediction.csv`（q50/q85 NPL、覆盖率、按类别/新品/高库存压力切片）与 `artifacts/zhao/metrics_decision.csv`（服务下界、月度持有成本、订单率、可行性、Pareto 前沿）。9 月、10 月及合并结果分开报告，并按 SKU 聚类 bootstrap 给出不确定性区间。

完成判定：原始连接可对账；信息隔离自动检查通过；冻结测试上预测/决策主键完整；报告所有结论均称为“观测销售”和“单期代理”，不称真实需求或连续库存因果效应。
