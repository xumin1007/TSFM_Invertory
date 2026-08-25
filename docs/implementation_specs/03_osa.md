# OSA-Data：完整状态压力测试与连续机制回放实现规格

## 0. 目标与边界

OSA 是日级合成机制基准，用于比较需求分位数、库存状态与策略在同一需求/到货轨迹下的服务—成本权衡。它支持连续回放，但其销售列是否等于未截断真实需求必须先经数据说明或生成机制确认；确认前，销售仍记为 `observed_sales`。

## 1. 读取的数据与关键连接门

| 输入 | 必读字段 |
| --- | --- |
| `osa-data/osa_raw_data.csv` | `date, store_id, sku, product_category, total_sales_units, on_hand_inventory_units, replenishment_units, inventory_pipeline, units_in_transit, units_in_dc, units_on_order, units_under_promotion, shelf_capacity` |
| `osa-data/vendor_leadtime_info.csv` | `vendor_id, sub_vendor_id, store_id, item_id, LEAD_TIME_IN_DC, LEAD_TIME_IN_TRANSIT, LEAD_TIME_ON_ORDER` |

步骤 1：输出 `artifacts/osa/raw_audit.json`：日期范围、`(store_id,sku,date)` 重复、负库存/销售、库存流量缺失、每日/系列连续性和每列范围。

步骤 2：单独输出 `artifacts/osa/leadtime_join_audit.csv`。原始日表没有 `vendor_id`/`sub_vendor_id`，且其 `store_id, sku` 与提前期表的 `store_id, item_id` 是否同一编码尚未被证明。分别测试候选键 `(store_id,sku)=(store_id,item_id)` 的覆盖率和一对一性。

验收门：只有在键覆盖率、基数和业务编码均被人工确认后，才将提前期字段写入训练/回放表。未通过时，日表预测仍可运行，但 ETA 压力与提前期回放必须标记为不可用，不能以零或中位数静默填充。

## 2. 面板、切分与预测任务

步骤 3：输出 `artifacts/osa/panel_daily.parquet`，主键 `(store_id, sku, date)`，包含原始状态、`observed_sales_next_day=total_sales_units`、滞后销量（1/7/28）、滚动统计、当前库存、在订/在途/DC 管道、促销和货架压力比例。每个特征输出到 `feature_lineage.csv`，并记录最大可用日期。

步骤 4：输出 `artifacts/osa/forecast_origins.parquet`。按整段日历的时间顺序划分训练、验证、测试；实际边界由数据审计后写入 `configs/osa.yaml`，不得随机切分。每个 origin 预测 next-day 与预设覆盖期累计需求；累计目标应由日预测分布聚合/卷积，不能把单日 q85 当成覆盖期 q85。

步骤 5：运行移动平均/季节性朴素、Croston-SBA/TSB（稀疏 SKU）、LightGBM/HGB q50/q85、TSFM q50/q85；风险头固定比较 `state-HGB` 与 `state + TSFM pressure-HGB`。输出 `artifacts/osa/predictions.parquet`：

`store_id, sku, origin_date, target_start, target_end, split, model_id, q50, q85, observed_demand, availability_state, cold_start_flag`。

验收：过去特征日期严格早于 origin；所有模型使用相同 origin；预测完整、有限、非负且有序；点预测转换为分位数时，残差校准仅在训练/验证前置窗拟合。

## 3. 连续机制回放

步骤 6：定义并输出 `artifacts/osa/replay_manifest.json`：起点、至少 84 日路径、review cadence、策略候选、初始库存、外生需求路径、到货延迟情景、成本参数、lost-sales/欠单机制、seed。所有模型/策略共享该 manifest。

步骤 7：输出逐日状态表 `artifacts/osa/replay_daily.parquet`：

`run_id, model_id, policy_id, store_id, sku, date, beginning_on_hand, arrivals, demand_path, fulfilled, lost_sales_or_backorder, ending_on_hand, pipeline_before, order_qty, pipeline_after, eta_scenario`。

每日期间顺序固定为：接收以前下单且在当天到达的数量 → 按固定需求路径履约 → 按策略下单。不得用历史期末库存强行覆盖模拟库存；历史库存只可用作每条独立路径的初始状态或校验。

步骤 8：输出 `artifacts/osa/replay_metrics.csv`：周期服务率、填补率、平均库存、持有/采购/调拨/短缺成本、订单事件数、服务—成本 Pareto、按低库存/促销/延迟/冷启动的切片。

验收：每日库存守恒（`end = begin + arrivals - fulfilled`，以及声明的状态变换）；`fulfilled≤demand_path`；无负库存（lost-sales 情景）；同一需求与到货路径跨策略一致；先以观测销售路径报告，再将任何高需求路径明确标为合成敏感性。

## 4. 完成判定

原始状态审计、提前期连接审计、预测表和回放状态表均须通过各自主键/守恒检查。若需求真值或提前期连接未能验证，成果可限于预测和观测销售路径敏感性，不能报告真实短缺或 ETA 因果结论。
