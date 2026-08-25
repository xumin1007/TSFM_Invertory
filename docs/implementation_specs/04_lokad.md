# Lokad distribution-network workshop：预测输入与网络库存回放实现规格

## 0. 目标与边界

本规格复现 Workshop #3 的既有 `WeeklyForecast` 网络库存/调拨分析，并比较 Workshop #4 低维概率模型及本研究分位数模型。Lokad 是合成教程数据；其结果用于机制和敏感性证据，不代表 Zhao 的真实成本或政策事实。

## 1. 读取的数据、粒度和审计

| 输入 | 必读字段 | 用途 |
| --- | --- | --- |
| `lokad/Catalog.tsv` | `Ref, Category, Supplier, Brand, BuyPrice, SellPrice` | SKU 成本与属性 |
| `lokad/SKU.tsv` | `Sku, Ref, Loc, StockOnHand, BuyPrice, SellPrice` | 当前网络状态 |
| `lokad/StockHistory.tsv` | `Date, Sku, Ref, Loc, Category, StockOnHand` | 周级库存历史 |
| `lokad/Orders.tsv.gz` | 至少 `Date`、SKU/地点、销售量字段（先由 schema 审计冻结精确列名） | 实际销售历史 |
| `lokad/PurchaseOrders.tsv` | `PONumber, Date, Ref, DeliveryDate, OrderQty, DeliveryQty, NetAmount, IsClosed, Type, OriginLoc, DestinationLoc` | 采购/调拨、已交付量与 ETA |
| `lokad/WeeklyForecast.tsv` | `Date, Sku, Ref, Loc, Sales, Forecast` | Workshop #3 的给定预测输入 |

步骤 1：输出 `artifacts/lokad/raw_audit.json` 与 `schema_snapshot.md`：每表日期范围、行数、重复键、每个 `Sku/Ref/Loc` 的连接覆盖率、订单数量与收货数量异常、未关闭订单、订单日期晚于交付日期、负库存和按时间的表重叠。

`StockHistory`、销售、PO 与 `WeeklyForecast` 的有效日期范围不必完全重叠；不能默认任意一张表可用于另一张表所有日期。步骤 2 输出 `artifacts/lokad/time_overlap_manifest.parquet`，只列出拥有所需库存、销售、PO 和预测信息的有效 origin。

验收：`Sku` 与 `(Ref,Loc)` 的一对一/多对一关系已记录；每一回放 origin 都在 overlap manifest 内；`DeliveryDate` 被标注为教程中的业务承诺/交付日期，其可用时点规则明确。

## 2. 预测输入与可比较表

步骤 3：输出 `artifacts/lokad/panel_weekly.parquet`，主键 `(Sku, week)`，含历史销量、历史库存、未完成 PO 的 `OrderQty-DeliveryQty`、来源/目的地、商品成本与类别。只用 `Date < origin` 的订单和库存状态构造事前特征。

步骤 4：生成三个预测输入：既有 `WeeklyForecast`；Workshop #4 低维模型 `exp(level)×seasonality×linear trend`（训练目标为点预测 MSE，并依来源工作构造概率分布）；本研究 q50/q85 模型。输出统一表 `artifacts/lokad/predictions_weekly.parquet`：

`Sku, Ref, Loc, origin_week, target_week, split, model_id, point_forecast, q50, q85, observed_sales, model_fit_end`。

低维模型的概率分布以 CRPS 评估。若既有 `WeeklyForecast` 是点值，进入统一 q50/q85 比较前只能用前置窗口残差校准，不能利用测试销量。

验收：三种输入均覆盖同一有效 `Sku×week` 集合；模型拟合截止日早于 target；所有分位数合法；`WeeklyForecast` 不被误称为本项目训练结果。

## 3. 网络库存与调拨回放

步骤 5：先完成来源相容的静态 Workshop #3 视图，输出 `artifacts/lokad/workshop3_decisions.parquet`：

`as_of_date, Ref, origin_loc, destination_loc, model_id, stock_on_hand, forecast_input, coverage_demand, excess_qty, transfer_qty, transfer_cost, service_proxy`。

其固定约束（无欠单、四周调拨节奏、上次调拨 7 日前、覆盖期为提前期加两个调拨周期）必须在 `replay_manifest.json` 中逐项声明，不得改写为来源的连续回放。

步骤 6：本研究新增的连续机制回放须先在 overlap manifest 内选择至少 84 日路径，输出 `artifacts/lokad/replay_daily.parquet`，字段与 OSA 规格一致，增加 `origin_loc,destination_loc,transfer_cost`。每条路径按“到货→履约→下单/调拨”推进，候选模型共享需求、初始库存、订单约束、ETA 情景和 seed。

步骤 7：输出三组结果：`forecast_metrics_source.csv`（MSE、CRPS）、`forecast_metrics_unified.csv`（q50/q85 NPL、覆盖率）和 `decision_metrics.csv`（可用率、缺货/积压、持有成本、调拨成本、ROI、订单数、Pareto）。

验收：库存守恒；订单/调拨量非负；未完成量不超过订单剩余量；来源静态 Workshop 与新增连续回放分别标识；外部成本参数与 Zhao 成本绝不混用。

## 4. 完成判定

仅当时间重叠被证明、预测输入同样本可比较、静态来源视图与连续机制视图被分开报告且回放守恒通过，才可写入 Lokad 的服务—成本结论。
