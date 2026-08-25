# Mendeley Retail Transactions and Stocks：门店-SKU外部库存/销量验证规格

## 0. 目标与边界

该数据集用于外部门店-SKU 库存/销量、间歇需求和冷启动的可迁移性检查。它没有采购订单、到货日期或 ETA，故不进行 ETA 推断、连续补货回放或真实短缺成本结论。

## 1. 读取的数据与语义确认门

| 输入 | 必读字段 |
| --- | --- |
| `Retail Transactions and Stocks Data/retail_sales_ml_apl.csv` | `Transaction Date, Product No, Store, Sales Channel, Sales Type, Is Return, Qty Sold, Sales Amount, Cogs, Number of Transactions` 及分类/供应商字段 |
| `Retail Transactions and Stocks Data/retail_inventory_ml_apl.csv` | `Start Date, End Date, Product No, Store, Stock Status, Qty on hand, Stocks Selling Amount, Cost of Stocks, Stock Unit Selling Price, Stock Unit Cost Price` 及分类/供应商字段 |

步骤 1：输出 `artifacts/mendeley/raw_audit.json`：日期范围、每文件行数、`(Store,Product No,date)` 重复、退货比例、负销量/库存、库存区间长度分布、分类属性在同一 SKU 上的冲突率。

**语义确认门**：库存表含 `Start Date` 与 `End Date`，其记录是否为区间状态、区间内常量、还是事后汇总，不能从字段名自行假定。输出 `artifacts/mendeley/inventory_semantics_audit.md`，检查同一 `(Store,Product No)` 的区间重叠、缺口、同日冲突，并在数据发布说明中确认可用时点。未确认前不得把 `End Date` 或覆盖至未来的库存区间作为任一预测时点的特征。

## 2. 安全的规范面板

步骤 2：输出 `artifacts/mendeley/sales_daily.parquet`。按 `(Store, Product No, Transaction Date)` 聚合净 `Qty Sold`；退货处理规则必须预注册（例如保留负净销量，或按交易日净额），并同时保存毛销量、退货量和交易数。销售总量必须与原始行对账。

步骤 3：若语义确认门通过，输出 `artifacts/mendeley/panel_daily.parquet`，每行 `(Store, Product No, date)`：过去销量滞后/滚动、历史退货、静态产品/门店/渠道属性，以及在 date 时点**已经可知**的库存观察。库存特征必须记录 `inventory_observation_date`；该日期不得晚于预测 origin。若门无法通过，则输出 `sales_only_panel.parquet`，并把库存实验状态标为 blocked，而不是通过区间末日回填。

步骤 4：输出 `feature_lineage.csv` 和 `forecast_origins.parquet`。采用按时间的训练—验证—测试切分，具体边界以日期审计后写入配置；仅在拥有至少一个完整年度周期时使用年度季节性朴素，否则退化为同频上一期/滚动历史。

验收：销售/库存连接不得因 inner join 丢失销售日期；每一特征能追溯到 origin 前；新 SKU、新门店、新组合定义为相对训练窗未见，并保持单独标志。

## 3. 模型、预测表与评价

步骤 5：运行季节性朴素（或前述可用的同频历史）、Croston-SBA、TSB、LightGBM/HGB q50/q85、TSFM q50/q85。输出 `artifacts/mendeley/predictions.parquet`：

`store_id, product_no, origin_date, target_start, target_end, split, model_id, q50, q85, observed_net_sales, inventory_observation_date, sparse_series_flag, cold_start_flag`。

Croston/TSB 与其他点预测基线的分位数由训练/验证前置残差校准器生成；GBDT 直接训练 q50/q85。保存每次运行的 `run_manifest.json`、参数、seed、样本数和编码器拟合截止日。

步骤 6：输出 `artifacts/mendeley/metrics_prediction.csv`：来源没有官方模型排行榜，故主要报告 q50/q85 NPL、覆盖率、校准、WAPE（作为易解释补充）和按稀疏度/库存可见性/新品/门店切片的指标。对有库存但缺货语义不充分的样本，将 pinball 标为观测净销量损失，不称真实需求损失。

验收：每模型同一测试主键完整；分位数合法；切片含样本数；全部指标的零需求分母规则写入 `metrics_definition.md`；库存语义未通过时，报告自动排除库存特征和所有决策层指标。

## 4. 完成判定

完成需要销售面板和时序预测可重跑、区间库存可用时点得到证实或被明确排除、间歇需求与冷启动切片齐全。研究报告必须把 Mendeley 定位为外部预测/库存可见性验证，而非 ETA 或补货政策因果验证。
