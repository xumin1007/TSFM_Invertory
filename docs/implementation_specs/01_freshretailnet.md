# FreshRetailNet-50K / FreshRetailNet-LT：截断需求与售罄风险复现实规格

## 0. 目标与边界

本规格首先完成来源任务的可比较复现：利用门店×商品的小时销量、小时可售状态和已知协变量，复现需求恢复与随后 7 日预测/售罄风险分析。该数据**没有**数量型现货、采购订单或 ETA，因此不产出补货回放、持有成本或真实未满足需求结论。

本地有两个不可混用的版本：50K 为 `freshretailnet-50k/dataset/` 的官方缓存；LT 为 `freshretailnet_lt/data/train.parquet` 与 `eval.parquet`。每次实验必须在 `dataset_version` 中二选一，不能把二者拼接后称为官方复现。

## 1. 读取的数据与启动验收

| 输入 | 必读字段 | 用途 |
| --- | --- | --- |
| `data/external/freshretailnet-50k/dataset/data-00000-of-00005.arrow` 至 `...04...arrow`，以及 `dataset_info.json`、`source.json` | `city_id, store_id, management_group_id, first_category_id, second_category_id, third_category_id, product_id, dt, sale_amount, hours_sale, stock_hour6_22_cnt, hours_stock_status, discount, holiday_flag, activity_flag, precpt, avg_temperature, avg_humidity, avg_wind_level` | 50K 训练数据与版本记录 |
| `data/external/freshretailnet_lt/data/train.parquet`、`eval.parquet` | 同上 | LT 的已给定训练/评估拆分 |

步骤 1：读取但不改写原始文件，解析 `dt` 为日期，构造字符串主键 `series_id = store_id + product_id`（城市和分类保留为属性）。

步骤 2：输出 `artifacts/fresh/raw_audit.json` 和 `artifacts/fresh/raw_schema.md`，至少包含行数、日期范围、每列缺失率、每个向量列长度、`(series_id, dt)` 重复数、`hours_stock_status` 中 0/1 以外的值数量。

验收：`hours_sale` 与 `hours_stock_status` 对每行长度相等且为 24；`sale_amount` 与 `hours_sale` 的和的差异分布被保存；`stock_hour6_22_cnt` 与营业时段库存状态的定义在数据字典中明确。任何重复主键、未知状态或向量长度异常均须写入隔离表，不可静默聚合。

**50K 数据门**：目前本地状态文件标记五个 Arrow shard 为 `train`，没有本地官方 `eval` shard。故 50K 首轮只能完成训练数据审计和按论文明确写出的时间切分；若无法取得带版本记录的官方评估 split，就不能报告“官方 eval 复现”。LT 的 `train/eval` 则可直接作为首个可运行复现。

## 2. 规范面板与信息集

步骤 3：输出日级规范面板 `artifacts/fresh/panel_daily.parquet`，每行一个 `(dataset_version, series_id, dt)`，含全部静态/分类属性、`observed_sales_daily=sale_amount`、24 维销量/可售向量、`stockout_hour_count`、`availability_ratio`、促销/节假日/天气特征。

步骤 4：输出小时级长表 `artifacts/fresh/panel_hourly.parquet`，每行一个 `(series_id, dt, hour)`；`observed_sales_hour=hours_sale[hour]`。

**极性（已实测，见 `00_global_conventions.md` §5.4.4）：`hours_stock_status == 1` 表示缺货，与来源论文 eq. 1 的 $s$ 相反。** 因此：

```
is_stockout = hours_stock_status[hour]        # 数据集口径：1 = 缺货
s_paper     = 1 - hours_stock_status[hour]    # 论文口径：1 = 有货
```

两列都要落表。必须固化断言 `mean(hours_sale[is_stockout==1]) < mean(hours_sale[is_stockout==0])`，并校验 `stock_hour6_22_cnt == sum(hours_stock_status[6:22])`。

向量展开后逐日重新求和必须还原 `sale_amount`（允许的浮点误差取 `FLOAT_TOL`）。

步骤 5：建立 `artifacts/fresh/forecast_origins.parquet`。每个 origin 记录 `series_id, origin_date, history_start, history_end, target_start, target_end, split`。历史窗口、预测窗口（7 日）和是否允许使用未来已知天气/节假日/活动/折扣必须写入 `configs/fresh_lt.yaml` 或 `configs/fresh_50k.yaml`（按 `dataset_version` 二选一，不可拼接）；未知未来促销/折扣不得从标签期直接泄漏到特征中。

验收：任一预测行只含 `history_end < target_start` 的销售和库存状态；origin 不跨给定的 LT train/eval 边界；按 `series_id`、日期和库存状态统计的行数在面板与 origin 间可对账。

## 3. 来源复现：需求恢复与 7 日预测

步骤 6：以原始销量作为不恢复基线，输出 `artifacts/fresh/recovery_predictions_raw.parquet`：`series_id, dt, hour, split, observed_sales_hour, is_available, recovered_demand_hour, model_id`。原始基线令 `recovered_demand_hour=observed_sales_hour`。

步骤 7：分别训练 TimesNet、iTransformer、DLinear 三个代表性恢复模型；模型只能使用 origin 时点前销量/可售历史和在时点可知的协变量。将每个模型写入同一预测表，另保存 `artifacts/fresh/recovery_run_manifest.json`（代码版本、数据版本、seed、窗口、参数、训练截止日）。SAITS、ImputeFormer、GPVAE、CSDI 不在首轮必跑集内；文档中引用其来源结果而非伪造复现。

**恢复输出的硬约束（来源论文 eq. 2）**：$d = y\odot s_{\text{paper}} + \hat d\odot(1-s_{\text{paper}})$，即**有货小时必须原样保留观测销量，模型估计只填入缺货小时**。落表前逐行校验 `recovered_demand_hour == observed_sales_hour` 在 `is_stockout == 0` 的全部行上成立（容差 `FLOAT_TOL`）；违反即 `FAIL`，不得以「模型平滑」为由放行。

步骤 8：把恢复后的小时需求按日求和，建立 7 日预测任务。运行 `Raw + TFT`、`TimesNet + TFT`、`iTransformer + TFT`、`TimesNet + DLinear`、`TimesNet + SSA`，输出 `artifacts/fresh/forecast_7d_predictions.parquet`：

`dataset_version, series_id, origin_date, target_date, model_id, demand_point, q50, q85, observed_sales_daily, target_available, split`。

点预测模型进入统一研究比较前，只能用训练窗拟合、验证窗冻结的残差分位数校准器生成 `q50/q85`；测试标签不得参与校准。

验收：每个 `model_id × origin × target_date` 一行；q50/q85 有限、非负且 q50≤q85；缺失预测率为零或有明确的失败清单；同一 seed 重跑预测哈希一致（浮点容差在配置中固定）。

## 4. 产出表与评价

| 表 | 内容 | 验收 |
| --- | --- | --- |
| `recovery_metrics.csv` | `model_id, split, WAPE, WPE, rho_DS, abs_rho_DS, rho_ds_excluded_pairs, rho_ds_excluded_mu_share, n_hours, stockout_share` | 只在来源定义允许的小时/样本上计算（MNAR 恢复只在合成删失区间）；分母为零的组按 `00_global_conventions.md` §5.5 单列报告；$\rho_{DS}$ 与来源 Table 2 锚点值量级比对 |
| `forecast_metrics_source.csv` | `model_id, split, WAPE, WPE, n_eligible_days` | 来源复现仅在运营无缺货目标期计算；资格掩码保存 |
| `forecast_metrics_unified.csv` | `model_id, split, NPL_50_85, coverage_50, coverage_85, n` | 只对可识别目标，或清楚标为“观测销量下界” |
| `fresh_stress_slices.csv` | 低可用性、活动/促销、高库存压力、未见系列切片 | 每片同时有样本数、预测指标；空切片必须标记 |

WAPE、WPE、$\rho_{DS}$ 的定义已从来源论文逐项转写至 `docs/00_global_conventions.md` §5.3–§5.4（含来源锚点值、方向性、以及四项由本项目声明的实现细节）。统一 NPL 使用 `00_global_conventions.md` §4 的 q50/q85 加权归一化 pinball；它不替代来源指标。本数据集的 $y_{\min}$ 按 §4.3 的阶梯规则在原始审计阶段选定后写入 `configs/fresh_lt.yaml` 或 `configs/fresh_50k.yaml`（按 `dataset_version` 二选一，不可拼接）。

## 5. 本数据集的完成判定

完成需要同时满足：原始审计通过；LT 的 train/eval 复现流程可重跑；所有预测主键完整且无测试泄漏；来源指标和统一预测指标均附带资格掩码；研究报告明确写明本数据不支持数量型库存、PO、ETA 或服务—成本回放。若 50K 官方 eval 未在本地，50K 结果标为“训练数据审计/内部时间切分”，不得标为官方评估复现。
