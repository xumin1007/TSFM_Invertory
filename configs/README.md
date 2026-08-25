# configs/ — 逐数据集冻结配置

每个文件固定一个数据集的**切分边界、历史窗、指标常量与能力开关**。所有取值均由实测得出（见各文件 `provenance` 段），不是估计值。

机制与公式一律在 `docs/00_global_conventions.md`；本目录只放**取值**。文件中出现的常量不得在代码里被覆盖。

| 文件 | 数据集 | 粒度 | 决策层 |
| --- | --- | --- | --- |
| `fresh_lt.yaml` | FreshRetailNet-LT | 日 | 不支持 |
| `fresh_50k.yaml` | FreshRetailNet-50K | 日 | 不支持 |
| `zhao.yaml` | Zhao SKU×月 | 月 | 单期代理（层 B） |
| `osa.yaml` | OSA-Data | 日 | 连续回放（层 C） |
| `lokad.yaml` | Lokad workshop | 周 | 静态 W#3 + 连续回放（层 C） |
| `mendeley.yaml` | Mendeley | 日 | 不支持 |

spec 01 中的 `configs/fresh.yaml` 按 `dataset_version` 解析为 `fresh_lt.yaml` 或 `fresh_50k.yaml`，二者**不可拼接**。

## 通用字段语义

- `origin`：决策时点。特征只能用**严格早于** origin 的数据。
- 目标窗：`[origin, origin + horizon - 1]` 闭区间。
- `splits.*.origins`：该 split 的 origin 闭区间，按 `calendar.origin_frequency` 枚举。
- `feature_history_start` vs `state_history_start`：前者是可用于构造滞后/滚动特征的最早日期，后者是库存等**状态**字段的最早可用日期。二者可以不同（Lokad 即如此）。
- `capabilities.*`：`false` 表示该数据集**结构性不支持**，对应的实验层与验收场景直接跳过，不记为 FAIL（见 `00_global_conventions.md` §9.3 退出码 2）。

## 一致性检查（实现方必须在加载时断言）

1. 每个 split 的 origin 区间不重叠，且 train < validation < test；
2. 最后一个 test origin 的目标窗末日 $\le$ 数据实际最大日期；
3. `metric.y_min` 与 `metric.y_min_bind_rate` 同时存在，且 bind_rate $\le$ 0.10；
4. `capabilities.decision_layer = false` 时，配置中不得出现成本参数。
