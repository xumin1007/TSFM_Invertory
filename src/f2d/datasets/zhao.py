"""Zhao SKU×月 数据管线。对应 docs/implementation_specs/02_zhao_sku_month.md。

一个样本 = (sku_ID, month t)：在月初 t 的信息集下预测该自然月的
observed_sales_next_month。所有特征的 max_source_date 必须严格早于 month t。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARTIFACT_DIR, DATA_DIR

RAW = DATA_DIR / "external" / "Zhao"
ART = ARTIFACT_DIR / "zhao"
CACHE = ART / "_cache"

FILES = {
    "inventory": "nav21957-sup-0001-supinfo01.xlsx",
    "orders": "nav21957-sup-0002-supinfo02.xlsx",
    "sales": "nav21957-sup-0003-supinfo03.xlsx",
    "attrs": "nav21957-sup-0004-supinfo04.xlsx",
    "shelf": "nav21957-sup-0005-supinfo05.xlsx",
}

# 实测的列名变体（Feb/Oct 尾空格、Jun 拼写、Jul 拼写）
COLUMN_ALIASES = {
    "quantiity": "quantity",
    "sales_revenue ": "sales_revenue",
    "sales_reveue": "sales_revenue",
    "on-order_inventory": "on_order_inventory",
}

STATIC_COLS = ["category", "subcategory", "unit", "brand_ID", "operation_mode"]
CAT_COLS = ["category", "subcategory", "unit", "brand_ID", "operation_mode"]


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COLUMN_ALIASES.get(str(c).strip(), str(c).strip())
                              for c in df.columns})


def _month(s) -> pd.Series:
    """把 '01/2019' 或日期解析为月初 Timestamp。"""
    s = pd.Series(s)
    if s.dtype == object and s.astype(str).str.match(r"^\d{2}/\d{4}$").all():
        return pd.to_datetime(s, format="%m/%Y")
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def load_raw(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """读取五个原始文件，统一列名与月份，不改写原始文件。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    for key in FILES:
        cp = CACHE / f"{key}.parquet"
        if use_cache and cp.exists():
            out[key] = pd.read_parquet(cp)
            continue

        path = RAW / FILES[key]
        if key == "sales":
            frames = []
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                d = _norm_cols(pd.read_excel(xl, sheet_name=sheet))
                d["_sheet"] = sheet
                frames.append(d)
            d = pd.concat(frames, ignore_index=True)
            d["month"] = _month(d["date"])
        else:
            d = _norm_cols(pd.read_excel(path))
            if "date" in d.columns:
                d["month"] = _month(d["date"])
            if key == "attrs":
                # 实测：stop_year 以 ' '(空格) 表示缺失，dtype 为 object。
                # 强制转数值，空格与其他非数值一律成为 NaN 交给 GBDT 原生处理。
                for c in ("introduction_year", "stop_year"):
                    d[f"{c}_nonnumeric"] = (
                        d[c].notna() & pd.to_numeric(d[c], errors="coerce").isna())
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            if key == "orders":
                d["order_date"] = pd.to_datetime(d["order_date"])
                d["arrival_date"] = pd.to_datetime(d["arrival_date"])
                d["order_month"] = d["order_date"].dt.to_period("M").dt.to_timestamp()
                d["arrival_month"] = d["arrival_date"].dt.to_period("M").dt.to_timestamp()
                # 承诺 ETA 严格由 order_date + normal_lead_time 推算，与实际到货分开
                d["promised_eta"] = d["order_date"] + pd.to_timedelta(
                    d["normal_lead_time"].fillna(0), unit="D")
                d["promised_eta_month"] = d["promised_eta"].dt.to_period("M").dt.to_timestamp()
                d["eta_error_days"] = (d["arrival_date"] - d["promised_eta"]).dt.days
        d.to_parquet(cp, index=False)
        out[key] = d
    return out


def raw_audit(raw: dict[str, pd.DataFrame]) -> dict:
    """步骤 1：原始审计。"""
    a: dict = {"tables": {}}
    for k, d in raw.items():
        info = {
            "rows": int(len(d)),
            "columns": list(map(str, d.columns)),
            "n_sku": int(d["sku_ID"].nunique()) if "sku_ID" in d else None,
            "missing_rate": {c: round(float(d[c].isna().mean()), 6) for c in d.columns},
        }
        if "month" in d:
            info["month_range"] = [str(d["month"].min().date()), str(d["month"].max().date())]
        a["tables"][k] = info

    inv = raw["inventory"]
    a["inventory_pk_duplicates"] = int(inv.duplicated(["sku_ID", "month"]).sum())
    sales = raw["sales"]
    a["sales_negative_qty"] = int((sales["quantity"] < 0).sum())
    o = raw["orders"]
    a["orders"] = {
        "n": int(len(o)),
        "eta_on_time": round(float((o["eta_error_days"] == 0).mean()), 4),
        "eta_early": round(float((o["eta_error_days"] < 0).mean()), 4),
        "eta_late": round(float((o["eta_error_days"] > 0).mean()), 4),
        "normal_lead_time_median": float(o["normal_lead_time"].median()),
        "arrival_after_order_violations": int((o["arrival_date"] < o["order_date"]).sum()),
    }
    return a


def build_panel(raw: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """步骤 2-3：月度销售聚合 + 事前特征面板。

    返回 (panel, feature_lineage)。panel 主键 (sku_ID, month)，目标为该月观测销量。
    """
    inv = raw["inventory"].copy()
    sales = raw["sales"]
    orders = raw["orders"]
    attrs = raw["attrs"]
    shelf = raw["shelf"]

    # --- 目标：月度销量。仅当该 (sku, month) 存在于月初库存快照时补 0 ---
    sm = (sales.groupby(["sku_ID", "month"], as_index=False)
                .agg(observed_sales_next_month=("quantity", "sum"),
                     sales_rows=("quantity", "size")))

    panel = inv[["sku_ID", "month", "beginning_inventory", "on_order_inventory", "stock_value"]].copy()
    panel = panel.merge(sm, on=["sku_ID", "month"], how="left")
    panel["target_was_zero_filled"] = panel["observed_sales_next_month"].isna()
    panel["observed_sales_next_month"] = panel["observed_sales_next_month"].fillna(0.0)
    panel["sales_rows"] = panel["sales_rows"].fillna(0).astype(int)

    # --- 静态属性 + 月度陈列 ---
    panel = panel.merge(
        attrs[["sku_ID", "brand_ID", "introduction_year", "stop_year", "operation_mode"]]
        .drop_duplicates("sku_ID"), on="sku_ID", how="left")
    panel = panel.merge(
        inv[["sku_ID", "month", "category", "subcategory", "unit"]]
        .drop_duplicates(["sku_ID", "month"]), on=["sku_ID", "month"], how="left")
    panel = panel.merge(
        shelf[["sku_ID", "month", "facing_number", "shelf_capacity"]]
        .drop_duplicates(["sku_ID", "month"]), on=["sku_ID", "month"], how="left")

    # --- 销售滞后/滚动：只用 t-1 及更早 ---
    # 从 panel 的目标构造（而非原始 sales），使滞后与目标的补零口径一致：
    # 存在于月初库存快照 => 该月需求为 0；不存在 => 该 (sku, month) 根本没有行。
    wide = (panel.pivot_table(index="sku_ID", columns="month",
                              values="observed_sales_next_month")
                 .sort_index(axis=1))
    months = list(wide.columns)
    lag_frames = []
    for i, m in enumerate(months):
        past = wide.iloc[:, :i]  # 严格早于 m
        f = pd.DataFrame(index=wide.index)
        f["sales_lag1"] = past.iloc[:, -1] if i >= 1 else np.nan
        f["sales_roll3"] = past.iloc[:, -3:].mean(axis=1) if i >= 1 else np.nan
        f["sales_roll6"] = past.iloc[:, -6:].mean(axis=1) if i >= 1 else np.nan
        f["sales_roll6_std"] = past.iloc[:, -6:].std(axis=1) if i >= 2 else np.nan
        # 该 SKU 在 m 之前实际被观测到的月数（NaN 表示当时尚未进入快照）
        f["months_observed"] = past.notna().sum(axis=1).astype(float) if i >= 1 else 0.0
        f["month"] = m
        lag_frames.append(f.reset_index())
    panel = panel.merge(pd.concat(lag_frames, ignore_index=True), on=["sku_ID", "month"], how="left")

    # --- 实际收货：只用 arrival_date < month ---
    recv = (orders.groupby(["sku_ID", "arrival_month"], as_index=False)
                  .agg(recv_qty=("quantity", "sum")))
    rw = recv.pivot_table(index="sku_ID", columns="arrival_month",
                          values="recv_qty", fill_value=0.0).sort_index(axis=1)
    rmonths = list(rw.columns)
    rec_frames = []
    for m in months:
        prior = [c for c in rmonths if c < m]
        f = pd.DataFrame(index=rw.index)
        p = rw[prior] if prior else rw.iloc[:, :0]
        f["recv_1m"] = p.iloc[:, -1] if p.shape[1] >= 1 else 0.0
        f["recv_3m"] = p.iloc[:, -3:].sum(axis=1) if p.shape[1] >= 1 else 0.0
        f["recv_6m"] = p.iloc[:, -6:].sum(axis=1) if p.shape[1] >= 1 else 0.0
        nz = (p > 0)
        f["months_since_last_receipt"] = (
            nz.shape[1] - nz.values.argmax(axis=1) if p.shape[1] else np.nan)
        f.loc[~nz.any(axis=1) if p.shape[1] else slice(None), "months_since_last_receipt"] = 99.0
        f["month"] = m
        rec_frames.append(f.reset_index())
    panel = panel.merge(pd.concat(rec_frames, ignore_index=True), on=["sku_ID", "month"], how="left")

    # --- 在订与 ETA：只用 order_date < month 且截至 month 尚未到货的订单 ---
    # 注意 arrival_date >= m 不是未来信息：它等价于"截至 m 尚未到货"，
    # 该状态在 m 时点可观察（§2.4.1）。未来的具体到货日期本身从不进入特征。
    next_month = {m: (m + pd.offsets.MonthBegin(1)) for m in months}
    eta_frames = []
    for m in months:
        open_po = orders[(orders["order_date"] < m) & (orders["arrival_date"] >= m)]
        g = open_po.groupby("sku_ID")
        f = pd.DataFrame({
            # 重建总量：仅供与快照对账，不进 FEATURES_NUM（§2.4.1）
            "open_po_qty_rebuilt": g["quantity"].sum(),
            # ETA 分桶：按承诺 ETA（order_date + normal_lead_time），是允许的特征
            "open_po_due_this_month": open_po[open_po["promised_eta_month"] == m]
                                        .groupby("sku_ID")["quantity"].sum(),
            "open_po_due_next_month": open_po[open_po["promised_eta_month"] == next_month[m]]
                                        .groupby("sku_ID")["quantity"].sum(),
            "open_po_overdue": open_po[open_po["promised_eta"] < m]
                                        .groupby("sku_ID")["quantity"].sum(),
        })
        # ETA 误差：只用 month 前已完成的订单
        done = orders[orders["arrival_date"] < m]
        eg = done.groupby("sku_ID")["eta_error_days"]
        f["eta_err_p50"] = eg.median()
        f["eta_err_p85"] = eg.quantile(0.85)
        f["month"] = m
        eta_frames.append(f.reset_index())
    eta = pd.concat(eta_frames, ignore_index=True)
    for c in ["open_po_qty_rebuilt", "open_po_due_this_month",
              "open_po_due_next_month", "open_po_overdue"]:
        eta[c] = eta[c].fillna(0.0)
    panel = panel.merge(eta, on=["sku_ID", "month"], how="left")

    # --- 单位成本（用于 w_i 与决策层），只用 order_date < month 的订单 ---
    cost_frames = []
    for m in months:
        past_o = orders[orders["order_date"] < m]
        c = past_o.groupby("sku_ID")["unit_cost"].median().rename("unit_cost_hist")
        cost_frames.append(c.reset_index().assign(month=m))
    panel = panel.merge(pd.concat(cost_frames, ignore_index=True),
                        on=["sku_ID", "month"], how="left")

    # --- 静态事件日期属性按 origin 截断（spec 02 §2）---
    # stop_year 编码停产**事件年份**，不是恒久属性。实测 562 个 SKU 的
    # stop_year=2019（另 140 个 >2019）落在数据期内；在 2019 年任一 origin 上
    # "该 SKU 将于 2019 停产"不可观察。面板 3,653 行(1.5%)受影响，且该值与
    # 目标强相关（已停产行均值 2.4 vs 未停产 43.3），属会抬高分数的真实泄漏。
    _yr = panel["month"].dt.year
    _known = panel["stop_year"] < _yr          # 严格早于当年才可观察
    panel["stop_year_obs"] = np.where(_known, panel["stop_year"], np.nan)
    panel["is_discontinued"] = _known.astype(float)
    panel["years_since_stop"] = np.where(_known, _yr - panel["stop_year"], np.nan)
    panel["years_since_intro"] = np.where(
        panel["introduction_year"] <= _yr, _yr - panel["introduction_year"], np.nan)
    panel.attrs["stop_year_censored_rows"] = int(
        ((panel["stop_year"] >= _yr) & panel["stop_year"].notna()).sum())

    # --- 结构性零 vs 真缺失（§8.2）---
    # "无在订单"/"无收货记录" 是可确证的零，不是缺失；补 0。
    # "无历史订单故无 ETA 误差估计"/"无历史成本" 是真缺失；保持 NaN 交给 GBDT 原生处理。
    STRUCTURAL_ZERO = ["recv_1m", "recv_3m", "recv_6m", "open_po_qty_rebuilt",
                       "open_po_due_this_month", "open_po_due_next_month", "open_po_overdue"]
    for c in STRUCTURAL_ZERO:
        panel[c] = panel[c].fillna(0.0)
    # 99 = 从无入库（与阿里题 days_since_last_receipt=90 的哨兵语义一致）
    panel["months_since_last_receipt"] = panel["months_since_last_receipt"].fillna(99.0)

    # §2.4.1：快照为准，重建量只报差异，绝不静默替代
    d = panel["open_po_qty_rebuilt"] - panel["on_order_inventory"]
    panel.attrs["open_po_snapshot_vs_rebuild"] = {
        "n": int(len(d)),
        "exact_match_share": round(float((d == 0).mean()), 4),
        "rebuild_gt_snapshot_share": round(float((d > 0).mean()), 4),
        "abs_diff_median": float(d.abs().median()),
        "abs_diff_p95": float(d.abs().quantile(0.95)),
    }

    panel = panel.sort_values(["month", "sku_ID"]).reset_index(drop=True)
    panel["series_id"] = panel["sku_ID"].astype(str)

    lineage = _lineage()
    return panel, lineage


def _lineage() -> pd.DataFrame:
    rows = [
        ("beginning_inventory", "inventory", "月初快照，t 时点已知", "raw", "t"),
        ("on_order_inventory", "inventory", "月初快照（首选在订量）", "raw", "t"),
        ("stock_value", "inventory", "月初快照", "raw", "t"),
        ("facing_number", "shelf", "月初陈列", "raw", "t"),
        ("shelf_capacity", "shelf", "月初陈列", "raw", "t"),
        ("sales_lag1", "sales", "仅 t-1 及更早", "lag", "t-1"),
        ("sales_roll3", "sales", "仅 t-1 及更早", "rolling_mean_3", "t-1"),
        ("sales_roll6", "sales", "仅 t-1 及更早", "rolling_mean_6", "t-1"),
        ("sales_roll6_std", "sales", "仅 t-1 及更早", "rolling_std_6", "t-1"),
        ("months_observed", "inventory", "t 之前出现在月初快照的月数", "count", "t-1"),
        ("stop_year_obs", "attrs", "stop_year < year(origin) 才可观察", "censor_at_origin", "t-1"),
        ("is_discontinued", "attrs", "截至 origin 是否已停产", "censor_at_origin", "t-1"),
        ("years_since_stop", "attrs", "origin 年 - stop_year，仅可观察者", "censor_at_origin", "t-1"),
        ("years_since_intro", "attrs", "origin 年 - introduction_year", "censor_at_origin", "t-1"),
        ("recv_1m", "orders", "arrival_date < t", "sum", "t-1"),
        ("recv_3m", "orders", "arrival_date < t", "sum", "t-1"),
        ("recv_6m", "orders", "arrival_date < t", "sum", "t-1"),
        ("months_since_last_receipt", "orders", "arrival_date < t", "gap", "t-1"),
        ("open_po_due_this_month", "orders", "承诺 ETA 落在 t 月", "sum", "t-1"),
        ("open_po_due_next_month", "orders", "承诺 ETA 落在 t+1 月", "sum", "t-1"),
        ("open_po_overdue", "orders", "承诺 ETA 早于 t", "sum", "t-1"),
        ("eta_err_p50", "orders", "仅 arrival_date < t 的已完成订单", "median", "t-1"),
        ("eta_err_p85", "orders", "仅 arrival_date < t 的已完成订单", "q85", "t-1"),
        ("unit_cost_hist", "orders", "order_date < t", "median", "t-1"),
    ]
    return pd.DataFrame(rows, columns=["feature", "source_table", "availability_rule",
                                       "transformation", "max_source_month"])


FEATURES_NUM = [
    "beginning_inventory", "on_order_inventory", "stock_value",
    "facing_number", "shelf_capacity",
    "sales_lag1", "sales_roll3", "sales_roll6", "sales_roll6_std", "months_observed",
    "recv_1m", "recv_3m", "recv_6m", "months_since_last_receipt",
    "open_po_due_this_month", "open_po_due_next_month", "open_po_overdue",
    "eta_err_p50", "eta_err_p85",
    "unit_cost_hist",
    # 静态事件日期属性：原始 stop_year 已按 origin 截断，不得直接使用
    "stop_year_obs", "is_discontinued", "years_since_stop", "years_since_intro",
]


def build_margin_block(raw: dict[str, pd.DataFrame],
                       months: list) -> pd.DataFrame:
    """逐 (sku_ID, month) 的售价、进价与单位毛利。**严格只用 month 之前的记录。**

    动机：`07_decision_layer.md` §5.3 原写「数据未提供短缺罚金」，故 $p_i$ 只能
    由 $\alpha$ 导出。该前提有误 —— `sales.original_unit_price` 提供售价，
    `orders.unit_cost` 提供进价，缺货的真实经济后果（损失毛利）**可由数据算出**。

    实测：售价逐 SKU 恒定（变异系数中位 0.000），缺失率 0，非正值占比 0。

    回退按 §6 的 L0/L1/L2 三级：本 SKU -> subcategory -> 全局，逐级记录命中层。
    回退是必要的 —— origin 前无销售记录的 SKU 拿不到售价，而那正是需求稀疏
    的冷启动品，不能直接丢弃。
    """
    sales, orders = raw["sales"], raw["orders"]
    sub_of = dict(zip(sales["sku_ID"], sales["subcategory"]))

    frames = []
    for m in months:
        ps = sales[sales["date"] < m]
        oc = orders[orders["order_date"] < m]
        price0 = ps.groupby("sku_ID")["original_unit_price"].median()
        cost0 = oc.groupby("sku_ID")["unit_cost"].median()
        price1 = ps.groupby("subcategory")["original_unit_price"].median()
        cost1 = oc.groupby("subcategory")["unit_cost"].median()
        price2 = float(ps["original_unit_price"].median())
        cost2 = float(oc["unit_cost"].median())

        skus = np.asarray(sorted(set(price0.index) | set(cost0.index)))
        subs = pd.Series([sub_of.get(k) for k in skus], index=skus)

        def _fill(l0, l1, glob):
            v = l0.reindex(skus)
            lvl = pd.Series(np.where(v.notna(), "L0", ""), index=skus)
            f1 = subs.map(l1)
            take1 = v.isna() & f1.notna()
            v = v.where(~take1, f1)
            lvl[take1] = "L1"
            take2 = v.isna()
            v = v.fillna(glob)
            lvl[take2] = "L2"
            return v, lvl

        pv, pl = _fill(price0, price1, price2)
        cv, cl = _fill(cost0, cost1, cost2)
        frames.append(pd.DataFrame({
            "sku_ID": skus, "month": m,
            "price_hist": pv.to_numpy(), "cost_hist": cv.to_numpy(),
            "price_level": pl.to_numpy(), "cost_level": cl.to_numpy()}))

    out = pd.concat(frames, ignore_index=True)
    # 单位毛利。0.9% 的 SKU 售价低于进价（促销/清仓），截断为 0：负的缺货
    # 罚金会让"故意缺货"变成最优，那是数据噪声而非业务事实。
    out["margin_unit"] = np.clip(out["price_hist"] - out["cost_hist"], 0.0, None)
    return out


def write_audit(audit: dict, path: Path | None = None) -> Path:
    p = path or (ART / "raw_audit.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


# ===========================================================================
# 日级路径（方案 A）
#
# 与月级路径的关系：月级面板用于**决策层**（月初库存快照是月度的）；
# 日级面板用于**预测层**，因为 TSFM 在月度上下文（最多 10 点）上不可用。
# 日级预测经 f2d.aggregation 聚合回月度，供决策层使用。
#
# 三条已实测确立的构造规则：
#   1. 销售表是**交易日志**（1,439,480 行中仅 1,618 行 quantity==0，占 0.11%），
#      故「无记录日 = 零销量」可辩护，在生命期内补零。
#   2. 补零**只在生命期内**（首末交易之间）。首次交易前未上架、末次交易后已
#      下架，补零等于凭空造数据。
#   3. 聚合到周/月时只保留**完整**周/月，不完整者排除并记 reason code。
# ===========================================================================

MIN_LIFETIME_DAYS = 100          # 生命期筛选阈值；实测 80.2% 的 SKU 达标


def build_daily_panel(raw: dict[str, pd.DataFrame],
                      min_lifetime_days: int = MIN_LIFETIME_DAYS
                      ) -> tuple[pd.DataFrame, dict]:
    """日级稠密面板。返回 (panel, audit)。

    panel 列：sku_ID / series_id / d / y / lifetime_days
    audit 记录筛选与补零的规模，供验收 JSON 引用。
    """
    sales = raw["sales"].copy()
    sales["d"] = pd.to_datetime(sales["date"])
    daily = sales.groupby(["sku_ID", "d"], as_index=False)["quantity"].sum()

    life = daily.groupby("sku_ID").d.agg(["min", "max"])
    life["days"] = (life["max"] - life["min"]).dt.days + 1
    keep = life[life.days >= min_lifetime_days].index

    audit = {
        "n_sku_total": int(len(life)),
        "n_sku_kept": int(len(keep)),
        "kept_share": round(float(len(keep) / max(len(life), 1)), 4),
        "min_lifetime_days": min_lifetime_days,
        "lifetime_quantiles": {q: float(life.days.quantile(q))
                               for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "zero_qty_rows_in_sales": int((sales["quantity"] == 0).sum()),
        "zero_qty_share": round(float((sales["quantity"] == 0).mean()), 6),
        "zero_fill_justification":
            "销售表为交易日志（零值占 %.4f%%），故无记录日视为零销量；"
            "补零仅限生命期内" % (100 * float((sales["quantity"] == 0).mean())),
    }
    if audit["zero_qty_share"] > 0.01:
        raise ValueError(
            f"销售表零值占比 {audit['zero_qty_share']:.4f} > 1%，"
            "不能视为交易日志；「无记录=零销量」的前提不成立（§8.2）")

    daily = daily[daily.sku_ID.isin(keep)]
    grid = pd.concat(
        [pd.DataFrame({"sku_ID": sid, "d": pd.date_range(a, b)})
         for sid, (a, b, _) in life.loc[keep].iterrows()], ignore_index=True)
    panel = grid.merge(daily, on=["sku_ID", "d"], how="left")
    panel["y"] = panel["quantity"].fillna(0.0)
    panel = panel.drop(columns=["quantity"])
    panel["series_id"] = panel["sku_ID"].astype(str)
    panel = panel.merge(life.loc[keep, ["days"]].rename(columns={"days": "lifetime_days"}),
                        left_on="sku_ID", right_index=True, how="left")
    panel = panel.sort_values(["sku_ID", "d"]).reset_index(drop=True)

    audit["n_rows"] = int(len(panel))
    audit["zero_share_after_fill"] = round(float((panel.y == 0).mean()), 4)
    return panel, audit


def aggregate_to_period(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """把日级面板聚合到周（`"W"`，周一起点）或自然月（`"M"`）。

    **只保留完整周期**：周需 7 天、月需该月全部日历日均被生命期覆盖。
    不完整者排除并在返回值的 attrs 中记 n_excluded_incomplete。
    """
    d = daily.copy()
    if freq == "W":
        d["origin"] = d.d - pd.to_timedelta(d.d.dt.weekday, unit="D")
        need = pd.Series(7, index=d.index)
    elif freq == "M":
        d["origin"] = d.d.values.astype("datetime64[M]")
        need = d.d.dt.days_in_month
    else:
        raise ValueError(f"freq 只支持 'W' / 'M'，收到 {freq!r}")

    g = (d.assign(_need=need)
          .groupby(["sku_ID", "series_id", "origin"], as_index=False)
          .agg(y=("y", "sum"), n_days=("y", "size"), need=("_need", "first")))
    complete = g[g.n_days == g.need].drop(columns=["need"]).reset_index(drop=True)
    complete.attrs["n_excluded_incomplete"] = int(len(g) - len(complete))
    complete.attrs["freq"] = freq
    return complete.sort_values(["sku_ID", "origin"]).reset_index(drop=True)


def add_lag_features(panel: pd.DataFrame, lags: tuple[int, ...] = (1,),
                     rolls: tuple[int, ...] = (4, 8),
                     value_col: str = "y", group_col: str = "sku_ID") -> pd.DataFrame:
    """严格只用 origin 之前的期。shift(1) 保证不含当期。"""
    out = panel.copy()
    g = out.groupby(group_col)[value_col]
    for k in lags:
        out[f"lag{k}"] = g.shift(k)
    shifted = g.shift(1)
    for w in rolls:
        out[f"roll{w}"] = (shifted.rolling(w, min_periods=1)
                           .mean().reset_index(level=0, drop=True))
    out["periods_observed"] = g.transform(lambda s: np.arange(len(s), dtype=float))
    return out


def empirical_quantile_grid(daily: pd.DataFrame, levels: np.ndarray,
                            up_to: pd.Timestamp,
                            min_obs: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """每条序列在 `up_to` **之前**的日级经验分位网格。返回 (series_ids, grid)。

    必须用 `method="inverted_cdf"`：默认线性插值会在序统计量之间产生分数值，
    把零原子抹平（实测使 q50 MAE 由 0.000 恶化到 0.290）。
    """
    hist = daily[daily.d < up_to]
    keep = hist.groupby("series_id").size()
    keep = keep[keep >= min_obs].index
    hist = hist[hist.series_id.isin(keep)]
    sids, grids = [], []
    for sid, sub in hist.groupby("series_id"):
        sids.append(sid)
        grids.append(np.quantile(sub.y.values, levels, method="inverted_cdf"))
    return np.array(sids), np.asarray(grids, float)
