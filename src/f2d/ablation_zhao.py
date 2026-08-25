"""Zhao 特征消融。回答三个问题：

  A  baseline            当前 27 特征
  B  no_brand            去掉 brand_ID（3106 类，高基数）
  C  eta_pooled          ETA 误差退到 category 级池化（缺失率 96.3% -> ~0）
  D  stop_year_censored  stop_year 按 origin 截断 + 派生可解释特征（修泄漏）
  E  no_stop_year        直接去掉 stop_year
  F  C+D                 两处修正合并

用法:  PYTHONPATH=src python -m f2d.ablation_zhao
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .conventions import SEED_BASE, W_FLAT, W_VALUES
from .datasets import zhao
from .encoding import VocabStore
from .metrics import evaluate_slice
from .models.gbdt import QuantileGBDT
from .postprocess import postprocess
from .run_zhao import assign_weights

ART = cfgmod.ARTIFACT_DIR / "zhao"


# ---------------- 面板变换 ----------------

def tf_eta_pooled(panel: pd.DataFrame, raw: dict) -> pd.DataFrame:
    """ETA 误差改为 sku -> category 两层回退，并记录实际使用层级。

    与 00_global_conventions §6.3 校准器同构：最具体的可用层，且层级落表。
    """
    p = panel.copy()
    # orders 自带 category 列，先丢弃以免 merge 产生后缀冲突；
    # 统一使用 panel（来自 inventory 快照）的 category，保证与其他分组口径一致
    orders = raw["orders"].drop(columns=["category"], errors="ignore")
    o = orders.merge(
        p[["sku_ID", "category"]].drop_duplicates("sku_ID"), on="sku_ID", how="left")
    frames = []
    for m in sorted(p["month"].unique()):
        done = o[o["arrival_date"] < m]           # 只用 t 前已完成订单
        by_sku = done.groupby("sku_ID")["eta_error_days"].agg(
            sku_p50="median", sku_p85=lambda s: s.quantile(0.85), sku_n="size")
        by_cat = done.groupby("category")["eta_error_days"].agg(
            cat_p50="median", cat_p85=lambda s: s.quantile(0.85), cat_n="size")
        f = p.loc[p.month == m, ["sku_ID", "category", "month"]].merge(
            by_sku.reset_index(), on="sku_ID", how="left").merge(
            by_cat.reset_index(), on="category", how="left")
        use_sku = f["sku_n"].fillna(0) >= 5       # sku 级最小样本阈值
        f["eta_err_p50"] = np.where(use_sku, f["sku_p50"], f["cat_p50"])
        f["eta_err_p85"] = np.where(use_sku, f["sku_p85"], f["cat_p85"])
        f["eta_level"] = np.where(use_sku, "sku", np.where(f["cat_n"].notna(), "cat", "none"))
        frames.append(f[["sku_ID", "month", "eta_err_p50", "eta_err_p85", "eta_level"]])
    e = pd.concat(frames, ignore_index=True)
    p = p.drop(columns=["eta_err_p50", "eta_err_p85"]).merge(
        e, on=["sku_ID", "month"], how="left")
    return p


def tf_stop_year_censored(panel: pd.DataFrame, raw: dict) -> pd.DataFrame:
    """按 origin 截断 stop_year，修 562 个 stop_year==2019 的泄漏。

    在 origin t，只有 stop_year < year(t) 是可观察的；stop_year >= year(t)
    等于预知未来停产决定，必须屏蔽。同时派生两个可解释特征。
    """
    p = panel.copy()
    yr = p["month"].dt.year
    known = p["stop_year"] < yr                    # 严格早于当年才可观察
    p["stop_year_obs"] = np.where(known, p["stop_year"], np.nan)
    p["is_discontinued"] = known.astype(float)     # 截至 t 已停产
    p["years_since_stop"] = np.where(known, yr - p["stop_year"], np.nan)
    p["years_since_intro"] = np.where(
        p["introduction_year"] <= yr, yr - p["introduction_year"], np.nan)
    return p


# ---------------- 变体定义 ----------------

# 显式钉死消融基线的特征集，**不**从 zhao.FEATURES_NUM 动态取。
# 原因（实测踩过）：BASE_NUM = list(zhao.FEATURES_NUM) 在 import 时求值，
# 而 zhao.FEATURES_NUM 在主管线修 stop_year 泄漏后发生了变化 ——
# 于是同一份 ablation_paired.parquet 与之后重新 import 得到的 VARIANTS
# 不再对应，历史结果无法复现且差异不可归因。消融基线必须是冻结常量。
BASE_NUM_FROZEN_2026_08_22 = [
    "beginning_inventory", "on_order_inventory", "stock_value",
    "facing_number", "shelf_capacity",
    "sales_lag1", "sales_roll3", "sales_roll6", "sales_roll6_std", "months_observed",
    "recv_1m", "recv_3m", "recv_6m", "months_since_last_receipt",
    "open_po_due_this_month", "open_po_due_next_month", "open_po_overdue",
    "eta_err_p50", "eta_err_p85",
    "unit_cost_hist", "introduction_year", "stop_year",   # 含泄漏的原始列，作为对照基线
]
BASE_NUM = list(BASE_NUM_FROZEN_2026_08_22)
BASE_CAT = ["category", "subcategory", "unit", "brand_ID", "operation_mode"]

VARIANTS = {
    "A_baseline": dict(num=BASE_NUM, cat=BASE_CAT, tf=None),
    "B_no_brand": dict(num=BASE_NUM, cat=[c for c in BASE_CAT if c != "brand_ID"], tf=None),
    "C_eta_pooled": dict(num=BASE_NUM, cat=BASE_CAT, tf=tf_eta_pooled),
    "D_stop_censored": dict(
        num=[c for c in BASE_NUM if c != "stop_year"]
            + ["stop_year_obs", "is_discontinued", "years_since_stop", "years_since_intro"],
        cat=BASE_CAT, tf=tf_stop_year_censored),
    "E_no_stop_year": dict(num=[c for c in BASE_NUM if c != "stop_year"], cat=BASE_CAT, tf=None),
    "F_eta_pooled+stop_censored": dict(
        num=[c for c in BASE_NUM if c != "stop_year"]
            + ["stop_year_obs", "is_discontinued", "years_since_stop", "years_since_intro"],
        cat=BASE_CAT,
        tf=lambda p, r: tf_stop_year_censored(tf_eta_pooled(p, r), r)),
}


def run() -> pd.DataFrame:
    cfg = cfgmod.load("zhao")
    raw = zhao.load_raw()
    panel0, _ = zhao.build_panel(raw)

    tr_end = pd.Timestamp(cfg.origins("train")[-1])
    va = [pd.Timestamp(m) for m in cfg.origins("validation")]
    te = [pd.Timestamp(m) for m in cfg.origins("test")]
    panel0["split"] = np.where(panel0.month <= tr_end, "train",
                        np.where(panel0.month.isin(va), "validation",
                         np.where(panel0.month.isin(te), "test", "unused")))
    w, wsrc, _ = assign_weights(panel0, panel0.split == "train")
    panel0["w_i"], panel0["weight_source"] = w, wsrc
    y_min = float(cfg.metric["y_min"])
    months = sorted(panel0["month"].unique())

    out = []
    for name, spec in VARIANTS.items():
        t0 = time.time()
        p = spec["tf"](panel0, raw) if spec["tf"] else panel0.copy()
        vs = VocabStore.fit(p[p.split == "train"], spec["cat"], frozen_on=str(tr_end.date()))
        p = vs.transform(p)
        num = [c for c in spec["num"] if c in p.columns]

        preds = []
        for m in va + te:
            tr = p[p.month < m]
            cur = p[p.month == m]
            prev = months[months.index(m) - 1]
            fit_tr, fit_va = tr[tr.month < prev], tr[tr.month == prev]
            if not len(fit_va):
                fit_tr, fit_va = tr, None
            gb = QuantileGBDT(num, spec["cat"]).fit(
                fit_tr, "observed_sales_next_month", "w_i", fit_va)
            q50, q85 = gb.predict(cur)
            q50, q85, _, _ = postprocess(q50, q85)
            preds.append(pd.DataFrame({
                "split": cur["split"].to_numpy(), "y": cur["observed_sales_next_month"].to_numpy(),
                "q50": q50, "q85": q85, "w": cur["w_i"].to_numpy(),
                "cold": (cur["months_observed"].fillna(0) == 0).to_numpy()}))
        pr = pd.concat(preds, ignore_index=True)

        for split in ("validation", "test"):
            g = pr[pr.split == split]
            for sl, mask in {"overall": np.ones(len(g), bool), "cold_start": g.cold.to_numpy()}.items():
                s = g[mask]
                r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
                out.append(dict(variant=name, split=split, slice=sl, n=r.n,
                                npl=r.npl, cov50_pos=r.cov_50_pos, cov85_pos=r.cov_85_pos,
                                n_features=len(num) + len(spec["cat"])))
        print(f"  {name:<28} {time.time() - t0:5.1f}s  features={len(num) + len(spec['cat'])}")
    return pd.DataFrame(out)


def main() -> int:
    print("Zhao 特征消融")
    df = run()
    df.to_csv(ART / "ablation.csv", index=False)

    base = df[(df.variant == "A_baseline")].set_index(["split", "slice"]).npl
    df["delta_vs_A"] = df.apply(lambda r: r.npl - base.loc[(r.split, r.slice)], axis=1)

    print("\n整体 NPL（越低越好）与相对 baseline 的变化")
    for sl in ("overall", "cold_start"):
        print(f"\n--- slice = {sl} ---")
        t = df[df.slice == sl].pivot(index="variant", columns="split",
                                     values=["npl", "delta_vs_A"])
        print(t.round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------- 配对 bootstrap（§9.5）----------------

def run_paired(variants: list[str], out_name: str = "ablation_paired.parquet") -> pd.DataFrame:
    """对指定变体保存逐行预测，供配对 bootstrap 使用。"""
    cfg = cfgmod.load("zhao")
    raw = zhao.load_raw()
    panel0, _ = zhao.build_panel(raw)
    tr_end = pd.Timestamp(cfg.origins("train")[-1])
    va = [pd.Timestamp(m) for m in cfg.origins("validation")]
    te = [pd.Timestamp(m) for m in cfg.origins("test")]
    panel0["split"] = np.where(panel0.month <= tr_end, "train",
                        np.where(panel0.month.isin(va), "validation",
                         np.where(panel0.month.isin(te), "test", "unused")))
    w, wsrc, _ = assign_weights(panel0, panel0.split == "train")
    panel0["w_i"], panel0["weight_source"] = w, wsrc
    months = sorted(panel0["month"].unique())

    frames = []
    for name in variants:
        spec = VARIANTS[name]
        p = spec["tf"](panel0, raw) if spec["tf"] else panel0.copy()
        vs = VocabStore.fit(p[p.split == "train"], spec["cat"], frozen_on=str(tr_end.date()))
        p = vs.transform(p)
        num = [c for c in spec["num"] if c in p.columns]
        for m in va + te:
            tr, cur = p[p.month < m], p[p.month == m]
            prev = months[months.index(m) - 1]
            fit_tr, fit_va = tr[tr.month < prev], tr[tr.month == prev]
            if not len(fit_va):
                fit_tr, fit_va = tr, None
            gb = QuantileGBDT(num, spec["cat"]).fit(
                fit_tr, "observed_sales_next_month", "w_i", fit_va)
            q50, q85 = gb.predict(cur)
            q50, q85, _, _ = postprocess(q50, q85)
            frames.append(pd.DataFrame({
                "variant": name, "series_id": cur["series_id"].to_numpy(),
                "month": m, "split": cur["split"].to_numpy(),
                "y": cur["observed_sales_next_month"].to_numpy(),
                "q50": q50, "q85": q85, "w": cur["w_i"].to_numpy()}))
        print(f"  paired: {name} done")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(ART / out_name, index=False)
    return out
