"""Zhao 端到端：面板 -> 权重 -> 词表 -> 逐 origin 预测 -> 后处理 -> 指标 -> 验收。

用法:  PYTHONPATH=src python -m f2d.run_zhao [--no-cache]

prequential：预测第 m 月时，训练集为全部 month < m 的行（expanding）。
这与 configs/zhao.yaml 的 splits 与 prequential_rule 一致。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfgmod
from .calibration import ResidualCalibrator
from .checks import CheckResult
from .conventions import EXIT_FAIL, EXIT_PASS, SEED_BASE, W_FLAT, W_VALUES
from .datasets import zhao
from .encoding import VocabStore
from .metrics import evaluate_slice
from .models.baselines import REGISTRY as BASELINES
from .models.gbdt import QuantileGBDT
from .postprocess import postprocess

ART = cfgmod.ARTIFACT_DIR / "zhao"


def assign_weights(panel: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.Series, pd.Series, list]:
    """§3.2：unit_cost 训练窗三分位，切点冻结后原样应用到验证/测试。"""
    cost = panel.loc[train_mask, "unit_cost_hist"].dropna()
    edges = list(np.quantile(cost, [1 / 3, 2 / 3])) if len(cost) else [np.nan, np.nan]
    c = panel["unit_cost_hist"]
    w = pd.Series(W_FLAT, index=panel.index, dtype=float)
    src = pd.Series("default_missing_cost", index=panel.index, dtype=object)
    known = c.notna()
    w[known & (c <= edges[0])] = W_VALUES["low"]
    w[known & (c > edges[0]) & (c <= edges[1])] = W_VALUES["mid"]
    w[known & (c > edges[1])] = W_VALUES["high"]
    src[known] = "tercile"
    return w, src, edges


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    chk = CheckResult(step_id="zhao_end_to_end", dataset="zhao", seed=SEED_BASE)
    cfg = cfgmod.load("zhao")
    ART.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 面板 ----------
    raw = zhao.load_raw(use_cache=not args.no_cache)
    audit = zhao.raw_audit(raw)
    zhao.write_audit(audit)
    panel, lineage = zhao.build_panel(raw)
    lineage.to_csv(ART / "feature_lineage.csv", index=False)
    panel.to_parquet(ART / "panel_monthly.parquet", index=False)

    chk.assert_true("面板主键唯一", not panel.duplicated(["sku_ID", "month"]).any())
    chk.assert_true("订单到货不早于下单",
                    audit["orders"]["arrival_after_order_violations"] == 0)

    months = sorted(panel["month"].unique())
    tr_end = pd.Timestamp(cfg.origins("train")[-1])
    va = [pd.Timestamp(m) for m in cfg.origins("validation")]
    te = [pd.Timestamp(m) for m in cfg.origins("test")]
    panel["split"] = np.where(panel.month <= tr_end, "train",
                       np.where(panel.month.isin(va), "validation",
                        np.where(panel.month.isin(te), "test", "unused")))

    # ---------- 2. 权重（训练窗冻结） ----------
    w, wsrc, edges = assign_weights(panel, panel.split == "train")
    panel["w_i"], panel["weight_source"] = w, wsrc
    chk.note("w_edges", [round(float(e), 4) for e in edges])
    chk.note("w_default_share", round(float((wsrc == "default_missing_cost").mean()), 4))

    # ---------- 3. 词表（训练窗冻结） ----------
    vs = VocabStore.fit(panel[panel.split == "train"], zhao.CAT_COLS, frozen_on=str(tr_end.date()))
    vs.save(ART / "vocab.json")
    panel = vs.transform(panel)
    chk.note("unk_rate_test", {k: round(v, 4) for k, v in vs.unk_rate(panel[panel.split == "test"]).items()})

    y_min = float(cfg.metric["y_min"])
    num_f = [c for c in zhao.FEATURES_NUM if c in panel.columns]
    rows = []

    # ---------- 4. 逐 origin 预测 ----------
    for m in va + te:
        tr = panel[panel.month < m]
        cur = panel[panel.month == m]
        if not len(tr) or not len(cur):
            continue
        # early stopping 用最近一个已实现月，不触碰 m 本身
        prev = months[months.index(m) - 1]
        fit_tr, fit_va = tr[tr.month < prev], tr[tr.month == prev]
        if not len(fit_va):
            fit_tr, fit_va = tr, None

        gb = QuantileGBDT(num_f, zhao.CAT_COLS).fit(
            fit_tr, "observed_sales_next_month", "w_i", fit_va)
        q50, q85 = gb.predict(cur)
        rows.append(_pack(cur, m, "lgbm", q50, q85, "native", None, gb.best_iters))

        # 点预测基线 + 残差校准器
        for mid, bl in BASELINES.items():
            resid = tr.assign(residual=tr["observed_sales_next_month"] - bl.predict(tr),
                              target_end=tr["month"], horizon=1)
            cal = ResidualCalibrator().fit(resid, origin=m, horizon=1,
                                           series_col="series_id", cat_col="category")
            c50, c85, lvl = cal.predict(bl.predict(cur), cur["series_id"], cur["category"])
            rows.append(_pack(cur, m, mid, c50, c85, "calibrated", lvl, cal.level_sizes))

    pred = pd.concat(rows, ignore_index=True)
    pred.to_parquet(ART / "predictions.parquet", index=False)

    chk.assert_true("预测非负", bool((pred[["q50", "q85"]] >= 0).all().all()))
    chk.assert_true("q50 <= q85", bool((pred.q50 <= pred.q85).all()))
    chk.assert_true("预测有限", bool(np.isfinite(pred[["q50", "q85"]].to_numpy()).all()))
    n_expect = len(BASELINES) + 1
    chk.assert_true("每 origin×model 主键完整",
                    not pred.duplicated(["series_id", "origin", "model_id"]).any()
                    and pred.groupby("origin").model_id.nunique().eq(n_expect).all())

    # ---------- 5. 指标 ----------
    mrows = []
    for (mid, split), g in pred.groupby(["model_id", "split"]):
        for sl, mask in _slices(g).items():
            sub = g[mask]
            r = evaluate_slice(sub.y_obs, sub.q50, sub.q85, sub.w_i, y_min).as_dict()
            mrows.append({"model_id": mid, "split": split, "slice": sl, **r})
    met = pd.DataFrame(mrows)
    met.to_csv(ART / "metrics_prediction.csv", index=False)

    chk.n_rows = len(pred)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.note("calib_level_share",
             pred[pred.calib_level.notna()].calib_level.value_counts(normalize=True).round(4).to_dict())
    chk.finish(ART / "checks" / "zhao_end_to_end.json")

    _report(met, chk)
    return EXIT_PASS if chk.status == "PASS" else EXIT_FAIL


def _pack(cur, m, model_id, q50, q85, qsrc, level, extra) -> pd.DataFrame:
    q50, q85, bad, crossing = postprocess(q50, q85)
    return pd.DataFrame({
        "dataset": "zhao", "series_id": cur["series_id"].to_numpy(),
        "origin": m, "target_start": m, "target_end": m,
        "split": cur["split"].to_numpy(), "model_id": model_id,
        "run_id": f"{model_id}@{m:%Y-%m}", "seed": SEED_BASE,
        "q50": q50, "q85": q85,
        "y_obs": cur["observed_sales_next_month"].to_numpy(),
        "w_i": cur["w_i"].to_numpy(), "weight_source": cur["weight_source"].to_numpy(),
        "quantile_source": qsrc,
        "calib_level": level if level is not None else pd.NA,
        "fallback_level": np.where(bad, "F2", "none"),
        "is_cold_start": (cur["months_observed"].fillna(0) == 0).to_numpy(),
        "crossing_rate": crossing,
    })


def _slices(g: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "overall": pd.Series(True, index=g.index),
        "cold_start": g.is_cold_start,
        "critical_w": g.w_i >= W_VALUES["high"],
        "zero_target": g.y_obs == 0,
    }


def _report(met: pd.DataFrame, chk: CheckResult) -> None:
    print(f"\n{'=' * 74}\nZhao 端到端  status={chk.status}\n{'=' * 74}")
    piv = (met[met.slice == "overall"]
           .pivot(index="model_id", columns="split", values="npl")
           .reindex(columns=["validation", "test"]))
    print("\n整体 NPL(q50,q85)  越低越好")
    print(piv.round(5).to_string())
    print("\n测试窗切片 NPL")
    t = met[met.split == "test"].pivot(index="model_id", columns="slice", values="npl")
    print(t.round(5).to_string())
    print("\n测试窗覆盖率（目标 .50 / .85）")
    cols = ["model_id", "cov_50", "cov_85", "cov_50_pos", "cov_85_pos", "zero_share", "n"]
    c = met[(met.split == "test") & (met.slice == "overall")][cols]
    print(c.round(4).to_string(index=False))
    print("  注：zero_share>0.10 时以 cov_*_pos 判校准（00_global_conventions.md §5.1）")


if __name__ == "__main__":
    sys.exit(main())
