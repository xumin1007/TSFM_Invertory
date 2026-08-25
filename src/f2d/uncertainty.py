"""按 series 聚类的配对 bootstrap。对应 docs/00_global_conventions.md §9.5。

模型比较必须报 ΔNPL 的 CI，而不是比较各自 CI 是否重叠，也不是只报点估计。
配对：同一组重抽样索引施于所有被比较的变体。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conventions import BOOTSTRAP_B, BOOTSTRAP_CI, SEED_BASE
from .metrics import npl


@dataclass
class DeltaCI:
    baseline: str
    variant: str
    split: str
    delta: float
    lo: float
    hi: float
    n_series: int
    n_rows: int
    b: int
    slice_id: str = "overall"

    @property
    def significant(self) -> bool:
        """CI 不跨 0 才算显著。"""
        return (self.lo > 0) == (self.hi > 0)

    def as_dict(self) -> dict:
        return {**self.__dict__, "significant": self.significant}


def paired_bootstrap(
    preds: pd.DataFrame,
    baseline: str,
    variants: list[str],
    y_min: float,
    split: str | None = None,
    b: int = BOOTSTRAP_B,
    ci: float = BOOTSTRAP_CI,
    seed: int = SEED_BASE,
    variant_col: str = "variant",
    series_col: str = "series_id",
    slice_col: str | None = None,
    slice_id: str = "overall",
) -> list[DeltaCI]:
    """preds 需含 variant/series_id/y/q50/q85/w，且各变体行序一致。

    行序一致性会被断言——不一致时配对失效，结果无意义。
    slice_col 给定时，只在该布尔列为 True 的行上计算（切片级 bootstrap：
    在切片内部按 series 重抽样，与 NPL 不可分解的性质一致，§4.4）。
    """
    if split is not None:
        preds = preds[preds["split"] == split]
    if slice_col is not None:
        preds = preds[preds[slice_col].astype(bool)]
        if preds.empty:
            raise ValueError(f"切片 {slice_id} 为空")

    frames = {v: g.sort_values([series_col, "month"]).reset_index(drop=True)
              for v, g in preds.groupby(variant_col)}
    missing = [v for v in [baseline, *variants] if v not in frames]
    if missing:
        raise KeyError(f"缺少变体: {missing}")

    key = frames[baseline][[series_col, "month"]].to_numpy()
    for v in variants:
        if not np.array_equal(frames[v][[series_col, "month"]].to_numpy(), key):
            raise ValueError(f"变体 {v} 与基线行序不一致，配对 bootstrap 失效")

    def _npl(g: pd.DataFrame) -> float:
        return npl(g["y"].to_numpy(), g["q50"].to_numpy(),
                   g["q85"].to_numpy(), g["w"].to_numpy(), y_min)[0]

    groups = frames[baseline].groupby(series_col).indices
    series = np.array(list(groups))
    rng = np.random.default_rng(seed)

    # 预抽样，保证所有变体共享同一组索引
    row_sets = []
    for _ in range(b):
        samp = rng.choice(series, size=len(series), replace=True)
        row_sets.append(np.concatenate([groups[s] for s in samp]))

    out = []
    base_full = _npl(frames[baseline])
    for v in variants:
        d = np.array([_npl(frames[v].iloc[r]) - _npl(frames[baseline].iloc[r])
                      for r in row_sets])
        lo, hi = np.percentile(d, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
        out.append(DeltaCI(
            baseline=baseline, variant=v, split=split or "all",
            delta=_npl(frames[v]) - base_full, lo=float(lo), hi=float(hi),
            n_series=len(series), n_rows=len(frames[baseline]), b=b,
            slice_id=slice_id))
    return out


def min_detectable_effect(cis: list[DeltaCI]) -> float:
    """CI 半宽的中位数——低于该量级的差异在本 split 上不可分辨。"""
    return float(np.median([(c.hi - c.lo) / 2 for c in cis]))


def report(cis: list[DeltaCI]) -> pd.DataFrame:
    df = pd.DataFrame([c.as_dict() for c in cis])
    df["verdict"] = np.where(df["significant"], "显著", "不显著(CI 跨 0)")
    return df[["split", "slice_id", "baseline", "variant", "delta", "lo", "hi",
               "significant", "verdict", "n_series", "n_rows", "b"]]


def paired_bootstrap_mean(
    df: pd.DataFrame,
    value_col: str,
    baseline: str,
    variants: list[str],
    variant_col: str = "variant",
    series_col: str = "series_id",
    b: int = BOOTSTRAP_B,
    ci: float = BOOTSTRAP_CI,
    seed: int = SEED_BASE,
) -> list[DeltaCI]:
    """按序列聚簇的配对 bootstrap，用于**逐行可分解的均值型指标**（成本、
    短缺量等）。

    与 paired_bootstrap 的分工：NPL 有比值形式的归一化分母，不可按行分解，
    必须在重抽样后整体重算；成本是逐行相加的，均值差即可直接重抽样。二者
    不能互相替代 —— 用本函数算 NPL 会得到错误的 CI。
    """
    frames = {v: g.sort_values([series_col, "month"]).reset_index(drop=True)
              for v, g in df.groupby(variant_col)}
    if baseline not in frames:
        raise ValueError(f"基准 {baseline!r} 不在 {sorted(frames)}")
    base = frames[baseline]
    series = base[series_col].to_numpy()
    for v in variants:
        if not np.array_equal(frames[v][series_col].to_numpy(), series):
            raise ValueError(f"{v} 与基准行序不一致，配对失效")

    uniq = np.asarray(sorted(set(series)))
    idx_of = {s: np.flatnonzero(series == s) for s in uniq}
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    draws = [np.concatenate([idx_of[s] for s in rng.choice(uniq, uniq.size, True)])
             for _ in range(b)]

    out = []
    for v in variants:
        d = frames[v][value_col].to_numpy(float) - base[value_col].to_numpy(float)
        boot = np.array([d[ix].mean() for ix in draws])
        out.append(DeltaCI(baseline=baseline, variant=v, split="validation",
                           delta=float(d.mean()),
                           lo=float(np.quantile(boot, lo_q)),
                           hi=float(np.quantile(boot, hi_q)),
                           n_series=uniq.size, n_rows=len(d), b=b))
    return out
