"""配置加载与一致性校验。对应 configs/README.md 的四条加载断言。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

# 各数据集的实测末日（provenance），用于断言 2
DATA_MAX = {
    "fresh_lt": "2025-07-20",
    "fresh_50k": "2024-06-25",
    "osa": "2021-05-03",
    "lokad": "2022-12-31",
    "mendeley": "2026-04-24",
}


class Config(dict):
    """薄封装，保留 dict 语义并提供点号访问常用段。"""

    @property
    def name(self) -> str:
        return self["dataset"] if "dataset_version" not in self else f"{self['dataset']}"

    @property
    def splits(self) -> dict:
        return self["splits"]

    @property
    def metric(self) -> dict:
        return self["metric"]

    @property
    def caps(self) -> dict:
        return self["capabilities"]

    def origins(self, split: str) -> list[dt.date]:
        """枚举该 split 的 origin。月频返回每月首日。"""
        a, b = self.splits[split]["origins"]
        cal = self["calendar"]
        if cal["origin_frequency"] == "monthly":
            a = _as_month(a)
            b = _as_month(b)
            out, cur = [], a
            while cur <= b:
                out.append(cur)
                cur = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
            return out
        a, b = _as_date(a), _as_date(b)
        step = dt.timedelta(days=7)
        out, cur = [], a
        while cur <= b:
            out.append(cur)
            cur += step
        return out


def _as_date(v) -> dt.date:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    return dt.date.fromisoformat(str(v))


def _as_month(v) -> dt.date:
    s = str(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return _as_date(v).replace(day=1)
    y, m = s.split("-")[:2]
    return dt.date(int(y), int(m), 1)


def load(name: str) -> Config:
    """加载并校验一个数据集配置。任一断言失败即抛错，不静默放行。"""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"未找到配置 {path}；fresh 须指名 fresh_lt 或 fresh_50k")
    cfg = Config(yaml.safe_load(path.read_text(encoding="utf-8")))
    _validate(name, cfg)
    return cfg


def _validate(name: str, cfg: Config) -> None:
    cal = cfg["calendar"]
    monthly = cal["origin_frequency"] == "monthly"

    # 断言 1：三个 split 有序且不重叠
    bounds = []
    for s in ("train", "validation", "test"):
        a, b = cfg.splits[s]["origins"]
        a, b = (_as_month(a), _as_month(b)) if monthly else (_as_date(a), _as_date(b))
        if a > b:
            raise ValueError(f"{name}.{s}: origin 区间倒置 {a}..{b}")
        bounds.append((a, b))
    for (s1, (_, b1)), (s2, (a2, _)) in zip(
        zip(("train", "validation"), bounds[:2]),
        zip(("validation", "test"), bounds[1:]),
    ):
        if b1 >= a2:
            raise ValueError(f"{name}: {s1} 与 {s2} 的 origin 区间重叠或未严格递增")

    # 断言 2：最后一个 test 目标窗末日不越界
    if not monthly and name in DATA_MAX:
        last = bounds[2][1] + dt.timedelta(days=cal["horizon_days"] - 1)
        if last > _as_date(DATA_MAX[name]):
            raise ValueError(f"{name}: 末个 test 目标窗 {last} 超出数据末日 {DATA_MAX[name]}")

    # 断言 3：y_min 与 bind_rate 成对；10% 上限**只对连续量目标**生效（§4.3 路径 B）
    m = cfg.metric
    integer_units = bool(m.get("integer_units", True))
    if m.get("y_min") is not None and m.get("y_min_bind_rate") is not None:
        if not integer_units and m["y_min_bind_rate"] > 0.10:
            raise ValueError(
                f"{name}: 连续量目标 y_min_bind_rate={m['y_min_bind_rate']} > 0.10（§4.3 路径 B）")
        if integer_units and abs(float(m["y_min"]) - 1.0) > 1e-9:
            raise ValueError(
                f"{name}: 整数计数目标的 y_min 必须为 1.0（§4.3 路径 A），实为 {m['y_min']}")

    # 断言 4：无决策层时不得出现成本参数
    if not cfg.caps.get("decision_layer", False) and "cost" in cfg:
        raise ValueError(f"{name}: capabilities.decision_layer=false 却声明了 cost 段")

    # 回放窗为闭区间：end = start + days - 1
    rp = cfg.get("replay", {}).get("window")
    if rp:
        st, dd = _as_date(rp["start"]), int(rp["days"])
        if _as_date(rp["end"]) != st + dt.timedelta(days=dd - 1):
            raise ValueError(f"{name}: replay.window 违反闭区间约定 end=start+days-1")
