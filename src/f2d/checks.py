"""验收契约。对应 docs/00_global_conventions.md §9.3。

退出码 0=PASS / 1=FAIL / 2=BLOCKED。BLOCKED 表示已声明的数据门未通过，
下游跳过而非整链失败；它与 FAIL 是不同状态，不得混用。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .conventions import EXIT_BLOCKED, EXIT_FAIL, EXIT_PASS, FLOAT_DECIMALS


@dataclass
class CheckResult:
    step_id: str
    dataset: str
    seed: int
    assertions: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)
    n_rows: int = 0
    n_excluded: int = 0
    n_fallback: int = 0
    blocked_gates: list = field(default_factory=list)

    def assert_true(self, name: str, ok: bool, detail: str = "") -> bool:
        self.assertions.append({"name": name, "passed": bool(ok), "detail": detail})
        return bool(ok)

    def block(self, gate_id: str, reason: str) -> None:
        self.blocked_gates.append({"gate": gate_id, "reason": reason})

    def note(self, key: str, value) -> None:
        self.notes[key] = value

    @property
    def status(self) -> str:
        if any(not a["passed"] for a in self.assertions):
            return "FAIL"
        return "BLOCKED" if self.blocked_gates else "PASS"

    @property
    def exit_code(self) -> int:
        return {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL, "BLOCKED": EXIT_BLOCKED}[self.status]

    def finish(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "step_id": self.step_id, "dataset": self.dataset, "status": self.status,
            "assertions": self.assertions, "n_rows": self.n_rows,
            "n_excluded": self.n_excluded, "n_fallback": self.n_fallback,
            "blocked_gates": self.blocked_gates, "seed": self.seed, "notes": self.notes,
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        for a in self.assertions:
            if not a["passed"]:
                print(f"  [FAIL] {a['name']}  {a['detail']}")
        return p


def frame_hash(df: pd.DataFrame, key_cols: list[str], float_cols: list[str]) -> str:
    """§9.2 复现哈希：按主键排序、固定列序、浮点格式化到 6 位后取 sha256。"""
    d = df.sort_values(key_cols).reset_index(drop=True)
    cols = key_cols + [c for c in float_cols if c in d.columns]
    d = d[cols].copy()
    for c in float_cols:
        if c in d.columns:
            d[c] = d[c].map(lambda v: f"%.{FLOAT_DECIMALS}f" % float(v))
    return hashlib.sha256(d.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
