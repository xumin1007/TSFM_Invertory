"""类别编码。对应 docs/00_global_conventions.md §7.1-§7.2。

核心：映射由本项目做，不依赖任何库的 unknown 分支。
先 to_vocab() 归一化，再交给 pandas / sklearn / chronos 编码，
三者产出逐元素相同的整数码。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .conventions import MISSING, RESERVED_SLOTS, UNK


def build_vocab(train_values: pd.Series) -> list[str]:
    """仅由训练窗构建的冻结词表。前两位为保留槽，顺序即编码。

    索引 0 = __UNK__（有记录但训练期未见），1 = __MISSING__（无记录）。
    二者必须分开：前者是冷启动信号，后者是数据质量信号（§7.1）。
    """
    real = sorted({str(v) for v in train_values.dropna().unique()})
    for slot in RESERVED_SLOTS:
        if slot in real:
            raise ValueError(f"训练数据中出现保留槽字面量 {slot!r}，需换用其他哨兵")
    return list(RESERVED_SLOTS) + real


def to_vocab(s: pd.Series, vocab: list[str]) -> np.ndarray:
    """把任意取值归一化到冻结词表。未见 -> __UNK__，缺失 -> __MISSING__。"""
    real = set(vocab[len(RESERVED_SLOTS):])
    v = s.astype(object)
    out = np.where(v.isna(), MISSING, np.where(v.astype(str).isin(real), v.astype(str), UNK))
    return out.astype(object)


def to_categorical(s: pd.Series, vocab: list[str]) -> pd.Categorical:
    """LightGBM / Chronos-2 路径：categories 显式设为完整冻结词表。

    必须显式传 categories —— 若让 pandas 从数据推断，未见值会在
    Chronos-2 侧变成 NaN sentinel，与 GBDT 侧行为分叉（§7.2）。
    """
    return pd.Categorical(to_vocab(s, vocab), categories=vocab)


def to_codes(s: pd.Series, vocab: list[str]) -> np.ndarray:
    """整数码。三个模型族共用同一映射。"""
    return to_categorical(s, vocab).codes.astype(np.int32)


class VocabStore:
    """全部类别列的冻结词表，连同构建截止时点。写 artifacts/<ds>/vocab.json。"""

    def __init__(self, vocabs: dict[str, list[str]], frozen_on: str):
        self.vocabs = vocabs
        self.frozen_on = frozen_on

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str], frozen_on: str) -> "VocabStore":
        return cls({c: build_vocab(df[c]) for c in columns}, frozen_on)

    def transform(self, df: pd.DataFrame, as_codes: bool = False) -> pd.DataFrame:
        out = df.copy()
        for c, vocab in self.vocabs.items():
            if c in out.columns:
                out[c] = to_codes(out[c], vocab) if as_codes else to_categorical(out[c], vocab)
        return out

    def unk_rate(self, df: pd.DataFrame) -> dict[str, float]:
        """各列落入 __UNK__ 的比例 —— 冷启动强度的直接度量。"""
        return {
            c: float(np.mean(to_vocab(df[c], v) == UNK))
            for c, v in self.vocabs.items()
            if c in df.columns
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"frozen_on": self.frozen_on, "vocabs": self.vocabs},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "VocabStore":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocabs"], d["frozen_on"])
