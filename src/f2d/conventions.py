"""全局常量。对应 docs/00_global_conventions.md §1。

这些值不得被任何调用方覆盖；逐数据集的取值放在 configs/*.yaml。
"""

from __future__ import annotations

TAU_MAIN: float = 0.50
TAU_UPPER: float = 0.85  # gamma
EPS_DENOM: float = 1e-12
CALIB_N_MIN: int = 30
QUANTILE_METHOD: str = "linear"  # numpy.quantile method (type-7)
BOOTSTRAP_B: int = 1000
BOOTSTRAP_CI: float = 0.95
SEED_BASE: int = 42
FLOAT_TOL: float = 1e-6
FLOAT_DECIMALS: int = 6  # 复现哈希的格式化位数

# §5.2 校准曲线网格
CAL_GRID = tuple(round(0.05 * k, 2) for k in range(1, 20))  # 0.05 .. 0.95

# §7.1 词表保留槽（顺序即编码，不得调换）
UNK = "__UNK__"
MISSING = "__MISSING__"
RESERVED_SLOTS = (UNK, MISSING)

# §3.1 业务权重档位
W_VALUES = {"low": 1.0, "mid": 1.5, "high": 2.5}
W_FLAT = 1.0

# §4.3 y_min 阶梯（取满足 bind_rate <= Y_MIN_BIND_CAP 的最大值）
Y_MIN_LADDER = (1.0, 0.5, 0.1, 0.05, 0.01)
Y_MIN_BIND_CAP = 0.10

# 验收退出码（§9.3）
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2
