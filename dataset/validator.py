"""落盘前 OHLCV 合法性校验（收集全部违反项）。

校验规则（不自行增设业务规则）：

```text
timestamp 唯一
timestamp 升序
OHLC 数值合法（非空、有限数）
volume >= 0
high >= max(open, close)
low  <= min(open, close)
```

设计要点：**收集全部违反项**（非 fail-fast）；结构门禁（必需列 / 非空 /
tz-aware）作为前置；OHLC 关系只对四值齐全且有限的行判定，避免级联误报；
``volume = 0`` 合法（真实数据存在无成交分钟）。

移植自早期版本数据层的 validator 模块（去掉
``DataValidator`` 包装类，保留 ``Violation`` / ``ValidationReport`` /
``validate_ohlcv`` 三个公开对象）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset.errors import ValidationError
from dataset.ohlcv import OHLCV_COLUMNS

_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass(frozen=True)
class Violation:
    """一条违反项：机器可读代码 + 人读消息。"""

    code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """校验结果集合。"""

    violations: tuple[Violation, ...]

    @property
    def ok(self) -> bool:
        """是否全部通过。"""
        return not self.violations

    def raise_if_invalid(self) -> None:
        """存在违反项时抛出 :class:`ValidationError`（含全部明细）。"""
        if self.violations:
            details = "; ".join(f"[{v.code}] {v.message}" for v in self.violations)
            raise ValidationError(
                f"OHLCV 校验失败（{len(self.violations)} 项）: {details}"
            )


def validate_ohlcv(df: pd.DataFrame) -> ValidationReport:
    """校验标准 OHLCV DataFrame，收集全部违反项。"""
    violations: list[Violation] = []

    # ---- 结构门禁：字段完整性（列不齐时后续检查无意义，提前返回）----
    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        violations.append(Violation("MISSING_COLUMNS", f"缺少必需列: {missing}"))
        return ValidationReport(tuple(violations))
    if df.empty:
        return ValidationReport((Violation("EMPTY_DATA", "OHLCV 为空"),))

    # ---- timestamp：tz-aware / 非缺失 / 唯一 / 升序 ----
    ts = df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts) or ts.dt.tz is None:
        violations.append(
            Violation(
                "TIMESTAMP_DTYPE",
                f"timestamp 必须为 tz-aware datetime64，实际 dtype: {ts.dtype}",
            )
        )
    else:
        nat_count = int(ts.isna().sum())
        if nat_count:
            violations.append(
                Violation("TIMESTAMP_MISSING", f"timestamp 存在缺失（{nat_count} 行）")
            )
        duplicated = ts.duplicated()
        if bool(duplicated.any()):
            examples = ts[duplicated].head(3).astype(str).tolist()
            violations.append(
                Violation(
                    "TIMESTAMP_DUPLICATE",
                    f"timestamp 存在重复（{int(duplicated.sum())} 行），例: {examples}",
                )
            )
        if not ts.dropna().is_monotonic_increasing:
            violations.append(
                Violation("TIMESTAMP_NOT_ASCENDING", "timestamp 未按升序排列")
            )

    # ---- OHLC：非空、有限数 ----
    prices = {col: pd.to_numeric(df[col], errors="coerce") for col in _PRICE_COLUMNS}
    complete = pd.Series(True, index=df.index)  # 四值齐全且有限的行
    for col, values in prices.items():
        missing_mask = values.isna()
        if bool(missing_mask.any()):
            violations.append(
                Violation(
                    "OHLC_MISSING_VALUE",
                    f"{col} 存在缺失或非数值（{int(missing_mask.sum())} 行）",
                )
            )
        finite_mask = np.isfinite(values.to_numpy(dtype="float64"))  # = notna 且有限
        inf_mask = values.notna() & ~finite_mask  # inf/-inf（缺失已单独报错）
        if bool(inf_mask.any()):
            violations.append(
                Violation(
                    "OHLC_NOT_FINITE",
                    f"{col} 存在非有限数（{int(inf_mask.sum())} 行）",
                )
            )
        complete &= finite_mask

    # ---- volume >= 0 ----
    volume = pd.to_numeric(df["volume"], errors="coerce")
    if bool(volume.isna().any()):
        violations.append(
            Violation(
                "VOLUME_MISSING",
                f"volume 存在缺失或非数值（{int(volume.isna().sum())} 行）",
            )
        )
    else:
        negative = volume < 0
        if bool(negative.any()):
            violations.append(
                Violation("VOLUME_NEGATIVE", f"volume 存在负值（{int(negative.sum())} 行）")
            )

    # ---- OHLC 关系（仅对完整行判定，避免级联误报）----
    if bool(complete.any()):
        frame = pd.DataFrame({col: values[complete] for col, values in prices.items()})
        bad_high = frame["high"] < frame[["open", "close"]].max(axis=1)
        if bool(bad_high.any()):
            violations.append(
                Violation(
                    "HIGH_BELOW_MAX_OPEN_CLOSE",
                    f"high < max(open, close)（{int(bad_high.sum())} 行）",
                )
            )
        bad_low = frame["low"] > frame[["open", "close"]].min(axis=1)
        if bool(bad_low.any()):
            violations.append(
                Violation(
                    "LOW_ABOVE_MIN_OPEN_CLOSE",
                    f"low > min(open, close)（{int(bad_low.sum())} 行）",
                )
            )

    return ValidationReport(tuple(violations))
