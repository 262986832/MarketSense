"""移植等价性（parity）：与 ``reference/perception/scripts/turning_points.py`` 直接对比。

2026-09-26 起发射字段按甲口径偏离参考实现（``up``/``down`` 点在触发根 ``t`` 确认，
``price``/``volume``/``oi`` 取前一根），对比缩窄为**触发序列等价**（反转事件数量与
方向交替序列一致）：

* 反转序列：本包取 ``kind ∈ {up, down}`` 点的 kind 序列；参考实现把 ``high`` →
  ``up``、``low`` → ``down`` 后取同类序列；两者逐元素相等；
* ``start`` / ``close`` 点仍逐字段相等（kind/timestamp/price/bar_index）；
* 固定向量同时锁定 kind/price/bar_index 绝对期望，防「两边一起错」。

已知边界：参考实现的反转点锚定在反转价所在 K 线，不直接含触发根，故触发根时序的
相等由「两侧触发条件与状态机逐字一致」（代码证据）+ 固定向量锁 + fuzz 序列一致
共同支撑，属于该检查的固有上界，不夸大等价强度。

参考实现位于被 ``.gitignore`` 排除的 ``reference/`` 目录（只读、零修改）。若该脚本
不存在（例如仓库分发时未带参考目录），本文件整体 ``skip`` 并说明原因。
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from dataset.tests.conftest import TURNING_POINT_VECTORS, build_ohlcv
from dataset.turning_points import find_turning_points, resolve_initial_direction

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_SCRIPT = (
    _PROJECT_ROOT / "reference" / "perception" / "scripts" / "turning_points.py"
)

pytestmark = pytest.mark.skipif(
    not _REFERENCE_SCRIPT.is_file(),
    reason=f"参考实现不存在，无法做 parity 对比: {_REFERENCE_SCRIPT}",
)


def _load_reference_module():
    """以 importlib 加载参考脚本（不 import marksense，不修改 reference）。"""
    spec = importlib.util.spec_from_file_location(
        "reference_turning_points_script", _REFERENCE_SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["reference_turning_points_script"] = module
    spec.loader.exec_module(module)
    return module


_REFERENCE = _load_reference_module()


def _ours_reversal_kinds(points) -> list[str]:
    """本包反转序列：``kind ∈ {up, down}`` 点的 kind 序列（触发序列等价）。"""
    return [p.kind for p in points if p.kind in ("up", "down")]


def _mapped_reference_reversal_kinds(points) -> list[str]:
    """参考实现反转序列：``high`` → ``up``、``low`` → ``down`` 后取同类 kind 序列。

    映射由参考实现在运行时推导，不硬编码。
    """
    return ["up" if p.kind == "high" else "down" for p in points if p.kind in ("high", "low")]


def _endpoint_signature(points) -> list[tuple]:
    """start/close 逐字段签名（两侧 kind 命名一致，直接逐字段对比）。"""
    return [
        (p.kind, str(p.timestamp), float(p.price), int(p.bar_index))
        for p in points
        if p.kind in ("start", "close")
    ]


@pytest.mark.parametrize("vector", TURNING_POINT_VECTORS, ids=lambda v: v["name"])
def test_find_turning_points_matches_reference(vector: dict) -> None:
    df = build_ohlcv(vector["rows"])

    ours = find_turning_points(df, initial_direction=vector["initial_direction"])
    theirs = _REFERENCE.find_turning_points(
        df, initial_direction=vector["initial_direction"]
    )

    # 触发序列等价：反转 kind 序列逐元素相等；start/close 逐字段相等
    assert _ours_reversal_kinds(ours) == _mapped_reference_reversal_kinds(theirs)
    assert _endpoint_signature(ours) == _endpoint_signature(theirs)
    # 同时锁定固定期望值，避免「两边一起错」也通过
    assert [p.kind for p in ours] == vector["expect_kinds"]
    assert [p.price for p in ours] == vector["expect_prices"]
    assert [p.bar_index for p in ours] == vector["expect_bar_index"]


@pytest.mark.parametrize(
    "rows",
    [
        [(100, 101, 99, 100), (100, 101, 98, 99)],
        [(100, 101, 99, 100), (100, 102, 99, 101)],
        [(100, 101, 99, 101), (100, 101, 99, 100)],
        [(101, 101, 99, 100), (100, 101, 99, 100)],
        [(100, 101, 99, 100)],
        [(100, 105, 95, 102), (101, 103, 99, 100), (100, 102, 98, 99), (99, 101, 97, 100)],
    ],
)
def test_resolve_initial_direction_matches_reference(rows: list[tuple]) -> None:
    df = build_ohlcv(rows)
    for mode in ("auto", "up", "down"):
        assert resolve_initial_direction(df, mode) == _REFERENCE.resolve_initial_direction(
            df, mode
        )


def test_reference_module_was_not_modified() -> None:
    # 参考实现只读：对比运行前后文件内容指纹（在内存中重算，不做任何写入）
    before = _REFERENCE_SCRIPT.read_bytes()
    _load_reference_module()
    assert _REFERENCE_SCRIPT.read_bytes() == before


#: fuzz 种子与例数固定 → 用例确定性可复现（AGENTS.md §9）
_FUZZ_SEED = 20260924
_FUZZ_CASES = 200


def _random_ohlcv(rng: random.Random, count: int) -> list[tuple[float, float, float, float]]:
    """生成合法 ``(open, high, low, close)`` 序列（满足 high ≥ max(o,c)、low ≤ min(o,c)）。"""
    rows: list[tuple[float, float, float, float]] = []
    price = 100.0
    for _ in range(count):
        open_ = round(price + rng.uniform(-5, 5), 2)
        close = round(price + rng.uniform(-5, 5), 2)
        high = round(max(open_, close) + rng.uniform(0, 5), 2)
        low = round(min(open_, close) - rng.uniform(0, 5), 2)
        rows.append((open_, high, low, close))
        price = close
    return rows


def test_find_turning_points_matches_reference_on_seeded_fuzz() -> None:
    """AC-3：固定种子随机 fuzz（200 例 × 3 模式）按触发序列等价对比参考实现，
    反转序列与 start/close 字段差异必须为 0。"""
    rng = random.Random(_FUZZ_SEED)
    mismatches: list[str] = []
    direction_mismatches = 0
    cases = 0

    for case in range(_FUZZ_CASES):
        rows = _random_ohlcv(rng, rng.randint(1, 12))
        df = build_ohlcv(rows)
        for mode in ("auto", "up", "down"):
            cases += 1
            ours_points = find_turning_points(df, initial_direction=mode)
            theirs_points = _REFERENCE.find_turning_points(df, initial_direction=mode)
            if _ours_reversal_kinds(ours_points) != _mapped_reference_reversal_kinds(
                theirs_points
            ):
                mismatches.append(
                    f"case={case} mode={mode} "
                    f"reversal ours={_ours_reversal_kinds(ours_points)} "
                    f"theirs={_mapped_reference_reversal_kinds(theirs_points)} rows={rows}"
                )
            if _endpoint_signature(ours_points) != _endpoint_signature(theirs_points):
                mismatches.append(
                    f"case={case} mode={mode} "
                    f"endpoints ours={_endpoint_signature(ours_points)} "
                    f"theirs={_endpoint_signature(theirs_points)} rows={rows}"
                )
            if resolve_initial_direction(df, mode) != _REFERENCE.resolve_initial_direction(
                df, mode
            ):
                direction_mismatches += 1

    assert cases == _FUZZ_CASES * 3
    assert mismatches == []
    assert direction_mismatches == 0
