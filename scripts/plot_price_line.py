#!/usr/bin/env python3
"""将转折点 CSV 中的 price 画成折线图，直观查看价格走势。

横坐标使用「累计 dt_minutes」：
  - CSV 中 dt_minutes 是相邻转折点之间的间隔（分钟），首行 start 为空；
  - 画图时逐行累加，得到每个转折点的横坐标（首点为 0），
    使水平间距与原始数据的时间比例一致，而不是按行号均匀排布；
  - 末行 close 的 dt_minutes=0，因此与前一转折点共用同一横坐标（忠实于数据）。
纵坐标为原始 price 值，不做归一化或缩放。

用法：
  python plot_price_line.py [csv_path] [-o OUTPUT]

默认读取项目内 data/turning_points/DCE.v2701_1d.csv，
输出 PNG 保存在 CSV 同目录下（<csv 文件名>_price.png）。

仅依赖标准库 + matplotlib。需要无头运行时使用 Agg 后端，输出确定。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也可保存图片

import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "turning_points" / "DCE.v2701_1d.csv"


def load_points(csv_path: Path) -> list[tuple[float, float, str]]:
    """读取转折点 CSV，返回 [(累计dt_minutes, price, kind), ...]。

    - 首行（start）dt_minutes 为空，横坐标取 0；
    - 其后每行横坐标 = 上一横坐标 + 本行 dt_minutes；
    - 非首行若 dt_minutes 为空或无法解析，视为数据错误并报错退出。
    """
    points: list[tuple[float, float, str]] = []
    x = 0.0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            price_raw = (row.get("price") or "").strip()
            if not price_raw:
                raise ValueError(f"第 {i + 2} 行 price 为空，无法绘图")
            dt_raw = (row.get("dt_minutes") or "").strip()
            if i == 0:
                if dt_raw:
                    raise ValueError("首行（start）dt_minutes 应为空")
            else:
                if not dt_raw:
                    raise ValueError(f"第 {i + 2} 行 dt_minutes 为空，无法累加横坐标")
                x += float(dt_raw)
            points.append((x, float(price_raw), (row.get("kind") or "").strip()))
    return points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把转折点 CSV 的 price 画成折线图（x=累计 dt_minutes，y=price）"
    )
    parser.add_argument("csv", nargs="?", default=None, help="输入 CSV 路径（默认项目内 DCE.v2701_1d.csv）")
    parser.add_argument("-o", "--output", default=None, help="输出 PNG 路径（默认与 CSV 同目录）")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
    if not csv_path.is_file():
        parser.error(f"CSV 文件不存在: {csv_path}")

    points = load_points(csv_path)
    if not points:
        parser.error("CSV 中没有可绘制的数据行")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    output = Path(args.output) if args.output else csv_path.with_name(csv_path.stem + "_price.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, ys, color="#1f77b4", linewidth=1.2, marker="o", markersize=3)
    ax.set_xlabel("cumulative dt_minutes (min)")
    ax.set_ylabel("price")
    ax.set_title(f"price vs cumulative dt_minutes — {csv_path.name}")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="x", style="plain")  # 横坐标保持原始分钟数值，不用科学计数法
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

    print(f"points: {len(points)}")
    print(f"x range: {xs[0]:.0f} .. {xs[-1]:.0f} minutes")
    print(f"y range: {min(ys):.1f} .. {max(ys):.1f}")
    print(f"saved:   {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
