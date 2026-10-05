#!/usr/bin/env python3
"""把 run 中任一决策点的 v11 state 渲染成确定性单决策点快照 PNG。

图上可逐键核对模型所见状态与原始盘面：顶部回显区逐 token 原样渲染 state 六行文本，
主图为片段 1m 手绘蜡烛图（红涨绿跌）+ 日线/日内水平参考线（昨日高/低/收、
涨跌势极值 ×4、今高/今低，绝对价 = state 比值 × 片段首根开盘价）+ 决策 K 线高亮，
左下联动子图画联动品种 close（x 对齐主图 bar 序号的交集时间戳），
右下文字块列账户/现价补充/盘口/趋势键值。

仅解析 `marketsense.episode_state.v11`，v12+ 格式需另起解析器（fail loud，不静默画错）。

用法（仓库根目录，解释器用 base 环境 /opt/anaconda3/bin/python，
marketsense 环境无 matplotlib）：

  /opt/anaconda3/bin/python scripts/plot_state.py \
      [--run DIR] [--split SPLIT] [--segment ID] [--bar N] \
      [--segments FILE] [--ohlcv-dir DIR] [--linkage SYMBOL] [-o OUTPUT]

默认示例：
  --run data/nanojev_dataset/run-8886261c37d7 --split dev
  --segment DCE.v2701-td20260921-dev --bar 19
  --segments data/segments/sep2026.jsonl --ohlcv-dir data/ohlcv
  --linkage INE.sc2611
  -o <run>/plots/state-<segment>-bar<bar>.png

确定性：Agg 后端 + 固定 figsize/dpi/subplots_adjust + 固定字体候选表，
图内无墙钟时间；同机同命令双跑 PNG 字节一致。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也可保存图片，且输出确定

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCHEMA_V11 = "marketsense.episode_state.v11"
REQUIRED_HEADERS = ("账户", "日线", "日内", "联动", "现价", "盘口")

GRAY = "#888888"  # na 统一灰
UP_COLOR = "#D62728"  # 红涨
DOWN_COLOR = "#2CA02C"  # 绿跌
HIGHLIGHT = "#1F77B4"  # 决策 K 线高亮

# CJK 字体固定候选表（同机按顺序取首个可用 → 双跑字节一致）
FONT_CANDIDATES = (
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Heiti SC",
)

TREND_RE = re.compile(r"(涨势|跌势)\((-?\d+), (最高|最低)=([^,]+), 时长=(\d+)根\)")
LINK_RE = re.compile(r"^(?P<name>[^（]+)（突破=(?P<val>[^）]*)）$")


def die(msg: str) -> None:
    """致命错误：stderr 中文可读信息 + exit 2。"""
    print(f"[错误] {msg}", file=sys.stderr)
    sys.exit(2)


def warn(msg: str) -> None:
    """非致命告警：stderr 提示，继续执行（exit 0）。"""
    print(f"[警告] {msg}", file=sys.stderr)


def setup_fonts() -> None:
    """按固定候选表取系统首个可用 CJK 字体；全部缺失时 stderr 警告（非阻断）。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    found = next((c for c in FONT_CANDIDATES if c in installed), None)
    if found is None:
        warn("未找到候选 CJK 字体（候选表：{}），中文可能显示为方块".format("、".join(FONT_CANDIDATES)))
        return
    matplotlib.rcParams["font.sans-serif"] = [found]
    matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# v11 state 解析器（fail loud）
# ---------------------------------------------------------------------------

def parse_state(state_text: str) -> dict:
    """解析 v11 state 文本。

    返回 {行头: {"tokens": [原始 token], "kv": {键: 原始值},
                 "trends": [...], "linkages": [...]}}，
    kv/trend/linkage 中的值均保留原始字符串（含 na）。
    未知 token / 缺必需行 → 报错退出。
    """
    lines = state_text.split(" \n ")
    schema = lines[0].strip()
    if schema != SCHEMA_V11:
        die(f"state 非 {SCHEMA_V11}（实际：{schema!r}），本脚本仅支持 v11")

    result: dict[str, dict] = {}
    for line in lines[1:]:
        m = re.match(r"^(\S+): (.*)$", line)
        if not m:
            die(f"state 行无法解析（应为「行头: token ...」）：{line!r}")
        header, rest = m.group(1), m.group(2)
        row: dict = {"tokens": rest.split(" "), "kv": {}, "trends": [], "linkages": []}
        # 复合 token（涨跌势）内含空格，先整体提取再按空格切简单 token
        for tm in TREND_RE.finditer(rest):
            row["trends"].append(
                {"方向": tm.group(1), "序号": tm.group(2),
                 "极值键": tm.group(3), "值": tm.group(4), "时长": tm.group(5)}
            )
        work_tokens = [t for t in TREND_RE.sub(" ", rest).split(" ") if t]
        for tok in work_tokens:
            # 联动品种 token（可含多个品种，以全角逗号相连）：逐段匹配
            pieces = [LINK_RE.match(p) for p in tok.split("，")]
            if all(pieces):
                for lm in pieces:
                    row["linkages"].append({"name": lm.group("name"), "突破": lm.group("val")})
                continue
            if "=" in tok:
                key, _, val = tok.partition("=")
                row["kv"][key] = val
                continue
            # 无 "=" 的裸 token：仅允许整行单 token（如「盘口: na」）
            if len(work_tokens) == 1:
                row["kv"]["值"] = tok
            else:
                die(f"未知 token（{header} 行）：{tok!r}")
        result[header] = row

    missing = [h for h in REQUIRED_HEADERS if h not in result]
    if missing:
        die(f"state 缺少必需行：{'、'.join(missing)}")
    unknown = [h for h in result if h not in REQUIRED_HEADERS]
    if unknown:
        die(f"state 含未知行头（非 v11 六部分）：{'、'.join(unknown)}")
    return result


def is_na(val: str | None) -> bool:
    return val is None or val.strip() == "na"


# ---------------------------------------------------------------------------
# 数据装配（只读）
# ---------------------------------------------------------------------------

def load_record(run_dir: Path, split: str, segment: str, bar: int) -> dict:
    """从 <run>/<split>.jsonl 定位唯一记录（segment_id + bar_index）。"""
    path = run_dir / f"{split}.jsonl"
    if not path.is_file():
        die(f"run JSONL 不存在：{path}")
    hits = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            if meta.get("segment_id") == segment and meta.get("bar_index") == bar:
                hits.append(rec)
    if not hits:
        die(f"未找到记录：segment_id={segment} bar_index={bar}（文件：{path}）")
    if len(hits) > 1:
        die(f"记录不唯一（{len(hits)} 条）：segment_id={segment} bar_index={bar}（文件：{path}）")
    return hits[0]


def load_segment(segments_file: Path, segment: str) -> dict:
    """从片段清单按 segment_id 取片段行。"""
    if not segments_file.is_file():
        die(f"segments 文件不存在：{segments_file}")
    with open(segments_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("segment_id") == segment:
                return row
    die(f"segment 不在片段清单中：{segment}（文件：{segments_file}）")


def load_1m(csv_path: Path, start_ts: str, end_ts: str) -> pd.DataFrame:
    """读 1m CSV，时间戳转 Asia/Shanghai 后取 [start_ts, end_ts] 闭区间，升序。"""
    if not csv_path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True).dt.tz_convert("Asia/Shanghai")
    start = pd.Timestamp(start_ts).tz_localize("Asia/Shanghai")
    end = pd.Timestamp(end_ts).tz_localize("Asia/Shanghai")
    df = df[(df["ts"] >= start) & (df["ts"] <= end)].sort_values("ts").reset_index(drop=True)
    return df


def find_linkage_csv(ohlcv_dir: Path, display_name: str) -> Path | None:
    """按联动显示名查 CSV：先精确 `<name>_1m.csv`，再 glob `*.<name>_1m.csv` 取 sorted 首个。"""
    exact = ohlcv_dir / f"{display_name}_1m.csv"
    if exact.is_file():
        return exact
    hits = sorted(ohlcv_dir.glob(f"*.{display_name}_1m.csv"))
    if len(hits) > 1:
        warn(f"联动 CSV glob 多命中（{len(hits)} 个），取首个：{hits[0].name}")
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def place_labels(labels: list[tuple[float, str]], y_min: float, y_max: float) -> list[tuple[float, str]]:
    """右侧参考线标注防重叠：按 y 降序，相邻标签保持固定间距（确定性）。"""
    gap = (y_max - y_min) * 0.028
    out = []
    prev = None
    for y, text in sorted(labels, key=lambda t: -t[0]):
        if prev is not None and y > prev - gap:
            y = prev - gap
        prev = y
        out.append((y, text))
    return out


def plot(run_dir: Path, rec: dict, parsed: dict, seg: dict, main_df: pd.DataFrame,
         link_df: pd.DataFrame | None, link_name: str | None, out_path: Path) -> None:
    bar = int(rec["metadata"]["bar_index"])
    segment = rec["metadata"]["segment_id"]
    first_open = float(main_df.iloc[0]["open"])
    n = len(main_df)

    fig = plt.figure(figsize=(14, 9), dpi=150)
    grid = fig.add_gridspec(3, 2, height_ratios=[18, 52, 30], width_ratios=[60, 40])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.05, hspace=0.34, wspace=0.16)
    echo_ax = fig.add_subplot(grid[0, :])
    main_ax = fig.add_subplot(grid[1, :])
    link_ax = fig.add_subplot(grid[2, 0])
    text_ax = fig.add_subplot(grid[2, 1])

    # ---- 1. 回显区：state 六行逐 token 原样渲染 ----
    echo_ax.axis("off")
    raw_lines = [h + ": " + " ".join(parsed[h]["tokens"]) for h in REQUIRED_HEADERS]
    echo_ax.text(
        0.0, 1.0, "\n".join(raw_lines), transform=echo_ax.transAxes,
        va="top", ha="left", fontsize=8, linespacing=1.7,
        bbox=dict(facecolor="#F2F2F2", edgecolor="#CCCCCC", pad=6),
    )

    # ---- 2. 主图：蜡烛 + 水平参考线 + 决策 K 线高亮 ----
    xs = range(n)
    for i, row in main_df.iterrows():
        o, h, low, c = (float(row[k]) for k in ("open", "high", "low", "close"))
        color = UP_COLOR if c >= o else DOWN_COLOR
        main_ax.vlines(i, low, h, color=color, linewidth=0.9, zorder=2)
        body_low, body_h = min(o, c), max(abs(c - o), (h - low) * 1e-3)
        main_ax.add_patch(Rectangle((i - 0.35, body_low), 0.7, body_h,
                                    facecolor=color, edgecolor=color, linewidth=0.5, zorder=3))

    dai = parsed["日线"]["kv"]
    refs = []  # (y_price, 标注, 线型, 颜色, 线宽)
    for key, ls, color, lw in (
        ("昨日高", "--", "#B8860B", 1.0),
        ("昨日低", "--", "#4C72B0", 1.0),
        ("昨日收", "--", "#555555", 1.0),
    ):
        if not is_na(dai.get(key)):
            refs.append((float(dai[key]) * first_open, key, ls, color, lw))
    for t in parsed["日线"]["trends"]:
        if is_na(t["值"]):
            continue
        ls = ":" if t["序号"] == "-1" else "-."
        color = "#E24A33" if t["方向"] == "涨势" else "#1E7B34"
        label = f"{t['方向']}{t['序号']}·{t['极值键']}"
        refs.append((float(t["值"]) * first_open, label, ls, color, 1.0))
    intra = parsed["日内"]["kv"]
    for key in ("今高", "今低"):
        if not is_na(intra.get(key)):
            refs.append((float(intra[key]) * first_open, key, "-", HIGHLIGHT, 0.8))

    line_vals = [r[0] for r in refs]
    y_min = min(float(main_df["low"].min()), *line_vals) if line_vals else float(main_df["low"].min())
    y_max = max(float(main_df["high"].max()), *line_vals) if line_vals else float(main_df["high"].max())
    pad = (y_max - y_min) * 0.05 if y_max > y_min else max(abs(y_max) * 0.05, 1e-9)
    y_min, y_max = y_min - pad, y_max + pad

    for y_price, label, ls, color, lw in refs:
        main_ax.axhline(y_price, linestyle=ls, color=color, linewidth=lw, zorder=1)
    for y_price, label in place_labels([(r[0], f"{r[1]} {r[0]:.1f}") for r in refs], y_min, y_max):
        main_ax.text(n - 0.6, y_price, label, ha="right", va="center",
                     fontsize=6.5, color="#444444", zorder=5)

    # 决策 K 线描边高亮 + 箭头标注
    drow = main_df.iloc[bar]
    main_ax.add_patch(Rectangle((bar - 0.42, float(drow["low"])), 0.84,
                                max(float(drow["high"]) - float(drow["low"]), 1e-9),
                                facecolor="none", edgecolor=HIGHLIGHT, linewidth=2.2, zorder=4))
    main_ax.annotate(f"bar={bar}", xy=(bar, float(drow["high"])),
                     xytext=(bar, y_max - (y_max - y_min) * 0.03),
                     ha="center", fontsize=8, color=HIGHLIGHT,
                     arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, linewidth=1.0), zorder=6)

    main_ax.set_xlim(-1, n)
    main_ax.set_ylim(y_min, y_max)
    main_ax.set_ylabel("价格（绝对价 = state 比值 × 片段首根开盘）")
    main_ax.set_xlabel("bar 序号（片段窗口内，0 基）")
    main_ax.set_title(f"{SCHEMA_V11} | {segment} | bar={bar} | {rec.get('id', '')}", fontsize=10)
    main_ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    # ---- 3. 联动子图 ----
    link_row = next((l for l in parsed["联动"]["linkages"] if l["name"] == link_name), None)
    corr = parsed["联动"]["kv"].get("相关度", "na")
    title_val = link_row["突破"] if link_row else "na"
    link_ax.set_title(f"联动 {link_name or '?'} 突破={title_val} 相关度={corr}", fontsize=9)
    if link_df is None or link_df.empty:
        link_ax.text(0.5, 0.5, "无交集数据（na）", transform=link_ax.transAxes,
                     ha="center", va="center", fontsize=10, color=GRAY)
        link_ax.set_xticks([])
        link_ax.set_yticks([])
    else:
        ts_index = {ts: i for i, ts in enumerate(main_df["ts"])}
        pts = [(ts_index[ts], float(c)) for ts, c in zip(link_df["ts"], link_df["close"]) if ts in ts_index]
        if pts:
            lx, ly = zip(*pts)
            link_ax.plot(lx, ly, color="#9467BD", linewidth=1.2)
            link_ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
        else:
            link_ax.text(0.5, 0.5, "无交集数据（na）", transform=link_ax.transAxes,
                         ha="center", va="center", fontsize=10, color=GRAY)
            link_ax.set_xticks([])
            link_ax.set_yticks([])
    link_ax.set_xlabel("主图 bar 序号（交集时间戳对齐）", fontsize=8)

    # ---- 4. 文字块：账户/现价补充/盘口/趋势 ----
    text_ax.axis("off")
    lines: list[tuple[str, bool]] = []  # (文本, 是否灰色)
    for key in ("持仓", "开仓价", "止损价", "净值", "今日", "回撤"):
        val = parsed["账户"]["kv"].get(key)
        if val is not None:
            lines.append((f"账户 {key}={val}", is_na(val)))
    cur = parsed["现价"]["kv"]
    for key in ("成交量比", "持仓量比"):
        val = cur.get(key)
        if val is not None:
            lines.append((f"现价补充 {key}={val}", is_na(val)))
    lines.append(("盘口: " + parsed["盘口"]["kv"].get("值", "na"), True))
    trend_txt = " ".join(f"{k}={v}" for k, v in parsed["日线"]["kv"].items() if k in ("趋势", "时长"))
    if trend_txt:
        lines.append((f"日线 {trend_txt}", False))
    y = 0.98
    for text, gray in lines:
        text_ax.text(0.0, y, text, transform=text_ax.transAxes, va="top", ha="left",
                     fontsize=9.5, color=GRAY if gray else "#222222")
        y -= 0.115

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 run 中任一决策点的 v11 state 渲染成确定性单决策点快照 PNG（详见脚本 docstring）",
    )
    parser.add_argument("--run", default="data/nanojev_dataset/run-8886261c37d7", help="run 产物目录")
    parser.add_argument("--split", default="dev", help="split 名（train/dev/test）")
    parser.add_argument("--segment", default="DCE.v2701-td20260921-dev", help="segment_id")
    parser.add_argument("--bar", type=int, default=19, help="决策 K 线序号（片段窗口内，0 基）")
    parser.add_argument("--segments", default="data/segments/sep2026.jsonl", help="片段清单 JSONL")
    parser.add_argument("--ohlcv-dir", default="data/ohlcv", help="OHLCV CSV 目录")
    parser.add_argument("--linkage", default="INE.sc2611", help="联动品种（用于定位 CSV；与联动行显示名去前缀匹配）")
    parser.add_argument("-o", "--output", default=None, help="输出 PNG（默认 <run>/plots/state-<segment>-bar<bar>.png）")
    args = parser.parse_args(argv)

    if args.bar < 0:
        die(f"--bar 必须 ≥ 0（实际：{args.bar}）")

    setup_fonts()

    run_dir = Path(args.run)
    segments_file = Path(args.segments)
    ohlcv_dir = Path(args.ohlcv_dir)

    rec = load_record(run_dir, args.split, args.segment, args.bar)
    parsed = parse_state(rec["state"])
    seg = load_segment(segments_file, args.segment)

    symbol = seg["symbol"]
    main_df = load_1m(ohlcv_dir / f"{symbol}_1m.csv", seg["start_ts"], seg["end_ts"])
    if main_df.empty:
        die(f"主品种 1m 窗口为空：{symbol} [{seg['start_ts']}, {seg['end_ts']}]（CSV：{ohlcv_dir / (symbol + '_1m.csv')}）")
    if args.bar >= len(main_df):
        die(f"--bar 越界：bar={args.bar} ≥ 窗口内 K 线数 {len(main_df)}（segment={args.segment}）")

    # 联动品种：显示名 = --linkage 去交易所前缀；缺数据非致命（na 呈现，exit 0）
    link_name = args.linkage.split(".", 1)[-1] if args.linkage else None
    link_df = pd.DataFrame()
    if link_name:
        link_csv = find_linkage_csv(ohlcv_dir, link_name)
        if link_csv is None:
            warn(f"联动品种 CSV 缺失：{link_name}（{ohlcv_dir}），联动子图以 na 呈现")
        else:
            link_df = load_1m(link_csv, seg["start_ts"], seg["end_ts"])
            if link_df.empty:
                warn(f"联动品种与主品种窗口无交集数据：{link_name}，联动子图以 na 呈现")

    out_path = Path(args.output) if args.output else run_dir / "plots" / f"state-{args.segment}-bar{args.bar}.png"
    plot(run_dir, rec, parsed, seg, main_df, link_df, link_name, out_path)
    print(f"已写出：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
