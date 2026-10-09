"""dataset 命令行入口（``python -m dataset``）。

六个子命令：

```text
fetch          在线取数 → 校验 → 落盘 K 线 CSV + 来源指纹 sidecar
turning-points 离线读取已落盘 K 线 → 转折点 CSV + 窗口 sidecar（不联网）
prepare        fetch 后接转折点提取
episode-generate  片段清单 + 已落盘 1 分钟 K 线 → 按 split 的 NanoJev JSONL + 审计
board-state    离线读取已落盘 1m/1d K 线 → 盘面状态 CSV + sidecar（不联网）
state-now      在线/离线构造「此刻」state/question/candidates/metadata 快照（stdout，不落盘）
```

退出码：``0`` 成功；``1`` 运行期失败（配置/凭证/取数/校验/读取）；``2`` 用法错误
（参数组合非法）。错误信息一律写 stderr，且**不含**凭证值。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import yaml

from dataset.board_state import (
    BOARD_STATE_SUBDIR,
    build_board_states,
    save_board_states,
)
from dataset.config import DEFAULT_CONFIG_PATH, DatasetConfig, load_dataset_config
from dataset.errors import ConfigError, DataLoadError, DatasetError, UnknownPeriodError
from dataset.live_state import DENOMINATOR_NOTE, LiveStateSnapshot, build_live_state
from dataset.market_episode import (
    AUDIT_FILENAME,
    OUTCOMES_FILENAME,
    QUESTION_ID,
    SPLIT_ROLES,
    EpisodeParams,
    generate_dataset,
    load_episode_config,
    load_segments,
    load_symbols_config,
)
from dataset.market_episode.segments import load_episode_params
from dataset.ohlcv import TIMEZONE, parse_date_bound
from dataset.periods import resolve_duration_seconds, supported_periods_text
from dataset.provider import (
    SERIAL_MAX_DATA_LENGTH,
    TianQinProvider,
    validate_data_length,
)
from dataset.storage import load_ohlcv, ohlcv_filename, save_ohlcv
from dataset.watch_state import (
    EXIT_INTERRUPTED,
    market_lines,
    run_watch,
    timestamp_header,
)
from dataset.turning_points import (
    INITIAL_DIRECTION_MODES,
    build_window_meta,
    find_turning_points,
    load_turning_points,
    resolve_initial_direction,
    save_turning_points,
)
from dataset.validator import validate_ohlcv

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

#: 产物子目录（相对 output_dir）
OHLCV_SUBDIR = "ohlcv"
TURNING_POINTS_SUBDIR = "turning_points"

#: 盘面状态子命令唯一支持的周期（窗口语义按 1m 交易日口径定义）
BOARD_STATE_PERIOD = "1m"

#: 子命令名（argparse ``dest=command`` 取值）
COMMAND_FETCH = "fetch"
COMMAND_TURNING_POINTS = "turning-points"
COMMAND_PREPARE = "prepare"
COMMAND_EPISODE_GENERATE = "episode-generate"
COMMAND_BOARD_STATE = "board-state"
COMMAND_STATE_NOW = "state-now"

#: state-now 缺省 episode 参数文件（不存在时用内置默认 EpisodeParams）
_EPISODE_LOCAL_CONFIG_PATH = DEFAULT_CONFIG_PATH.parent / "episode.local.yaml"

#: state-now 在线 1m 拉取根数下限（突破动量需相邻两根）
_STATE_NOW_MIN_BARS = 2

#: state-now 在线 1m 缺省拉取根数（≥ 一个完整交易日+夜盘，v2701 约 345 根/日）
_STATE_NOW_DEFAULT_BARS = 512

#: state-now 在线 1d 缺省拉取根数（训练窗口 2026-01-19~09-30 共 171 根，留余量）
_STATE_NOW_DEFAULT_DAILY_BARS = 300


class UsageError(Exception):
    """命令行用法错误（参数组合非法）。"""


def ohlcv_dir(base_dir: str | Path) -> Path:
    """K 线目录：``<output_dir>/ohlcv``。"""
    return Path(base_dir) / OHLCV_SUBDIR


def turning_points_dir(base_dir: str | Path) -> Path:
    """转折点目录：``<output_dir>/turning_points``。"""
    return Path(base_dir) / TURNING_POINTS_SUBDIR


def _add_common_arguments(
    parser: argparse.ArgumentParser, *, window: bool, data_dir: bool
) -> None:
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        required=True,
        metavar="S",
        help="合约代码（可重复，如 DCE.v2701）",
    )
    parser.add_argument(
        "--period",
        required=True,
        metavar="P",
        help=f"K 线周期：{supported_periods_text()}",
    )
    if window:
        parser.add_argument(
            "--bars",
            type=int,
            metavar="N",
            help=(
                f"最近 N 根已收盘 K 线（取值 1~{SERIAL_MAX_DATA_LENGTH}；取上限值时"
                f"天勤序列至多 {SERIAL_MAX_DATA_LENGTH} 根，末根未收盘会被剔除，"
                f"实际至多返回 {SERIAL_MAX_DATA_LENGTH - 1} 根，届时会输出告警）"
            ),
        )
        parser.add_argument(
            "--start", metavar="ISO", help="区间起始（与 --bars 互斥，需与 --end 同时给出）"
        )
        parser.add_argument(
            "--end", metavar="ISO", help="区间结束（与 --bars 互斥，需与 --start 同时给出）"
        )
    parser.add_argument(
        "--initial-direction",
        dest="initial_direction",
        choices=INITIAL_DIRECTION_MODES,
        default=None,
        help="转折点窗口初始方向（默认取配置 initial_direction）",
    )
    if data_dir:
        parser.add_argument(
            "--data-dir",
            dest="data_dir",
            metavar="DIR",
            help=f"K 线输入目录（默认 <output_dir>/{OHLCV_SUBDIR}）",
        )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        metavar="DIR",
        help="产物根目录（默认取配置 output_dir）",
    )
    parser.add_argument("--config", dest="config", metavar="FILE", help="配置文件（YAML）")


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器（三个子命令）。"""
    parser = argparse.ArgumentParser(
        prog="python -m dataset",
        description="MarketSense 数据准备子应用：天勤 K 线落盘 + 转折点提取",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{fetch,turning-points,prepare,episode-generate,board-state}",
    )

    fetch = subparsers.add_parser(
        "fetch", help="在线取数 → 校验 → 落盘 K 线 CSV + 来源指纹"
    )
    _add_common_arguments(fetch, window=True, data_dir=False)

    turning = subparsers.add_parser(
        "turning-points", help="离线读取已落盘 K 线 → 转折点 CSV + 窗口 sidecar"
    )
    _add_common_arguments(turning, window=False, data_dir=True)

    prepare = subparsers.add_parser("prepare", help="fetch 后接转折点提取")
    _add_common_arguments(prepare, window=True, data_dir=False)

    episode = subparsers.add_parser(
        COMMAND_EPISODE_GENERATE,
        help="片段清单 + 已落盘 1 分钟 K 线 → 按 split 的 NanoJev JSONL + 审计",
    )
    episode.add_argument(
        "--segments",
        dest="segments",
        required=True,
        metavar="FILE",
        help="片段清单 JSONL（每行一个片段 = 一个 episode；模板见 dataset/config/segments.example.jsonl）",
    )
    episode.add_argument(
        "--output-dir",
        dest="output_dir",
        metavar="DIR",
        help="产物根目录（默认取配置 episode.output_dir，产物落在 <DIR>/<run_id>/）",
    )
    episode.add_argument(
        "--daily-turning-points-dir",
        dest="daily_turning_points_dir",
        metavar="DIR",
        help="日线折点目录（v5 日线行 trend 四值数据源；默认 <data_dir>/../turning_points）",
    )
    episode.add_argument(
        "--event-lookforward-open",
        dest="event_lookforward_open",
        type=int,
        metavar="N",
        default=None,
        help="覆盖配置文件 episode.event_lookforward_open（train/dev 开仓类事件前看 N 条决策记录；单位 = 决策记录条数；仅 train/dev 生效）",
    )
    episode.add_argument(
        "--event-lookforward-exit",
        dest="event_lookforward_exit",
        type=int,
        metavar="N",
        default=None,
        help="覆盖配置文件 episode.event_lookforward_exit（train/dev 离场类事件前看 N 条决策记录；单位 = 决策记录条数；仅 train/dev 生效）",
    )
    episode.add_argument("--config", dest="config", metavar="FILE", help="配置文件（YAML）")

    board = subparsers.add_parser(
        COMMAND_BOARD_STATE,
        help="离线读取已落盘 1m/1d K 线 → 盘面状态 CSV + sidecar（不联网）",
    )
    board.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        required=True,
        metavar="S",
        help="合约代码（可重复，如 DCE.v2701）",
    )
    board.add_argument(
        "--period",
        default=BOARD_STATE_PERIOD,
        metavar="P",
        help=f"K 线周期（目前仅支持 {BOARD_STATE_PERIOD}，窗口语义按 1m 交易日口径定义）",
    )
    board.add_argument(
        "--start",
        metavar="DATE",
        help="只生成 ≥ 该交易所交易日的状态（ISO 日期，与 --end 同时给出）",
    )
    board.add_argument(
        "--end",
        metavar="DATE",
        help="只生成 ≤ 该交易所交易日的状态（ISO 日期，与 --start 同时给出）",
    )
    board.add_argument(
        "--data-dir",
        dest="data_dir",
        metavar="DIR",
        help=f"K 线输入目录（默认 <output_dir>/{OHLCV_SUBDIR}）",
    )
    board.add_argument(
        "--output-dir",
        dest="output_dir",
        metavar="DIR",
        help="产物根目录（默认取配置 output_dir）",
    )
    board.add_argument("--config", dest="config", metavar="FILE", help="配置文件（YAML）")

    state_now = subparsers.add_parser(
        COMMAND_STATE_NOW,
        help="构造「此刻」state/question/candidates/metadata 快照（stdout，不落盘）",
    )
    state_now.add_argument(
        "--symbol",
        required=True,
        metavar="S",
        help="主品种（如 DCE.v2701）",
    )
    state_now.add_argument(
        "--offline",
        action="store_true",
        help="退到已落盘数据（1m CSV 末行为决策 K 线；默认在线拉取）",
    )
    state_now.add_argument(
        "--full",
        action="store_true",
        help="watch 输出完整布局；--once 已固定输出完整布局，此参数在该模式无影响",
    )
    state_now.add_argument(
        "--once",
        action="store_true",
        help="输出既有完整布局一次后退出（不加时间戳头；--full 无影响）",
    )
    state_now.add_argument(
        "--period",
        default=None,
        metavar="P",
        help=(
            f"watch 触发周期（默认 1m；支持 {supported_periods_text()}，1d 拒绝；"
            "快照内容恒为 1m 决策 K 线口径；仅 watch 模式有意义）"
        ),
    )
    state_now.add_argument(
        "--bars",
        type=int,
        default=_STATE_NOW_DEFAULT_BARS,
        metavar="N",
        help=(
            f"在线 1m 拉取根数（默认 {_STATE_NOW_DEFAULT_BARS}，"
            f"范围 {_STATE_NOW_MIN_BARS}~{SERIAL_MAX_DATA_LENGTH}）"
        ),
    )
    state_now.add_argument(
        "--daily-bars",
        dest="daily_bars",
        type=int,
        default=_STATE_NOW_DEFAULT_DAILY_BARS,
        metavar="N",
        help=(
            f"在线 1d 拉取根数（默认 {_STATE_NOW_DEFAULT_DAILY_BARS}，"
            f"范围 1~{SERIAL_MAX_DATA_LENGTH}；只含已收盘日线）"
        ),
    )
    state_now.add_argument(
        "--episode-config",
        dest="episode_config",
        metavar="FILE",
        help=(
            "episode 参数 YAML（读 episode: 段；缺省 dataset/config/episode.local.yaml，"
            "不存在时用内置默认）"
        ),
    )
    state_now.add_argument(
        "--data-dir",
        dest="data_dir",
        metavar="DIR",
        help=(
            "K 线目录（离线 1m/1d 输入与折点默认目录锚点；"
            f"默认 <output_dir>/{OHLCV_SUBDIR}）"
        ),
    )
    state_now.add_argument(
        "--turning-points-dir",
        dest="turning_points_dir",
        metavar="DIR",
        help="日线折点目录（默认 <data_dir>/../turning_points）",
    )
    state_now.add_argument("--config", dest="config", metavar="FILE", help="配置文件（YAML）")
    return parser


def _resolve_window(args: argparse.Namespace) -> tuple[int | None, str | None, str | None]:
    """校验窗口参数：``--bars`` 与 ``--start/--end`` 互斥且必须给其一。"""
    bars = getattr(args, "bars", None)
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)

    if bars is not None and (start is not None or end is not None):
        raise UsageError("--bars 与 --start/--end 互斥，请只提供其中一种窗口")
    if bars is None and not (start and end):
        raise UsageError("必须提供 --bars N，或同时提供 --start 与 --end")

    if bars is not None:
        try:
            validate_data_length(bars)
        except DatasetError as exc:
            raise UsageError(str(exc)) from exc
    if start is not None and end is not None:
        parse_date_bound(start, "start")  # 非法 ISO 直接抛 DatasetError
        parse_date_bound(end, "end")
    return bars, start, end


def _fetch_frame(
    provider: TianQinProvider,
    symbol: str,
    period: str,
    bars: int | None,
    start: str | None,
    end: str | None,
):
    """按窗口模式取数：``--bars`` 走最近 N 根，``--start/--end`` 走历史区间。"""
    if bars is not None:
        return provider.fetch_recent(symbol, period, bars)
    return provider.fetch_history(symbol, period, start, end)


def _warn_if_bars_shortfall(symbol: str, requested: int, actual: int) -> None:
    """实际返回根数少于请求时**显式告警**（不静默），并按真实原因给出归因。

    两类原因区分（P2-N4）：

    * 请求已达 ``SERIAL_MAX_DATA_LENGTH`` 上限：``get_kline_serial`` 无法再多取 1 根，
      末根未收盘被剔除后实际至多 N-1 根；
    * 其它：数据源自身提供的已收盘 K 线不足（与上限无关），不得归因于 8964。
    """
    message = _bars_shortfall_message(symbol, requested, actual)
    if message is not None:
        print(message, file=sys.stderr)


def _bars_shortfall_message(symbol: str, requested: int, actual: int) -> str | None:
    """shortfall 告警文本（无 shortfall 返回 ``None``）；文本由
    ``(品种, 请求, 实得)`` 组合唯一决定，watch 模式据此按文本去重。"""
    if actual >= requested:
        return None
    if requested >= SERIAL_MAX_DATA_LENGTH:
        reason = (
            f"请求已达 --bars 上限 {SERIAL_MAX_DATA_LENGTH}：天勤序列至多含 "
            f"{SERIAL_MAX_DATA_LENGTH} 根，末根未收盘会被剔除"
        )
    else:
        reason = (
            f"数据源返回的已收盘 K 线不足（请求 {requested} 根，实得 {actual} 根），"
            "与 --bars 上限无关"
        )
    return (
        f"警告：{symbol} 请求 {requested} 根已收盘 K 线，实际返回 {actual} 根"
        f"（{reason}；需要更早数据请改用 --start/--end）"
    )


def _print_stderr_warn(text: str) -> None:
    """stderr 告警出口（一次性路径/循环层失败计数用，不去重）。"""
    print(text, file=sys.stderr)


def _extract_turning_points(
    *,
    symbol: str,
    period: str,
    data_dir: str | Path,
    output_dir: str | Path,
    requested_direction: str,
) -> Path:
    """离线读取 K 线 → 校验 → 提取转折点 → 落盘（不联网）。"""
    loaded = load_ohlcv(symbol, period, data_dir=data_dir)
    validate_ohlcv(loaded.df).raise_if_invalid()
    resolved = resolve_initial_direction(loaded.df, requested_direction)
    points = find_turning_points(loaded.df, initial_direction=resolved)
    window_meta = build_window_meta(
        loaded.df,
        initial_direction_requested=requested_direction,
        initial_direction_resolved=resolved,
        input_source_data_version=loaded.source_data_version,
    )
    return save_turning_points(
        points,
        symbol=symbol,
        period=period,
        output_dir=output_dir,
        window_meta=window_meta,
    )


def _run_episode_generate(args: argparse.Namespace) -> int:
    """``episode-generate``：离线生成 episode 训练数据（不联网、不触天勤凭证）。"""
    config = load_episode_config(args.config)
    # 采样参数 CLI 覆盖（label-band-sampling）：CLI 给值优先于配置文件/内置默认；
    # 非法值（负数/非整数）在进入生成前拒绝（UsageError → 退出码 2）
    params = config.params
    for name in ("event_lookforward_open", "event_lookforward_exit"):
        value = getattr(args, name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise UsageError(f"--{name.replace('_', '-')} 必须为非负整数，实际: {value!r}")
        params = dataclasses.replace(params, **{name: value})
    segments = load_segments(Path(args.segments))
    symbols = load_symbols_config(config.symbols_path)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    # v5 日线折点目录：显式参数优先，否则取 <data_dir>/../turning_points
    # （turning-points 子命令落盘布局； episode data_dir 默认 <base>/ohlcv）
    daily_turning_points_dir = (
        Path(args.daily_turning_points_dir)
        if args.daily_turning_points_dir
        else turning_points_dir(Path(config.data_dir).parent)
    )
    result = generate_dataset(
        segments,
        symbols=symbols,
        data_dir=config.data_dir,
        params=params,
        output_dir=output_dir,
        daily_turning_points_dir=daily_turning_points_dir,
    )
    counts = ", ".join(
        f"{split}={result.record_counts[split]}" for split in SPLIT_ROLES
    )
    for segment_id, reason in sorted(result.board_state_skipped.items()):
        print(f"[board-state 跳过片段] {segment_id}: {reason}", file=sys.stderr)
    if result.board_state_skipped:
        print(
            f"共跳过 {len(result.board_state_skipped)} 个片段（缺上一交易日日线，详见 stderr）",
            file=sys.stderr,
        )
    for segment_id, reason in sorted(result.trend_extreme_skipped.items()):
        print(f"[trend-extreme 跳过片段] {segment_id}: {reason}", file=sys.stderr)
    if result.trend_extreme_skipped:
        print(
            f"共跳过 {len(result.trend_extreme_skipped)} 个片段"
            "（缺可用日线转折点，详见 stderr）",
            file=sys.stderr,
        )
    print(f"已生成 episode 训练数据：{result.run_dir}（记录数 {counts}）")
    outcome_total = sum(result.outcome_row_counts.values())
    outcome_counts = ", ".join(
        f"{split}={result.outcome_row_counts[split]}" for split in SPLIT_ROLES
    )
    print(
        f"outcome 旁挂：{result.run_dir / OUTCOMES_FILENAME}（开仓行 {outcome_total}: {outcome_counts}）"
    )
    print(f"审计文件：{result.run_dir / AUDIT_FILENAME}")
    return EXIT_OK


def _run_board_state(args: argparse.Namespace) -> int:
    """``board-state``：离线构建盘面状态序列（不联网、不触天勤凭证）。"""
    if args.period != BOARD_STATE_PERIOD:
        raise UsageError(
            f"board-state 目前只支持 --period {BOARD_STATE_PERIOD}（窗口语义按 1m 交易日口径定义）"
        )
    if (args.start is None) != (args.end is None):
        raise UsageError("--start 与 --end 必须同时提供（或都不提供）")
    start = end = None
    if args.start is not None:
        start = parse_date_bound(args.start, "start").date()
        end = parse_date_bound(args.end, "end").date()
        if start > end:
            raise UsageError(f"--start 不能晚于 --end: {start} > {end}")
    config = load_dataset_config(args.config)
    base_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    data_dir = (
        Path(args.data_dir) if getattr(args, "data_dir", None) else ohlcv_dir(base_dir)
    )
    for symbol in args.symbols:
        minute = load_ohlcv(symbol, BOARD_STATE_PERIOD, data_dir=data_dir)
        validate_ohlcv(minute.df).raise_if_invalid()
        daily_path = data_dir / ohlcv_filename(symbol, "1d")
        if not daily_path.is_file():
            raise DatasetError(
                f"盘面状态需要日线数据（前日高/低/收来源）: {daily_path}；"
                "请先 `python -m dataset fetch --period 1d` 落盘日线"
            )
        daily = load_ohlcv(symbol, "1d", data_dir=data_dir)
        validate_ohlcv(daily.df).raise_if_invalid()
        result = build_board_states(
            minute.df, daily.df, symbol=symbol, start=start, end=end
        )
        if result.unassigned_bars:
            print(
                f"警告：{symbol} 有 {result.unassigned_bars} 根 K 线时间在 15:00–20:59，"
                "不属于任何交易窗口，已排除（请确认数据时段是否预期）",
                file=sys.stderr,
            )
        for note in result.skipped_days:
            print(f"警告：{symbol} 跳过交易日 {note}", file=sys.stderr)
        path = save_board_states(
            result,
            symbol=symbol,
            period=BOARD_STATE_PERIOD,
            output_dir=base_dir,
            minute_path=minute.path,
            daily_path=daily.path,
        )
        print(
            f"已生成盘面状态：{path}（{len(result.df)} 行，"
            f"{len(result.trade_days)} 个交易日）"
        )
    return EXIT_OK


def _load_state_now_params(args: argparse.Namespace) -> EpisodeParams:
    """state-now 的 episode 参数来源：``--episode-config`` 显式文件 >
    ``dataset/config/episode.local.yaml``（存在时）> 内置默认。"""
    if args.episode_config:
        path = Path(args.episode_config)
    elif _EPISODE_LOCAL_CONFIG_PATH.is_file():
        path = _EPISODE_LOCAL_CONFIG_PATH
    else:
        return EpisodeParams()
    if not path.is_file():
        raise ConfigError(f"episode 参数文件不存在: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("episode") if isinstance(payload, dict) else None
    if not isinstance(section, dict):
        raise ConfigError(f"episode 参数文件缺少 episode: 段: {path}")
    return load_episode_params(section, where=f"--episode-config {path}")


def _validate_state_now_mode(args: argparse.Namespace) -> None:
    """state-now 模式面校验（先于深度/配置加载，exit 2；设计 §3.5）：

    * ``--period 1d`` 拒绝（快照基于 1m 决策 K 线，1d 一根一输出无监控意义）；
    * 未知周期拒绝（错误含支持列表）；
    * ``--period`` 与 ``--once`` 同给拒绝（period 仅 watch 有意义，不静默忽略）；
    * ``--offline`` 无 ``--once`` 拒绝（离线数据运行期不变，无 watch 意义）。
    """
    if args.period is not None:
        if args.period == "1d":
            raise UsageError(
                "--period 1d 不支持：state-now 快照基于 1m 决策 K 线"
                "（1d 一根一输出无监控意义）；--period 支持 1m/5m/15m/1h"
            )
        try:
            resolve_duration_seconds(args.period)
        except UnknownPeriodError as exc:
            raise UsageError(f"--period {exc}") from exc
        if args.once:
            raise UsageError(
                "--period 仅在 watch 模式（默认）有意义，与 --once 同给无意义；"
                "单次模式请去掉 --period"
            )
    if args.offline and not args.once:
        raise UsageError(
            "--offline 数据运行期不变，无 watch 意义；单次快照请加 --once"
        )


def _validate_state_now_depth(args: argparse.Namespace) -> None:
    """校验在线拉取深度：``--bars`` ≥2（突破动量需相邻两根）、深度在平台范围内。"""
    for flag, value in (("--bars", args.bars), ("--daily-bars", args.daily_bars)):
        try:
            validate_data_length(value)
        except DatasetError as exc:
            raise UsageError(f"{flag} {exc}") from exc
    if args.bars < _STATE_NOW_MIN_BARS:
        raise UsageError(
            f"--bars 至少为 {_STATE_NOW_MIN_BARS}（突破动量需相邻两根），实际: {args.bars}"
        )


def _fetch_1m_frame(
    provider: TianQinProvider,
    *,
    symbol: str,
    bars: int,
    warn: Callable[[str], None],
) -> Any:
    """主品种 1m 帧：取数 → shortfall 告警 → 校验（顺序与一次性路径一致）。"""
    frame = provider.fetch_recent(symbol, "1m", bars)
    message = _bars_shortfall_message(symbol, bars, len(frame))
    if message is not None:
        warn(message)
    validate_ohlcv(frame).raise_if_invalid()
    return frame


def _fetch_rest_frames(
    provider: TianQinProvider,
    *,
    symbol: str,
    bars: int,
    daily_bars: int,
    params: EpisodeParams,
    warn: Callable[[str], None],
) -> tuple[Any, dict[str, Any | None], dict[str, Any]]:
    """主品种 1d、参考 1m 与参考 1d；参考日线取数失败硬错。"""
    frame_1d = provider.fetch_recent(symbol, "1d", daily_bars)
    message = _bars_shortfall_message(symbol, daily_bars, len(frame_1d))
    if message is not None:
        warn(message)
    validate_ohlcv(frame_1d).raise_if_invalid()
    linkage: dict[str, Any | None] = {}
    linkage_daily: dict[str, Any] = {}
    for link_symbol in params.linkage_symbols:
        try:
            frame = provider.fetch_recent(link_symbol, "1m", bars)
            validate_ohlcv(frame).raise_if_invalid()
            message = _bars_shortfall_message(link_symbol, bars, len(frame))
            if message is not None:
                warn(message)
        except DatasetError as exc:
            warn(f"警告：联动品种 {link_symbol} 在线取数失败，按 na 降级（{exc}）")
            frame = None
        linkage[link_symbol] = frame
        try:
            daily = provider.fetch_recent(link_symbol, "1d", daily_bars)
            validate_ohlcv(daily).raise_if_invalid()
        except DatasetError as exc:
            raise DatasetError(
                f"联动品种 {link_symbol!r} 在线 1d 取数/校验失败，不能计算状态：{exc}"
            ) from exc
        linkage_daily[link_symbol] = daily
    return frame_1d, linkage, linkage_daily


def _fetch_frames_with_provider(
    provider: TianQinProvider,
    *,
    symbol: str,
    bars: int,
    daily_bars: int,
    params: EpisodeParams,
    warn: Callable[[str], None],
) -> tuple[Any, Any, dict[str, Any | None], dict[str, Any]]:
    """已连接 provider 的取数内层（1m → 1d → 联动及联动 1d）。"""
    frame_1m = _fetch_1m_frame(provider, symbol=symbol, bars=bars, warn=warn)
    frame_1d, linkage, linkage_daily = _fetch_rest_frames(
        provider,
        symbol=symbol,
        bars=bars,
        daily_bars=daily_bars,
        params=params,
        warn=warn,
    )
    return frame_1m, frame_1d, linkage, linkage_daily


def _fetch_state_now_frames(
    args: argparse.Namespace,
    config: DatasetConfig,
    params: EpisodeParams,
    api_factory: Callable[[DatasetConfig], Any] | None,
) -> tuple[Any, Any, int, str, dict[str, Any | None], dict[str, Any]]:
    """读取主/参考品种数据；配置的参考日线缺失/无效时显式失败。"""
    data_dir = (
        Path(args.data_dir) if args.data_dir else ohlcv_dir(config.output_dir)
    )
    if not args.offline:
        provider = TianQinProvider(config, api_factory)
        try:
            provider.connect()
            frame_1m, frame_1d, linkage, linkage_daily = _fetch_frames_with_provider(
                provider,
                symbol=args.symbol,
                bars=args.bars,
                daily_bars=args.daily_bars,
                params=params,
                warn=_print_stderr_warn,
            )
        finally:
            provider.close()
        return frame_1m, frame_1d, len(frame_1m) - 1, "online", linkage, linkage_daily

    loaded_1m = load_ohlcv(args.symbol, "1m", data_dir=data_dir)
    validate_ohlcv(loaded_1m.df).raise_if_invalid()
    loaded_1d = load_ohlcv(args.symbol, "1d", data_dir=data_dir)
    validate_ohlcv(loaded_1d.df).raise_if_invalid()
    linkage = {}
    linkage_daily = {}
    for link_symbol in params.linkage_symbols:
        try:
            frame = load_ohlcv(link_symbol, "1m", data_dir=data_dir).df
            validate_ohlcv(frame).raise_if_invalid()
        except DatasetError as exc:
            print(
                f"警告：联动品种 {link_symbol} 离线读取失败，按 na 降级（{exc}）",
                file=sys.stderr,
            )
            frame = None
        linkage[link_symbol] = frame
        try:
            loaded_daily = load_ohlcv(link_symbol, "1d", data_dir=data_dir)
            validate_ohlcv(loaded_daily.df).raise_if_invalid()
        except DatasetError as exc:
            raise DatasetError(
                f"联动品种 {link_symbol!r} 的 1d 日线文件缺失或无效；请执行 "
                f"`python -m dataset fetch --symbol {link_symbol} --period 1d --bars <N>`"
            ) from exc
        linkage_daily[link_symbol] = loaded_daily.df
    return loaded_1m.df, loaded_1d.df, len(loaded_1m.df) - 1, "offline", linkage, linkage_daily


def _print_snapshot(
    snapshot: LiveStateSnapshot, *, data_source: str, full: bool
) -> None:
    """确定性 stdout；``full=False`` 仅行情五部分（watch 缺省显示层，设计 §3.4：
    按行前缀过滤 builder state 文本，schema/账户/question/candidates/metadata 不打印），
    ``full=True`` 为完整布局（既有 ``--once`` 验收形态逐字节不变）。
    metadata 在 symbol 后插入 data_source
    （CLI 层职责：builder 签名与 snapshot 本身不含该键）。"""
    if not full:
        for line in market_lines(snapshot.state_text):
            print(line)
        return
    print("=== state ===")
    print(snapshot.state_text)
    print("=== question ===")
    question = snapshot.question[QUESTION_ID]
    print(
        f"next_action: type={question['type']}  instructions={question['instructions']}"
    )
    print("=== candidates ===")
    print(" | ".join(f"{action}={text}" for action, text in snapshot.candidates.items()))
    print("=== metadata ===")
    printed: dict[str, Any] = {}
    for key, value in snapshot.metadata.items():
        printed[key] = value
        if key == "symbol":
            printed["data_source"] = data_source
    print(json.dumps(printed, ensure_ascii=False, indent=2))
    print(f"注：{DENOMINATOR_NOTE}")


def _default_now_fn() -> pd.Timestamp:
    """生产墙钟（北京时间；边界上取整在 :func:`dataset.watch_state.next_boundary`）。"""
    return pd.Timestamp.now(tz=TIMEZONE)


def _load_state_now_turning_points(args: argparse.Namespace, config: DatasetConfig) -> Any:
    """折点 CSV 加载（一次性/watch 两模式同源；缺失 → 明确报错并给生成命令）。"""
    data_dir = Path(args.data_dir) if args.data_dir else ohlcv_dir(config.output_dir)
    tp_dir = (
        Path(args.turning_points_dir)
        if args.turning_points_dir
        else turning_points_dir(Path(data_dir).parent)
    )
    try:
        return load_turning_points(args.symbol, "1d", data_dir=tp_dir)
    except DataLoadError as exc:
        raise DatasetError(
            f"日线折点数据缺失（{exc}）；"
            f"请先 `python -m dataset turning-points --symbol {args.symbol} --period 1d` 生成"
        ) from exc


def _run_state_now(
    args: argparse.Namespace,
    *,
    api_factory: Callable[[DatasetConfig], Any] | None,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    wait_fn: Callable[[float], None] | None = None,
) -> int:
    """``state-now`` 分发：``--once``/``--offline`` = 完整布局单次快照（现行为逐字节不变）；
    默认 = watch 循环（长连接会话复用，设计 §3.2/§3.3）。

    :param now_fn: 墙钟注入（测试假时钟；缺省北京时间 ``pd.Timestamp.now``）
    :param wait_fn: 边界等待注入（测试假等待；缺省 ``provider.wait_until``）
    """
    _validate_state_now_mode(args)
    _validate_state_now_depth(args)
    params = _load_state_now_params(args)
    config = load_dataset_config(args.config)
    if args.once or args.offline:
        return _run_state_now_once(
            args, config=config, params=params, api_factory=api_factory
        )
    return _run_state_now_watch(
        args,
        config=config,
        params=params,
        api_factory=api_factory,
        now_fn=now_fn,
        wait_fn=wait_fn,
    )


def _run_state_now_once(
    args: argparse.Namespace,
    *,
    config: DatasetConfig,
    params: EpisodeParams,
    api_factory: Callable[[DatasetConfig], Any] | None,
) -> int:
    """单次快照（``--once`` 在线/离线；state-now-view 已验收行为不回退）。"""
    frame_1m, frame_1d, decision_index, data_source, linkage, linkage_daily = _fetch_state_now_frames(
        args, config, params, api_factory
    )
    loaded_tp = _load_state_now_turning_points(args, config)
    snapshot = build_live_state(
        symbol=args.symbol,
        bars_1m=frame_1m,
        decision_index=decision_index,
        daily_rows=frame_1d,
        turning_points=loaded_tp.points,
        linkage_bars=linkage or None,
        linkage_daily_rows=linkage_daily or None,
        params=params,
        price_precision=params.price_precision,
    )
    metadata = snapshot.metadata
    if metadata["today_window"]["truncated"]:
        print(
            "警告：今日窗口可能被截断（帧首根仍属今日窗口且时刻晚于夜盘起点 21:00，"
            "今日开盘分母可能非真实开盘价）；增大 --bars 可消除",
            file=sys.stderr,
        )
    # --once preserves the previously accepted complete layout; --full is a no-op here.
    _print_snapshot(snapshot, data_source=data_source, full=True)
    return EXIT_OK


def _run_state_now_watch(
    args: argparse.Namespace,
    *,
    config: DatasetConfig,
    params: EpisodeParams,
    api_factory: Callable[[DatasetConfig], Any] | None,
    now_fn: Callable[[], pd.Timestamp] | None,
    wait_fn: Callable[[float], None] | None,
) -> int:
    """watch（默认）：connect 一次长连接复用 → :func:`run_watch` 边界判重循环
    （设计 §3.2/§3.3）。

    折点 CSV/episode 参数启动加载一次（运行期不重载，重生成需重启）；
    shortfall/联动降级告警按同一文本只告警一次（文本由 ``(品种, 请求, 实得)``
    组合唯一决定；非交易时段长跑不刷屏）；退出码：Ctrl+C 130 / 连续 3 次失败 1；
    连接释放由 ``finally`` 保证（幂等）。
    """
    loaded_tp = _load_state_now_turning_points(args, config)
    seen_warnings: set[str] = set()

    def dedupe_warn(text: str) -> None:
        if text in seen_warnings:
            return
        seen_warnings.add(text)
        print(text, file=sys.stderr)

    period = args.period or "1m"
    provider = TianQinProvider(config, api_factory)
    try:
        provider.connect()

        def fetch_1m() -> pd.DataFrame:
            return _fetch_1m_frame(
                provider,
                symbol=args.symbol,
                bars=args.bars,
                warn=dedupe_warn,
            )

        def fetch_rest() -> tuple[pd.DataFrame, dict[str, Any | None], dict[str, pd.DataFrame]]:
            return _fetch_rest_frames(
                provider,
                symbol=args.symbol,
                bars=args.bars,
                daily_bars=args.daily_bars,
                params=params,
                warn=dedupe_warn,
            )

        def render(snapshot: LiveStateSnapshot, decision_key: pd.Timestamp) -> None:
            print(timestamp_header(args.symbol, period, decision_key))
            _print_snapshot(snapshot, data_source="online", full=args.full)

        return run_watch(
            symbol=args.symbol,
            period=period,
            bars=args.bars,
            daily_bars=args.daily_bars,
            params=params,
            turning_points=loaded_tp.points,
            price_precision=params.price_precision,
            full=args.full,
            fetch_1m=fetch_1m,
            fetch_rest=fetch_rest,
            render=render,
            warn=_print_stderr_warn,
            now_fn=now_fn if now_fn is not None else _default_now_fn,
            wait_fn=wait_fn if wait_fn is not None else provider.wait_until,
        )
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    finally:
        provider.close()


def _run(
    args: argparse.Namespace,
    *,
    api_factory: Callable[[DatasetConfig], Any] | None,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    wait_fn: Callable[[float], None] | None = None,
) -> int:
    """执行子命令（配置加载前先校验周期与窗口参数，保证错误信息可操作）。"""
    if args.command == COMMAND_EPISODE_GENERATE:
        # 该子命令不涉及天勤凭证与窗口参数，先于周期/窗口校验分发
        return _run_episode_generate(args)
    if args.command == COMMAND_BOARD_STATE:
        # 该子命令不涉及天勤凭证，先于周期/窗口校验分发
        return _run_board_state(args)
    if args.command == COMMAND_STATE_NOW:
        # 该子命令自带参数面（无 --period/--start/--end 公共校验），先于其分发
        return _run_state_now(
            args, api_factory=api_factory, now_fn=now_fn, wait_fn=wait_fn
        )
    resolve_duration_seconds(args.period)  # 未知周期 → UnknownPeriodError（含支持列表）
    if args.command == COMMAND_TURNING_POINTS:  # 离线路径无窗口参数
        bars: int | None = None
        start: str | None = None
        end: str | None = None
    else:
        bars, start, end = _resolve_window(args)
    config = load_dataset_config(args.config)
    base_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    requested_direction = args.initial_direction or config.initial_direction

    if args.command == COMMAND_TURNING_POINTS:
        input_dir = (
            Path(args.data_dir) if getattr(args, "data_dir", None) else ohlcv_dir(base_dir)
        )
        for symbol in args.symbols:
            path = _extract_turning_points(
                symbol=symbol,
                period=args.period,
                data_dir=input_dir,
                output_dir=turning_points_dir(base_dir),
                requested_direction=requested_direction,
            )
            print(f"已生成转折点：{path}")
        return EXIT_OK

    provider = TianQinProvider(config, api_factory)
    ohlcv_paths: list[Path] = []
    try:
        provider.connect()
        for symbol in args.symbols:
            df = _fetch_frame(provider, symbol, args.period, bars, start, end)
            if bars is not None:
                _warn_if_bars_shortfall(symbol, bars, len(df))
            validate_ohlcv(df).raise_if_invalid()
            if bars is not None:
                # 最近 N 根模式无请求区间 → sidecar 记录实际窗口首末时间
                window_start = str(df["timestamp"].iloc[0])
                window_end = str(df["timestamp"].iloc[-1])
            else:
                window_start, window_end = start, end
            path = save_ohlcv(
                df,
                symbol=symbol,
                period=args.period,
                output_dir=ohlcv_dir(base_dir),
                start=window_start,
                end=window_end,
            )
            print(f"已落盘 K 线：{path}（{len(df)} 行）")
            ohlcv_paths.append(path)
    finally:
        provider.close()

    if args.command == COMMAND_PREPARE:
        for symbol in args.symbols:
            path = _extract_turning_points(
                symbol=symbol,
                period=args.period,
                data_dir=ohlcv_dir(base_dir),
                output_dir=turning_points_dir(base_dir),
                requested_direction=requested_direction,
            )
            print(f"已生成转折点：{path}")
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    api_factory: Callable[[DatasetConfig], Any] | None = None,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    wait_fn: Callable[[float], None] | None = None,
) -> int:
    """CLI 入口。

    :param argv: 参数列表（缺省取 ``sys.argv[1:]``）
    :param api_factory: 底层 API 工厂（测试注入桩；生产为 ``None`` → TqApi）
    :param now_fn: watch 墙钟注入（测试假时钟；仅 state-now watch 消费）
    :param wait_fn: watch 边界等待注入（测试假等待；仅 state-now watch 消费）
    :return: 退出码（0 成功 / 1 运行期失败 / 2 用法错误）
    """
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse：--help 或用法错误
        code = exc.code
        if code is None:
            return EXIT_OK
        return code if isinstance(code, int) else EXIT_USAGE

    try:
        return _run(args, api_factory=api_factory, now_fn=now_fn, wait_fn=wait_fn)
    except UsageError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_USAGE
    except DatasetError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_FAILURE
    except ValueError as exc:  # 转折点算法的参数误用
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_FAILURE
