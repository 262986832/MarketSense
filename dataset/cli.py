"""dataset 命令行入口（``python -m dataset``）。

五个子命令：

```text
fetch          在线取数 → 校验 → 落盘 K 线 CSV + 来源指纹 sidecar
turning-points 离线读取已落盘 K 线 → 转折点 CSV + 窗口 sidecar（不联网）
prepare        fetch 后接转折点提取
episode-generate  片段清单 + 已落盘 1 分钟 K 线 → 按 split 的 NanoJev JSONL + 审计
board-state    离线读取已落盘 1m/1d K 线 → 盘面状态 CSV + sidecar（不联网）
```

退出码：``0`` 成功；``1`` 运行期失败（配置/凭证/取数/校验/读取）；``2`` 用法错误
（参数组合非法）。错误信息一律写 stderr，且**不含**凭证值。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from dataset.board_state import (
    BOARD_STATE_SUBDIR,
    build_board_states,
    save_board_states,
)
from dataset.config import DatasetConfig, load_dataset_config
from dataset.errors import DatasetError
from dataset.market_episode import (
    AUDIT_FILENAME,
    SPLIT_ROLES,
    generate_dataset,
    load_episode_config,
    load_segments,
    load_symbols_config,
)
from dataset.ohlcv import parse_date_bound
from dataset.periods import resolve_duration_seconds, supported_periods_text
from dataset.provider import (
    SERIAL_MAX_DATA_LENGTH,
    TianQinProvider,
    validate_data_length,
)
from dataset.storage import load_ohlcv, ohlcv_filename, save_ohlcv
from dataset.turning_points import (
    INITIAL_DIRECTION_MODES,
    build_window_meta,
    find_turning_points,
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
    if actual >= requested:
        return
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
    print(
        f"警告：{symbol} 请求 {requested} 根已收盘 K 线，实际返回 {actual} 根"
        f"（{reason}；需要更早数据请改用 --start/--end）",
        file=sys.stderr,
    )


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
    segments = load_segments(Path(args.segments))
    symbols = load_symbols_config(config.symbols_path)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    result = generate_dataset(
        segments,
        symbols=symbols,
        data_dir=config.data_dir,
        params=config.params,
        output_dir=output_dir,
    )
    counts = ", ".join(
        f"{split}={result.record_counts[split]}" for split in SPLIT_ROLES
    )
    print(f"已生成 episode 训练数据：{result.run_dir}（记录数 {counts}）")
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


def _run(
    args: argparse.Namespace, *, api_factory: Callable[[DatasetConfig], Any] | None
) -> int:
    """执行子命令（配置加载前先校验周期与窗口参数，保证错误信息可操作）。"""
    if args.command == COMMAND_EPISODE_GENERATE:
        # 该子命令不涉及天勤凭证与窗口参数，先于周期/窗口校验分发
        return _run_episode_generate(args)
    if args.command == COMMAND_BOARD_STATE:
        # 该子命令不涉及天勤凭证，先于周期/窗口校验分发
        return _run_board_state(args)
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
    argv: Sequence[str] | None = None, *, api_factory: Callable[[DatasetConfig], Any] | None = None
) -> int:
    """CLI 入口。

    :param argv: 参数列表（缺省取 ``sys.argv[1:]``）
    :param api_factory: 底层 API 工厂（测试注入桩；生产为 ``None`` → TqApi）
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
        return _run(args, api_factory=api_factory)
    except UsageError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_USAGE
    except DatasetError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_FAILURE
    except ValueError as exc:  # 转折点算法的参数误用
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_FAILURE
