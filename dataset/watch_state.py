"""``state-now`` watch 模式：显示层纯函数 + 循环驱动（tech-design §3.3/§3.4）。

* :func:`market_lines`：builder ``state_text`` → 行情五部分行（裁剪显示，逐字节原文）；
* :func:`next_boundary`：绝对 epoch 纳秒上取整到周期网格（tz 原样保留）；
* :func:`timestamp_header`：watch 输出前缀 ``[YYYY-MM-DD HH:MM symbol period]``
  （时间 = 决策 K 线起点 + 60s，北京时间）；
* :func:`run_watch`：循环驱动——时钟/等待/取数/渲染全部注入，本模块无 I/O。

判重键 = 主品种最新已收盘 1m bar 时间戳（设计 D3）：同根静默、新根输出一次、
数据延迟下一轮补出（一次只输出最新一根）。退出策略（设计 §3.6）：单周期
``DatasetError`` → stderr 警告 + 继续，连续 ≥ :data:`_WATCH_MAX_CONSECUTIVE_FAILURES`
次 → 退出码 1；``KeyboardInterrupt`` → 退出码 :data:`EXIT_INTERRUPTED`（130，
由调用方 ``finally`` 负责释放连接）。
"""

from __future__ import annotations

from typing import Any, Callable, Final

import pandas as pd

from dataset.errors import DatasetError
from dataset.live_state import LiveStateSnapshot, build_live_state
from dataset.market_episode import EpisodeParams
from dataset.periods import resolve_duration_seconds
from dataset.turning_points import TurningPoint

#: 裁剪保留的行情行前缀（行首首个 ``":"`` 之前的部分）
MARKET_LINE_PREFIXES: tuple[str, ...] = ("日线", "日内", "联动", "现价", "盘口")

#: builder state 文本段间连接符（``render_state`` v3+ 口径：换行符前后各一空格）
_SEGMENT_SEPARATOR = " \n "

#: watch 单周期连续失败上限（工程稳健性常数，非研究阈值；设计 §3.6）
_WATCH_MAX_CONSECUTIVE_FAILURES: Final[int] = 3

#: Ctrl+C 退出码（128+SIGINT 惯例；设计 §3.6）
EXIT_INTERRUPTED: Final[int] = 130

#: 决策 K 线周期恒为 1m（builder 契约），时间戳头的「收盘时刻」= 起点 + 60s
_DECISION_BAR_SECONDS: Final[int] = 60

#: 时间戳头时区（北京时间；数据时间戳本项目统一 +08:00）
_HEADER_TIMEZONE = "Asia/Shanghai"


def market_lines(state_text: str) -> list[str]:
    """从 builder state 文本裁出行情五部分行（逐字节保留原文）。

    按 :data:`_SEGMENT_SEPARATOR` 分段后，保留行首前缀（首个 ``":"`` 之前）
    ∈ :data:`MARKET_LINE_PREFIXES` 的行；``STATE_SCHEMA`` 标记行（无冒号）与
    ``账户`` 行被排除。按行前缀而非位置索引过滤——模板未来加行时裁剪行为可预期
    （设计 §3.4）。
    """
    return [
        segment
        for segment in state_text.split(_SEGMENT_SEPARATOR)
        if segment.split(":", 1)[0] in MARKET_LINE_PREFIXES
    ]


def next_boundary(now: pd.Timestamp, duration_seconds: int) -> pd.Timestamp:
    """``now`` 上取整到周期网格的下一边界（纯函数，设计 §3.3）。

    在绝对 epoch 纳秒上取整，tz 原样保留；北京时间为整时区偏移，绝对 epoch 网格
    与交易所分钟刻自然对齐（含夜盘 23:05 等跨零点时刻，无日期边界假设）。

    边界语义 = 上一根周期 bar 的收盘时刻 = 下一根的起点；恰在边界醒来时上一根
    已收盘、会被 ``fetch_recent``（as_of=now）纳入。
    """
    period_ns = duration_seconds * 1_000_000_000
    return pd.Timestamp(((now.value // period_ns) + 1) * period_ns, tz=now.tz)


def timestamp_header(symbol: str, period: str, decision_bar_timestamp: pd.Timestamp) -> str:
    """watch 输出前置时间戳头 ``[YYYY-MM-DD HH:MM symbol period]``。

    时间 = 决策 K 线（最新已收盘 1m 根）的**收盘时刻**（起点 + 60s，北京时间）：
    确定性（不随输出墙钟漂移，数据延迟时仍指向内容所属分钟）；``period`` 字段 =
    watch 触发周期字符串（与内容口径的区别由 README 说明）。
    """
    ts = pd.Timestamp(decision_bar_timestamp)
    if ts.tz is None:
        ts = ts.tz_localize(_HEADER_TIMEZONE)
    else:
        ts = ts.tz_convert(_HEADER_TIMEZONE)
    close_at = ts + pd.Timedelta(seconds=_DECISION_BAR_SECONDS)
    return f"[{close_at.strftime('%Y-%m-%d %H:%M')} {symbol} {period}]"


def run_watch(
    *,
    symbol: str,
    period: str,
    bars: int,
    daily_bars: int,
    params: EpisodeParams,
    turning_points: tuple[TurningPoint, ...],
    price_precision: int,
    full: bool,
    fetch_1m: Callable[[], pd.DataFrame],
    fetch_rest: Callable[[], tuple[pd.DataFrame, dict[str, Any | None]]],
    render: Callable[[LiveStateSnapshot, pd.Timestamp], None],
    warn: Callable[[str], None],
    now_fn: Callable[[], pd.Timestamp],
    wait_fn: Callable[[float], None],
) -> int:
    """watch 循环驱动（时钟/等待/取数/渲染注入，无 I/O；设计 §3.3）。

    流程：首轮立即输出（启动即见当前快照，同 ``--once`` 内容）→ 逐周期在
    ``next_boundary`` 边界等待（``wait_fn`` 接收绝对 unix 秒）→ ``fetch_1m``
    判重门（同根静默、新根才 ``fetch_rest`` 取 1d/联动）→ build → ``render``。

    :param fetch_1m: 取主品种 1m 帧（只含已收盘根）；调用方负责 shortfall 告警与校验
    :param fetch_rest: 取 1d 帧与联动品种帧（仅在有新根时被调用）
    :param render: 输出快照（第二参数 = 判重键，供时间戳头格式化）
    :param warn: stderr 告警出口（单周期失败计数与循环层告警）
    :param now_fn: 当前时间（测试注入假时钟）
    :param wait_fn: 睡至 deadline（绝对 unix 秒；生产 = ``provider.wait_until``）
    :returns: 退出码——循环只能经 KeyboardInterrupt（130）或连续失败（1）离开
    """
    duration_seconds = resolve_duration_seconds(period)
    last_key: pd.Timestamp | None = None
    fail_streak = 0

    def cycle() -> None:
        nonlocal last_key, fail_streak
        frame_1m = fetch_1m()
        if frame_1m.empty:
            raise DatasetError("watch 周期取到空 1m 帧（无已收盘数据）")
        key = frame_1m["timestamp"].iloc[-1]
        if last_key is not None and key == last_key:
            return  # 同根 → 无任何 stdout（静默）
        frame_1d, linkage = fetch_rest()
        snapshot = build_live_state(
            symbol=symbol,
            bars_1m=frame_1m,
            decision_index=len(frame_1m) - 1,
            daily_rows=frame_1d,
            turning_points=turning_points,
            linkage_bars=linkage or None,
            params=params,
            price_precision=price_precision,
        )
        render(snapshot, key)
        last_key = key
        fail_streak = 0

    pending_first = True
    while True:
        try:
            if pending_first:
                pending_first = False  # 首轮立即输出，不等边界
            else:
                deadline = next_boundary(now_fn(), duration_seconds)
                wait_fn(deadline.timestamp())
            cycle()
        except DatasetError as exc:
            fail_streak += 1
            warn(f"警告：state-now watch 单周期失败（连续第 {fail_streak} 次）：{exc}")
            if fail_streak >= _WATCH_MAX_CONSECUTIVE_FAILURES:
                return 1  # 连续失败 → 会话不可恢复，终止（设计 §3.6）
        except KeyboardInterrupt:
            return EXIT_INTERRUPTED  # 连接释放由调用方 finally 负责
