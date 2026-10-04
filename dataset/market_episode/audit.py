"""审计与自检：确定性（双跑 sha256）、跨 split 隔离、状态泄漏抽查、跨片段账户净值链
硬校验、计数汇总。

本模块只做**只读检查**与审计文件内容构造，不修改账户、不生成标签。

* :func:`check_split_isolation` 镜像 NanoJev
  ``train_pipeline_decisions.read_training_records`` 的 state/source_group 跨 split
  检查口径（同一 ``state_id``／``metadata.source_group_id`` 只能属于一个 split）；
* :func:`check_state_leakage` **独立重算**决策 K 线的现价/联动/日线/日内四行（v5 日线行含
  日线折点趋势极值），并与记录内的状态文本比对（不调用状态序列化实现，避免自证），
  同时拒绝状态里出现绝对价格/量（含上一交易日日线绝对值与可用折点绝对极值价）；
  v8 起传入 ``account_inputs_by_segment``（T6b 接线）时对每决策点的账户行三值
  （净值/今日/回撤）经 :func:`_independent_account_values` 独立复算并逐值比对
  （防自证/tamper 防护；缺省 ``None`` 保持旧行为）；
* :func:`check_account_chain` 跨片段账户净值链硬校验（首评估片段起点净值 = 100、
  逐对「下一片段起点净值 = 前一片段末结算净值」精确比对，跳过片段链冻结穿过）；
* :func:`_independent_account_values` 由 TradeEvent 轨迹 + K 线 + 初始（净值, 峰值）
  **内联重放** t−1 盯市规则，复算片段内逐处理 bar 的净值/今日/回撤三值（不调用
  账户实现，防自证，供状态账户行独立比对使用；T6b 起由 ``check_state_leakage``
  接线调用）；
* :func:`verify_determinism` 同输入双跑并逐文件比对 sha256；
* :func:`build_audit_payload` 汇总死亡/剔除/每 split 计数、输入指纹与生成参数。
"""

from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from dataset.board_state import DAY_END, NIGHT_START
from dataset.errors import DatasetError
from dataset.storage import file_sha256

from dataset.market_episode.labels import SegmentOutcome
from dataset.market_episode.replay import (
    Bar,
    DIRECTIONS,
    LONG,
    TradeEvent,
    format_ratio_value,
    ratio_or_none,
)
from dataset.market_episode.segments import SPLIT_ROLES, EpisodeParams, Segment
from dataset.turning_points import TrendExtreme, TurningPoint

#: 审计文件 schema 标记
AUDIT_SCHEMA = "marketsense.episode_audit.v1"
#: 净值基数（展示口径：净值 = ``NET_VALUE_BASE × equity``，equity 为比值化记账权益、
#: 初始 1.0 → 首片段起点净值 100.0；与 ``dataset/account.py`` docstring 的约定一致。
#: 账户域本身以 equity 记账、不定义该常量——净值是展示层换算，账户链校验以净值口径进行）
NET_VALUE_BASE: float = 100.0
#: 记录必填字段（对齐 NanoJev ``validate_training_row``）
RECORD_REQUIRED_FIELDS = ("id", "state_id", "family_id", "split", "state", "questions")
#: NanoJev 允许的 gold_label_kind / gold_probs_kind（镜像其常量）
GOLD_LABEL_KINDS = frozenset(
    {
        "observed_outcome",
        "deterministic_truth",
        "reference_argmax_compatibility",
        "unspecified_compatibility_label",
        "hard_gold_unspecified",
        "unobserved",
    }
)

#: 本轮实现的冻结口径（写入审计文件，便于复算时对照；v2 起新增 board_state；
#: v5 起新增日线折点趋势极值；v6 起行重排（联动行移至日内行之后）；v7 起键名中文化 +
#: bar 序号与量/持仓量比值并入现价行（持仓量只保留收盘）+ 联动行 na 占位；
#: v8 起账户行扩展净值/今日（净值 = NET_VALUE_BASE × equity，跨片段链式延续；
#: 今日 = 片段内权益变化，每片段重置），AUDIT_SCHEMA 保持 v1：payload 为纯增量变更
#: （account_carry / today_pnl 为纯增量键），演进标记由 state_template 承载）
FROZEN_DECISIONS: Mapping[str, str] = {
    "reference_price": "segment_first_bar_open",
    "state_template": "marketsense.episode_state.v8:decision_bar_only+board_state+daily_trend_extremes+account_net_value",
    "stop_exit_fill": "decision_bar_opposite_extreme_minus_plus_tick",
    "mfe": "max_favorable_before_stop_touch__adverse_side_first_same_bar",
    "accounting": "net_value_base_100:equity=100*(1+cum_ratio_pnl),1_lot=1_notional,no_multiplier,no_fees",
    "flat_sample_band": "configurable_minutes_around_opportunity_minutes",
    "initial_state": "first_bar_is_first_decision_point,flat,peak=initial_net_value",
    "reversal_condition_2": "stop_reached_first_or_scan_end_without_exceeding_current_bar",
    "board_state_prev_day": "daily_file_1d_prev_trading_day_over_segment_first_open",
    "board_state_today": "segment_bars_cumulative_extrema_through_decision_bar",
    "daily_trend_extremes": "daily_tp_csv_confirmed_date_lt_trade_date__recent_1up_1down_over_segment_first_open",
    "account_carry": "net_value_and_peak_carry_across_segments_time_ordered_serial_replay",
    "today_pnl": "segment_equity_change_resets_per_segment",
    "question_template": "marketsense.episode_question.v1:concise_action_labels",
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


def check_record_shape(record: Any, *, where: str) -> None:
    """单条记录结构校验（镜像 NanoJev 契约的必要子集）。"""
    _require(isinstance(record, dict), f"{where} 记录必须是 JSON 对象")
    for key in RECORD_REQUIRED_FIELDS:
        _require(key in record, f"{where} 缺少必需字段: {key}")
    _require(record["split"] in SPLIT_ROLES, f"{where} split 非法: {record['split']!r}")
    for key in ("id", "state_id", "family_id"):
        value = record[key]
        _require(
            isinstance(value, str) and bool(value.strip()),
            f"{where} {key} 必须为非空字符串",
        )
    state = record["state"]
    _require(isinstance(state, str) and bool(state), f"{where} state 必须为非空字符串")
    questions = record["questions"]
    _require(isinstance(questions, dict) and questions, f"{where} questions 必须为非空对象")

    known_qids = set(questions)
    for key in ("gold", "gold_probs", "gold_label_kind", "gold_probs_kind"):
        value = record.get(key)
        _require(
            value is None or (isinstance(value, dict) and not (set(value) - known_qids)),
            f"{where} {key} 含未知 question ID",
        )
    for qid, question in questions.items():
        _require(isinstance(qid, str) and bool(qid.strip()), f"{where} question ID 非法")
        _require(isinstance(question, dict), f"{where}:{qid} 问题必须是对象")
        _require(
            not (set(question) - {"type", "instructions", "criteria"}),
            f"{where}:{qid} 含不支持的 question 字段",
        )
        question_type = question.get("type")
        instructions = question.get("instructions")
        _require(
            question_type in {"boolean", "choice", "score"}
            and isinstance(instructions, str)
            and bool(instructions.strip()),
            f"{where}:{qid} 题型或 instructions 无效",
        )
        criteria = question.get("criteria")
        if question_type == "choice":
            _require(
                isinstance(criteria, dict) and 2 <= len(criteria) <= 255,
                f"{where}:{qid} choice criteria 必须是含 2–255 项的对象",
            )
            _require(
                all(
                    isinstance(key, str)
                    and key.strip()
                    and isinstance(value, str)
                    and value.strip()
                    for key, value in criteria.items()
                ),
                f"{where}:{qid} choice 候选 ID 与描述必须为非空字符串",
            )
        elif question_type == "boolean":
            _require(
                criteria is None
                or (
                    isinstance(criteria, dict)
                    and not (set(criteria) - {"false", "true"})
                ),
                f"{where}:{qid} boolean criteria 非法",
            )
        gold = record.get("gold", {}).get(qid)
        if gold is not None:
            if question_type == "choice":
                _require(
                    isinstance(gold, str) and gold in (criteria or {}),
                    f"{where}:{qid} choice gold 必须是已提供的候选 ID",
                )
            elif question_type == "boolean":
                _require(type(gold) is bool, f"{where}:{qid} boolean gold 必须是布尔值")
            else:
                _require(
                    type(gold) is int and 0 <= gold < len(criteria or []),
                    f"{where}:{qid} score gold 必须是合法层级",
                )
    label_kind = record.get("gold_label_kind")
    for qid, kind in (label_kind or {}).items():
        _require(kind in GOLD_LABEL_KINDS, f"{where}:{qid} gold_label_kind 非法: {kind!r}")


def check_records(records: list[dict[str, Any]]) -> None:
    """全量记录检查：结构 + id 唯一 + state/source_group 跨 split 隔离。"""
    for index, record in enumerate(records):
        check_record_shape(record, where=f"记录 {index}")
    check_record_ids_unique(records)
    check_split_isolation(records)


def check_record_ids_unique(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        record_id = record.get("id")
        _require(record_id not in seen, f"记录 ID 重复: {record_id!r}")
        seen.add(record_id)


def check_split_isolation(records: list[dict[str, Any]]) -> None:
    """镜像 NanoJev：同一 ``state_id``／``source_group_id`` 不得跨 split。"""
    state_splits: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    for record in records:
        split = record.get("split")
        registry_pairs = (
            (record.get("state_id"), state_splits),
            (record.get("metadata", {}).get("source_group_id")
             if isinstance(record.get("metadata"), dict)
             else None, source_splits),
        )
        for key, registry in registry_pairs:
            if key is None:
                continue
            previous = registry.get(key)
            if previous is not None and previous != split:
                raise DatasetError(
                    f"同一 state/source group 跨 split: {key!r}（{previous} vs {split}）"
                )
            registry[key] = split


def parse_state_id(state_id: str) -> tuple[str, int]:
    """``{segment_id}:{决策 K 线序号}`` → ``(segment_id, bar_index)``。"""
    _require(isinstance(state_id, str) and ":" in state_id, f"state_id 格式非法: {state_id!r}")
    segment_id, _, raw_index = state_id.rpartition(":")
    _require(bool(segment_id), f"state_id 缺少 segment id: {state_id!r}")
    try:
        bar_index = int(raw_index)
    except ValueError:
        raise DatasetError(f"state_id 的决策 K 线序号非法: {state_id!r}") from None
    _require(bar_index >= 0, f"state_id 的决策 K 线序号必须 ≥ 0: {state_id!r}")
    return segment_id, bar_index


def _format_ratio(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def _ratio_or_na(numerator: float, denominator: float, precision: int) -> str:
    if denominator <= 0:
        return "na"
    return _format_ratio(numerator / denominator, precision)


def _independent_px_line(bar: Bar, reference: Bar, precision: int) -> str:
    """独立重算的决策 K 线价格/量/持仓量比值行（v7「现价」行；不调用状态序列化实现，
    避免自证）。v7 起 ``bar`` 序号与量/持仓量比值并入本行（持仓量只保留收盘时刻），
    bar 序号与 state_id 解析出的决策 K 线交叉锁定（bar 即 ``bars[bar_index]``）。"""
    values = (
        _ratio_or_na(bar.open, reference.open, precision),
        _ratio_or_na(bar.high, reference.open, precision),
        _ratio_or_na(bar.low, reference.open, precision),
        _ratio_or_na(bar.close, reference.open, precision),
        f"{bar.index}",
        _ratio_or_na(bar.volume, reference.volume, precision),
        _ratio_or_na(bar.close_oi, reference.close_oi, precision),
    )
    return (
        f"现价: 开={values[0]} 高={values[1]} 低={values[2]} 收={values[3]}"
        f" bar={values[4]} 成交量比={values[5]} 持仓量比={values[6]}"
    )


def _independent_daily_line(
    prev_daily: tuple[float, float, float],
    reference: Bar,
    precision: int,
    trend_up_extreme: TrendExtreme,
    trend_dn_extreme: TrendExtreme,
) -> str:
    """独立重算日线行（v5/v7：prev = 上一交易日日线值 ÷ 片段首根开盘，v7 起中文名
    昨日高/低/收；trend = 可用折点段极值 ÷ 片段首根开盘 + 段长整数直出）。"""
    prev_high, prev_low, prev_close = prev_daily
    return (
        "日线: "
        f"昨日高={format_ratio_value(ratio_or_none(prev_high, reference.open), precision)}"
        f" 昨日低={format_ratio_value(ratio_or_none(prev_low, reference.open), precision)}"
        f" 昨日收={format_ratio_value(ratio_or_none(prev_close, reference.open), precision)}"
        f" trend_up={format_ratio_value(ratio_or_none(trend_up_extreme.trend_extreme_price, reference.open), precision)}"
        f" trend_up_len={trend_up_extreme.segment_length}"
        f" trend_dn={format_ratio_value(ratio_or_none(trend_dn_extreme.trend_extreme_price, reference.open), precision)}"
        f" trend_dn_len={trend_dn_extreme.segment_length}"
    )


def _independent_bar_trade_date(bars: tuple[Bar, ...], bar: Bar) -> dt.date:
    """独立实现的决策交易日归属（与序列化侧同口径但**不调用其 helper**，防自证）。

    夜盘归属规则与 ``board_state.attribute_windows`` 同口径：日盘 bar（tod < 15:00）
    → 日历日；夜盘 bar（tod ≥ 21:00）→ 片段日盘日历日集合（升序）中其后的下一个
    交易日；``[15:00, 21:00)`` 或夜盘无下一交易日 → ``DatasetError``。
    """
    stamp = pd.Timestamp(bar.timestamp)
    time_of_day = stamp.time()
    if time_of_day < DAY_END:
        return stamp.date()
    trading_days = sorted(
        {
            pd.Timestamp(item.timestamp).date()
            for item in bars
            if pd.Timestamp(item.timestamp).time() < DAY_END
        }
    )
    if time_of_day < NIGHT_START:
        raise DatasetError(
            f"审计：决策 K 线 time-of-day 在 15:00–20:59，无法归属交易日: {bar.timestamp!r}"
        )
    position = bisect.bisect_right(trading_days, stamp.date())
    if position >= len(trading_days):
        raise DatasetError(
            f"审计：夜盘决策 K 线在片段内找不到后续交易日: {bar.timestamp!r}"
        )
    return trading_days[position]


def _independent_trend_extremes(
    points: tuple[TurningPoint, ...], trade_date: dt.date
) -> tuple[TrendExtreme, TrendExtreme] | None:
    """独立实现的可用折点选择（归属/过滤/选择均不调用序列化侧与数据层选择函数，防自证）。

    过滤确认根日期严格早于 ``trade_date`` 的 up/down 点（任一侧缺失 → ``None``，
    调用方报不一致）；选择 = 过滤后时间序**最后一个** up/down；段长由点序列推导
    （up/down 点的段起点 = 序列中上一个 up/down 点的触发根，无前序 → 0；过滤后
    为前缀，与全序列推导一致）。
    """
    usable = tuple(
        point
        for point in points
        if point.kind in ("up", "down") and point.timestamp.date() < trade_date
    )
    if not any(point.kind == "up" for point in usable) or not any(
        point.kind == "down" for point in usable
    ):
        return None
    last_up: TrendExtreme | None = None
    last_down: TrendExtreme | None = None
    previous_trigger: int | None = None
    for point in usable:
        if point.trend_extreme_price is None or point.trend_extreme_bar_index is None:
            raise DatasetError(
                f"审计：可用 up/down 折点缺当次趋势极值字段（kind={point.kind}, "
                f"bar_index={point.bar_index}）"
            )
        segment_start = 0 if previous_trigger is None else previous_trigger
        extreme = TrendExtreme(
            kind=point.kind,
            trend_extreme_price=point.trend_extreme_price,
            trend_extreme_bar_index=point.trend_extreme_bar_index,
            segment_length=point.bar_index - segment_start,
        )
        if point.kind == "up":
            last_up = extreme
        else:
            last_down = extreme
        previous_trigger = point.bar_index
    assert last_up is not None and last_down is not None
    return last_up, last_down


@dataclass(frozen=True)
class AccountReplayInputs:
    """账户行独立复算的逐片段入参（v8/T6b；与 :func:`_independent_account_values`
    参数一一对应，``bars`` 由 ``check_state_leakage`` 的 ``bars_by_segment`` 提供，
    不在此重复）。

    * ``events``：片段 TradeEvent 轨迹（生成侧 ``SegmentOutcome.events``）；
    * ``processed_bar_count``：实际处理 bar 数（死亡即止）；
    * ``initial_net_value`` / ``initial_peak_net_value``：片段账户链起点（净值, 峰值）
      快照（净值尺度，首评估片段 = ``NET_VALUE_BASE``；与 ``generate_dataset`` 的
      净值链 carry 同源同值）。
    """

    events: tuple[TradeEvent, ...]
    processed_bar_count: int
    initial_net_value: float
    initial_peak_net_value: float


def _independent_account_values(
    events: Iterable[TradeEvent],
    bars: Sequence[Bar],
    *,
    processed_bar_count: int,
    initial_net_value: float = NET_VALUE_BASE,
    initial_peak_net_value: float | None = None,
) -> dict[int, tuple[float, float, float]]:
    """独立复算片段内逐处理 bar 的账户三值 ``{bar_index: (净值, 今日, 回撤)}``。

    防自证（沿 ``_independent_trend_extremes`` 先例）：由 TradeEvent 轨迹 + K 线 +
    初始（净值, 峰值净值）**内联重放** t−1 盯市规则，**不调用** ``ReplayAccount``
    与任何序列化实现。净值口径：净值 = ``NET_VALUE_BASE × equity``（equity 为比值
    化记账权益），今日 = 净值 − 片段起点净值（``initial_net_value``，净值尺度），
    回撤 = (峰值 − 权益)/峰值（比值，权益尺度算术与实现一致）。逐项语义：

    * ``bar_index == i`` 的权益 = 初始权益 + 已实现盈亏 + 浮盈盯市；浮盈盯市价 =
      ``close[i−1]``（比值口径，分母 = 片段首根开盘价）；``i == 0`` 尚无持仓，
      权益 = 初始权益 → 今日 = 0；
    * 持仓由事件流重建：开仓事件（``pnl_ratio is None``）设持仓
      ``(direction, price_ratio)``，平仓事件累计已实现并清仓；决策点三值为该根
      成交**前**的盯市值（先记号后成交，与推进顺序一致；反手 = 同根先平后开两笔）；
    * 峰值在盯市时与平仓结算后更新（``peak = max(peak, equity)``，与
      ``ReplayAccount.mark_to_market``/``close_position`` 同语义；结算后更新
      确保片段末强平结算净值计入高水位）。

    :param processed_bar_count: 实际处理 bar 数（死亡即止；与 ``SegmentOutcome``
        的 ``processed_bar_count`` 同口径）；事件 bar 序号须落在处理范围内且升序。
    :return: ``{bar_index: (净值, 今日, 回撤)}``（原始 float，供状态账户行逐值比对；
        T6b 起由 ``check_state_leakage`` 接线调用）
    """
    if not bars:
        raise DatasetError("审计：账户复算需要非空 K 线序列")
    _require(
        0 < processed_bar_count <= len(bars),
        f"审计：账户复算 processed_bar_count 越界: {processed_bar_count}"
        f"（片段共 {len(bars)} 根）",
    )
    _require(
        float(initial_net_value) > 0,
        f"审计：账户复算片段起点净值必须为正数: {initial_net_value!r}",
    )
    reference = float(bars[0].open)
    _require(
        reference > 0,
        f"审计：片段首根开盘价必须为正数（比值表达的分母）: {bars[0].open!r}",
    )
    ordered_events = list(events)
    previous_bar_index: int | None = None
    for event in ordered_events:
        _require(
            0 <= event.bar_index < processed_bar_count,
            f"审计：账户复算成交事件 bar 序号越界: {event.bar_index}"
            f"（处理范围 0..{processed_bar_count - 1}）",
        )
        if previous_bar_index is not None:
            _require(
                event.bar_index >= previous_bar_index,
                f"审计：账户复算成交事件 bar 序号非时间升序: "
                f"{previous_bar_index} → {event.bar_index}",
            )
        previous_bar_index = event.bar_index

    initial_equity = float(initial_net_value) / NET_VALUE_BASE
    peak_equity = (
        float(initial_peak_net_value) / NET_VALUE_BASE
        if initial_peak_net_value is not None
        else initial_equity
    )
    realized = 0.0
    # 持仓 = (direction, entry_ratio)；entry_ratio 直接取开仓事件携带的 price_ratio
    position: tuple[str, float] | None = None
    values: dict[int, tuple[float, float, float]] = {}
    event_cursor = 0
    for index in range(processed_bar_count):
        # 先记号：t−1 盯市（与 ReplayAccount.mark_to_market 同规则，算术内联重写）
        equity = initial_equity + realized
        if position is not None and index >= 1:
            mark_ratio = float(bars[index - 1].close) / reference
            entry_ratio = position[1]
            equity += (
                mark_ratio - entry_ratio
                if position[0] == LONG
                else entry_ratio - mark_ratio
            )
        if equity > peak_equity:
            peak_equity = equity
        drawdown = 0.0 if peak_equity <= 0 else (peak_equity - equity) / peak_equity
        net_value = NET_VALUE_BASE * equity
        values[index] = (net_value, net_value - float(initial_net_value), drawdown)
        # 后成交：该 bar 的成交事件（事件流顺序即执行顺序：先平仓、后开仓）
        while (
            event_cursor < len(ordered_events)
            and ordered_events[event_cursor].bar_index == index
        ):
            event = ordered_events[event_cursor]
            if event.pnl_ratio is None:
                _require(
                    event.direction in DIRECTIONS,
                    f"审计：账户复算在 bar {index} 遇到非法开仓方向: {event.direction!r}",
                )
                _require(
                    position is None,
                    f"审计：账户复算在 bar {index} 遇到重复开仓事件（已有持仓）",
                )
                position = (event.direction, event.price_ratio)
            else:
                _require(
                    position is not None,
                    f"审计：账户复算在 bar {index} 遇到无持仓平仓事件"
                    f"（reason={event.reason!r}）",
                )
                realized += event.pnl_ratio
                position = None
                # 结算后峰值更新（与 ReplayAccount.close_position 同规则，
                # 算术内联重写）：平仓结算权益计入高水位
                settled_equity = initial_equity + realized
                if settled_equity > peak_equity:
                    peak_equity = settled_equity
            event_cursor += 1
    _require(
        event_cursor == len(ordered_events),
        "审计：账户复算存在未消费的成交事件（事件流与处理 bar 范围不一致）",
    )
    return values


def check_state_leakage(
    records: list[dict[str, Any]],
    *,
    bars_by_segment: Mapping[str, tuple[Bar, ...]],
    price_precision: int,
    prev_daily_by_segment: Mapping[str, tuple[float, float, float]],
    trend_points_by_symbol: Mapping[str, tuple[TurningPoint, ...]],
    symbols_by_segment: Mapping[str, str],
    account_inputs_by_segment: Mapping[str, AccountReplayInputs] | None = None,
) -> None:
    """状态泄漏抽查：决策点状态只能由 ≤ 决策 K 线的数据计算。

    * 记录里的现价/日线/日内三行必须等于按 ≤ 决策 K 线的数据**独立重算**的结果 →
      状态若误用下一根/其它根会失败（v7 起 ``bar`` 序号与量/持仓量比值在现价行内，
      同受此检查；联动行为 ``na`` 常量占位，按常量比对）；
    * 日线行 trend 四值必须等于按「确认根日期严格早于决策交易日」过滤后的
      最近 1 up + 1 down 折点独立重算（T 日及之后确认的折点若被误用即被检出）；
    * 账户行三值（净值/今日/回撤）：传入 ``account_inputs_by_segment``（v8/T6b 接线）
      时由 :func:`_independent_account_values` 独立复算并与 state 文本逐值比对，
      任一值被篡改/与重放不一致即 ``DatasetError``（防自证/tamper 防护）；缺省
      ``None`` 保持旧行为（账户行不参与复算比对）；
    * 状态文本中不得出现决策 K 线及其邻根的**绝对**价格/量，也不得出现上一交易日
      日线的绝对价格与该记录可用折点的绝对极值价（它们只能以比值出现）。
    """
    account_values_cache: dict[str, dict[int, tuple[float, float, float]]] = {}
    for record in records:
        segment_id, bar_index = parse_state_id(record["state_id"])
        bars = bars_by_segment.get(segment_id)
        _require(bars is not None, f"审计缺少片段 {segment_id!r} 的 K 线")
        _require(
            0 <= bar_index < len(bars),
            f"记录 {record['id']} 的决策 K 线序号超出片段范围: {bar_index}",
        )
        state_text = record["state"]
        reference = bars[0]
        bar = bars[bar_index]
        prefix = bars[: bar_index + 1]
        prev_daily = prev_daily_by_segment.get(segment_id)
        _require(
            prev_daily is not None,
            f"审计缺少片段 {segment_id!r} 的上一交易日日线（日线行无法独立重算）",
        )
        assert prev_daily is not None
        symbol = symbols_by_segment.get(segment_id)
        _require(symbol is not None, f"审计缺少片段 {segment_id!r} 的 symbol")
        points = trend_points_by_symbol.get(symbol)
        _require(
            points is not None,
            f"审计缺少 symbol {symbol!r} 的日线转折点（日线行 trend 四值无法独立重算）",
        )
        assert points is not None
        trade_date = _independent_bar_trade_date(bars, bar)
        extremes = _independent_trend_extremes(points, trade_date)
        _require(
            extremes is not None,
            f"审计缺少片段 {segment_id!r} 决策交易日 {trade_date} 的可用日线折点"
            "（确认日早于该交易日的 up/down 需各 ≥1）",
        )
        assert extremes is not None
        trend_up_extreme, trend_dn_extreme = extremes
        expected_lines = (
            ("现价", _independent_px_line(bar, reference, price_precision)),
            ("联动", "联动: na"),
            (
                "日线",
                _independent_daily_line(
                    prev_daily,
                    reference,
                    price_precision,
                    trend_up_extreme,
                    trend_dn_extreme,
                ),
            ),
            ("日内", _independent_intraday_line(prefix, bar_index, reference, price_precision)),
        )
        for line_label, expected in expected_lines:
            _require(
                expected in state_text,
                f"记录 {record['id']} 状态中的{line_label}行与决策 K 线不一致"
                f"（期望 {expected!r}）",
            )
        if account_inputs_by_segment is not None:
            # v8（T6b）：账户行三值独立复算比对（防自证/tamper 防护；复算结果按片段
            # 缓存，同片段多条记录共享一次重放）
            inputs = account_inputs_by_segment.get(segment_id)
            _require(
                inputs is not None,
                f"审计缺少片段 {segment_id!r} 的账户行独立复算入参",
            )
            assert inputs is not None
            replayed = account_values_cache.get(segment_id)
            if replayed is None:
                replayed = _independent_account_values(
                    inputs.events,
                    bars,
                    processed_bar_count=inputs.processed_bar_count,
                    initial_net_value=inputs.initial_net_value,
                    initial_peak_net_value=inputs.initial_peak_net_value,
                )
                account_values_cache[segment_id] = replayed
            _require(
                bar_index in replayed,
                f"记录 {record['id']} 的决策 K 线序号 {bar_index} 超出账户复算"
                f"处理范围（0..{inputs.processed_bar_count - 1}）",
            )
            replay_net, replay_today, replay_drawdown = replayed[bar_index]
            expected_account = (
                f"净值={format_ratio_value(replay_net, price_precision)}"
                f" 今日={format_ratio_value(replay_today, price_precision)}"
                f" 回撤={format_ratio_value(replay_drawdown, price_precision)}"
            )
            match = re.search(r"净值=(\S+) 今日=(\S+) 回撤=(\S+)", state_text)
            _require(
                match is not None,
                f"记录 {record['id']} 状态缺少账户行净值/今日/回撤"
                "（无法与独立复算比对）",
            )
            assert match is not None
            actual_account = (
                f"净值={match.group(1)} 今日={match.group(2)} 回撤={match.group(3)}"
            )
            _require(
                actual_account == expected_account,
                f"记录 {record['id']} 状态中的账户行净值/今日/回撤与独立复算不一致"
                f"（期望 {expected_account!r}，实际 {actual_account!r}）",
            )
        neighbours = {0, bar_index - 1, bar_index, bar_index + 1}
        for neighbour in sorted(index for index in neighbours if 0 <= index < len(bars)):
            check_no_absolute_values(state_text, bars[neighbour], precision=price_precision)
        for token in sorted(_absolute_price_tokens(prev_daily, price_precision)):
            if re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.])", state_text):
                raise DatasetError(
                    f"记录 {record['id']} 状态出现上一交易日绝对价格 {token!r}"
                    "（绝对数不得进入模型输入）"
                )
        usable = tuple(
            point
            for point in points
            if point.kind in ("up", "down") and point.timestamp.date() < trade_date
        )
        extreme_prices = [
            point.trend_extreme_price
            for point in usable
            if point.trend_extreme_price is not None
        ]
        for token in sorted(_absolute_price_tokens(extreme_prices, price_precision)):
            if re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.])", state_text):
                raise DatasetError(
                    f"记录 {record['id']} 状态出现可用折点绝对极值价 {token!r}"
                    "（绝对数不得进入模型输入）"
                )


def check_account_chain(entries: Sequence[Any]) -> list[dict[str, Any]]:
    """跨片段账户净值链硬校验（违规即 ``DatasetError``，同审计既有错误风格）。

    * ``entries`` 为**按时间处理序**的逐评估片段链条目，每项两种形式之一：
      ①含 ``segment_id`` / ``initial_net_value``（起点净值）/
      ``final_net_value``（末结算净值）键的映射；②``(SegmentOutcome, initial_net_value)``
      二元组——末结算净值取 ``NET_VALUE_BASE × outcome.final_equity``
      （净值 = 100×equity；outcome 同时携带 ``final_peak`` 供峰值链传递，但本校验
      只比对净值口径）；
    * 首个评估片段起点净值必须精确等于 ``NET_VALUE_BASE``（= 100.0，即
      ``NET_VALUE_BASE × 1.0``）；逐对核验「下一片段起点净值 == 上一片段末结算净值」
      （精确 ``==``：链内为同一 float 传递，确定性成立）；
    * 跳过片段（缺上一交易日日线/缺可用折点）不产出链条目——链在其前一片段末值上
      **冻结穿过**，逐对 ``==`` 核验天然接受；空序列（全跳过场景）返回空列表。
    * 通过后返回归一化链条目
      ``[{"segment_id", "initial_net_value", "final_net_value"}, ...]``
      （净值尺度，供 ``build_audit_payload`` 写入 per_segment ``account_chain`` 节，
      跨片段可追溯）。
    """
    normalized: list[tuple[str, float, float]] = []
    seen: set[str] = set()
    for position, item in enumerate(entries):
        if isinstance(item, dict):
            for key in ("segment_id", "initial_net_value", "final_net_value"):
                _require(key in item, f"审计账户链：第 {position} 项缺少键 {key!r}")
            segment_id = item["segment_id"]
            _require(
                isinstance(segment_id, str) and bool(segment_id.strip()),
                f"审计账户链：第 {position} 项 segment_id 必须为非空字符串: {segment_id!r}",
            )
            initial = float(item["initial_net_value"])
            final = float(item["final_net_value"])
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], SegmentOutcome)
        ):
            outcome, initial_net_value = item
            segment_id = outcome.segment_id
            initial = float(initial_net_value)
            final = NET_VALUE_BASE * float(outcome.final_equity)
        else:
            raise DatasetError(
                f"审计账户链：第 {position} 项形式非法（应为含起末净值的映射或 "
                f"(SegmentOutcome, 起点净值) 二元组）: {type(item).__name__}"
            )
        _require(
            segment_id not in seen,
            f"审计账户链：片段 {segment_id!r} 重复出现",
        )
        seen.add(segment_id)
        _require(
            initial > 0,
            f"审计账户链：片段 {segment_id} 起点净值必须为正数: {initial!r}",
        )
        normalized.append((segment_id, initial, final))

    chain: list[dict[str, Any]] = []
    previous_final: float | None = None
    for segment_id, initial, final in normalized:
        if previous_final is None:
            _require(
                initial == NET_VALUE_BASE,
                f"审计账户链：首片段 {segment_id} 起点净值 {initial!r} ≠ "
                f"NET_VALUE_BASE（{NET_VALUE_BASE}，首片段必须以 100.0 起算）",
            )
        else:
            _require(
                initial == previous_final,
                f"审计账户链：片段 {segment_id} 起点净值 {initial!r} ≠ "
                f"前一片段末结算净值 {previous_final!r}",
            )
        chain.append(
            {
                "segment_id": segment_id,
                "initial_net_value": round(initial, 12),
                "final_net_value": round(final, 12),
            }
        )
        previous_final = final
    return chain


def _absolute_price_tokens(values, precision: int) -> set[str]:
    """绝对价格 token 集合（固定小数位形式与浮点 ``repr`` 形式）。"""
    tokens: set[str] = set()
    for price in values:
        tokens.add(f"{price:.{precision}f}")
        tokens.add(repr(float(price)))
    return tokens


def _independent_intraday_line(
    bars_prefix,
    bar_index: int,
    reference: Bar,
    precision: int,
) -> str:
    """独立重算日内行（v7：today = 前缀累计极值，中文名今高/今低；``bar=`` 序号自 v7 起
    移入现价行，由 ``_independent_px_line`` 的 ``bars[bar_index].index`` 与 state_id 交叉锁定）。"""
    today_high = max(bar.high for bar in bars_prefix)
    today_low = min(bar.low for bar in bars_prefix)
    return (
        f"日内: 今高={format_ratio_value(ratio_or_none(today_high, reference.open), precision)}"
        f" 今低={format_ratio_value(ratio_or_none(today_low, reference.open), precision)}"
    )


def check_no_absolute_values(state_text: str, bar: Bar, *, precision: int) -> None:
    """状态文本不得包含该根 K 线的**绝对价格**（绝对数只留在行情层）。

    只检查价格 token（含固定小数位形式与浮点 ``repr`` 形式）：成交量/持仓量为小整数，
    与状态里 ``日内`` 行的 ``bar=<序号>`` 等数字 token 容易假阳性，且序列化器在结构上
    只输出比值行（由字节级夹具测试锁定），故不参与 token 检查。

    「账户」行（``账户: `` 开头，净值/今日/回撤）**豁免** token 扫描：六键均为
    净值尺度账户值（净值 = 100×equity，与绝对价不同域，语义上不构成泄漏）；
    其余行（账户外的日线/日内/联动/现价/盘口）邻根 token 扫描语义不变。
    """
    tokens = _absolute_price_tokens((bar.open, bar.high, bar.low, bar.close), precision)
    # 豁免「账户」行后逐行拼接再扫（行分隔符不属价格 token 字符，边界语义与逐行扫一致；
    # v3+ 行连接符为 " \n "，账户行带前导空格，故前导空格不敏感）
    scanned_text = "\n".join(
        line for line in state_text.splitlines() if not line.lstrip().startswith("账户: ")
    )
    for token in sorted(tokens):
        if re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.])", scanned_text):
            raise DatasetError(f"状态文本出现绝对价格 {token!r}（绝对数不得进入模型输入）")


def build_audit_payload(
    *,
    run_id: str,
    params: EpisodeParams,
    segments: tuple[Segment, ...],
    symbols: Mapping[str, float],
    outcomes: Mapping[str, SegmentOutcome],
    records_by_split: Mapping[str, list[dict[str, Any]]],
    split_digests: Mapping[str, str],
    input_hashes: Mapping[str, str],
    board_state_skipped: Mapping[str, str] | None = None,
    trend_extreme_skipped: Mapping[str, str] | None = None,
    trend_source_versions: Mapping[str, str] | None = None,
    daily_source_versions: Mapping[str, str | None] | None = None,
    account_chain: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造审计文件内容（不含墙钟时间，保证双跑逐字节一致）。

    v5 起：新增顶层 section ``trend_extremes``（缺折点跳过清单 + 1d 转折点 CSV 版本），
    ``AUDIT_SCHEMA`` 保持 v1（payload 为纯增量变更；v5 演进标记由
    ``frozen_decisions.state_template`` 承载）。
    v8 起：新增可选 ``account_chain``（:func:`check_account_chain` 的归一化返回）——
    校验通过的每片段起点/末结算净值写入 per_segment 条目的 ``account_chain`` 节
    （净值 = ``NET_VALUE_BASE × equity``，纯增量键；未传入时不写该键，行为与 v7 一致）。
    """
    chain_by_segment: dict[str, tuple[float, float]] = {}
    for position, chain_entry in enumerate(account_chain or ()):  # type: ignore[arg-type]
        _require(
            isinstance(chain_entry, dict),
            f"审计 account_chain 第 {position} 条目必须是映射",
        )
        for key in ("segment_id", "initial_net_value", "final_net_value"):
            _require(
                key in chain_entry,
                f"审计 account_chain 第 {position} 条目缺少键 {key!r}",
            )
        chain_by_segment[str(chain_entry["segment_id"])] = (
            float(chain_entry["initial_net_value"]),
            float(chain_entry["final_net_value"]),
        )

    per_segment: list[dict[str, Any]] = []
    for segment in segments:
        outcome = outcomes.get(segment.segment_id)
        if outcome is None:
            # board_state / 日线折点跳过的片段不产出 per_segment 审计（分别在
            # board_state.skipped_segments 与 trend_extremes.skipped_segments 记录）
            continue
        entry = {
            "segment_id": segment.segment_id,
            "symbol": segment.symbol,
            "period": segment.period,
            "split_role": segment.split_role,
            "start_ts": segment.start.isoformat(),
            "end_ts": segment.end.isoformat(),
            "tick_size": float(symbols[segment.symbol]),
            "source_data_version": outcome.source_data_version,
            "bars": outcome.bar_count,
            "processed_bars": outcome.processed_bar_count,
            "decision_points": len(outcome.decision_points),
            "selected": len(outcome.selected),
            "excluded_flat": outcome.excluded_flat,
            "stop_exit_bars": list(outcome.stop_exits),
            "deaths": [event.bar_index for event in outcome.deaths],
            "death_forced_close": [event.forced_close for event in outcome.deaths],
            "segment_end_forced_close": outcome.segment_end_forced_close,
            "action_counts": dict(sorted(outcome.action_counts.items())),
            "realized_pnl_ratio": round(outcome.realized_pnl_ratio, 12),
            "peak_equity": round(outcome.peak_equity, 12),
        }
        chain = chain_by_segment.get(segment.segment_id)
        if chain is not None:
            # v8：跨片段净值链（check_account_chain 校验通过后的起末净值，纯增量键）
            entry["account_chain"] = {
                "initial_net_value": round(chain[0], 12),
                "final_net_value": round(chain[1], 12),
            }
        per_segment.append(entry)

    per_split = {
        split: {
            "records": len(records_by_split.get(split, [])),
            "questions": sum(
                len(record["questions"]) for record in records_by_split.get(split, [])
            ),
        }
        for split in SPLIT_ROLES
    }
    return {
        "schema": AUDIT_SCHEMA,
        "run_id": run_id,
        "input": dict(sorted(input_hashes.items())),
        "frozen_decisions": dict(sorted(FROZEN_DECISIONS.items())),
        "params": params.as_dict(),
        "totals": {
            "segments": len(segments),
            "bars": sum(outcome.bar_count for outcome in outcomes.values()),
            "processed_bars": sum(
                outcome.processed_bar_count for outcome in outcomes.values()
            ),
            "decision_points": sum(
                len(outcome.decision_points) for outcome in outcomes.values()
            ),
            "selected": sum(len(outcome.selected) for outcome in outcomes.values()),
            "excluded_flat": sum(
                outcome.excluded_flat for outcome in outcomes.values()
            ),
            "stop_exits": sum(len(outcome.stop_exits) for outcome in outcomes.values()),
            "deaths": sum(len(outcome.deaths) for outcome in outcomes.values()),
            "segment_end_forced_closes": sum(
                1
                for outcome in outcomes.values()
                if outcome.segment_end_forced_close is not None
            ),
        },
        "per_split": per_split,
        "per_segment": per_segment,
        "outputs": {"split_files": dict(sorted(split_digests.items()))},
        "board_state": {
            "skipped_segments": [
                {"segment_id": key, "reason": value}
                for key, value in sorted((board_state_skipped or {}).items())
            ],
            "daily_source_versions": dict(sorted((daily_source_versions or {}).items())),
        },
        "trend_extremes": {
            "skipped_segments": [
                {"segment_id": key, "reason": value}
                for key, value in sorted((trend_extreme_skipped or {}).items())
            ],
            "source_versions": dict(sorted((trend_source_versions or {}).items())),
        },
    }


def check_audit_consistency(
    payload: Mapping[str, Any], records_by_split: Mapping[str, list[dict[str, Any]]]
) -> None:
    """审计计数与记录集的一致性校验（写出前自检，防止"计数与数据不一致"）。"""
    per_split = payload["per_split"]
    total_records = 0
    for split in SPLIT_ROLES:
        records = records_by_split.get(split, [])
        declared = per_split[split]["records"]
        _require(
            declared == len(records),
            f"审计 per_split[{split}].records={declared} 与记录数 {len(records)} 不一致",
        )
        declared_questions = per_split[split]["questions"]
        actual_questions = sum(len(record["questions"]) for record in records)
        _require(
            declared_questions == actual_questions,
            f"审计 per_split[{split}].questions={declared_questions} "
            f"与记录问题数 {actual_questions} 不一致",
        )
        total_records += len(records)
    selected = sum(
        entry["selected"] for entry in payload["per_segment"]
    )
    decision_points = sum(entry["decision_points"] for entry in payload["per_segment"])
    excluded = sum(entry["excluded_flat"] for entry in payload["per_segment"])
    _require(
        selected == total_records,
        f"审计 selected={selected} 与总记录数 {total_records} 不一致",
    )
    _require(
        decision_points == selected + excluded,
        f"审计 decision_points={decision_points} 与 selected+excluded="
        f"{selected + excluded} 不一致",
    )
    _require(
        payload["totals"]["selected"] == total_records,
        "审计 totals.selected 与记录数不一致",
    )


def verify_determinism(
    segments: tuple[Segment, ...],
    *,
    symbols: Mapping[str, float],
    data_dir: str | Path,
    params: EpisodeParams,
    work_root: str | Path,
) -> dict[str, str]:
    """同输入双跑生成全部输出文件并比对 sha256（确定性自检）。

    :return: ``{文件名: sha256}``（两跑一致时的结果）
    :raises DatasetError: 文件集合或任一文件摘要不一致
    """
    # 延迟导入：nanojev_records 在模块层依赖本模块的审计构造函数
    from dataset.market_episode.nanojev_records import generate_dataset

    root = Path(work_root)
    first = generate_dataset(
        segments, symbols=symbols, data_dir=data_dir, params=params, output_dir=root / "a"
    )
    second = generate_dataset(
        segments, symbols=symbols, data_dir=data_dir, params=params, output_dir=root / "b"
    )
    first_files = sorted(path.name for path in first.run_dir.iterdir() if path.is_file())
    second_files = sorted(path.name for path in second.run_dir.iterdir() if path.is_file())
    _require(first_files == second_files, "双跑输出文件集合不一致")
    _require(first.run_id == second.run_id, "双跑 run_id 不一致")
    digests: dict[str, str] = {}
    for name in first_files:
        left = file_sha256(first.run_dir / name)
        right = file_sha256(second.run_dir / name)
        _require(left == right, f"双跑输出不一致: {name}")
        digests[name] = left
    return digests


__all__ = [
    "AUDIT_SCHEMA",
    "AccountReplayInputs",
    "FROZEN_DECISIONS",
    "GOLD_LABEL_KINDS",
    "NET_VALUE_BASE",
    "RECORD_REQUIRED_FIELDS",
    "build_audit_payload",
    "check_account_chain",
    "check_audit_consistency",
    "check_no_absolute_values",
    "check_record_ids_unique",
    "check_record_shape",
    "check_records",
    "check_split_isolation",
    "check_state_leakage",
    "parse_state_id",
    "verify_determinism",
]
