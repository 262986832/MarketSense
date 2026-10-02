"""审计与自检：确定性（双跑 sha256）、跨 split 隔离、状态泄漏抽查、计数汇总。

本模块只做**只读检查**与审计文件内容构造，不修改账户、不生成标签。

* :func:`check_split_isolation` 镜像 NanoJev
  ``train_pipeline_decisions.read_training_records`` 的 state/source_group 跨 split
  检查口径（同一 ``state_id``／``metadata.source_group_id`` 只能属于一个 split）；
* :func:`check_state_leakage` **独立重算**决策 K 线的现价/联动/日线/日内四行（v5 日线行含
  日线折点趋势极值），并与记录内的状态文本比对（不调用状态序列化实现，避免自证），
  同时拒绝状态里出现绝对价格/量（含上一交易日日线绝对值与可用折点绝对极值价）；
* :func:`verify_determinism` 同输入双跑并逐文件比对 sha256；
* :func:`build_audit_payload` 汇总死亡/剔除/每 split 计数、输入指纹与生成参数。
"""

from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from dataset.board_state import DAY_END, NIGHT_START
from dataset.errors import DatasetError
from dataset.storage import file_sha256

from dataset.market_episode.labels import SegmentOutcome
from dataset.market_episode.replay import Bar, format_ratio_value, ratio_or_none
from dataset.market_episode.segments import SPLIT_ROLES, EpisodeParams, Segment
from dataset.turning_points import TrendExtreme, TurningPoint

#: 审计文件 schema 标记
AUDIT_SCHEMA = "marketsense.episode_audit.v1"
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
#: v5 起新增日线折点趋势极值，AUDIT_SCHEMA 保持 v1：payload 为纯增量变更，
#: v5 演进标记由 state_template 承载）
FROZEN_DECISIONS: Mapping[str, str] = {
    "reference_price": "segment_first_bar_open",
    "state_template": "marketsense.episode_state.v5:decision_bar_only+board_state+daily_trend_extremes",
    "stop_exit_fill": "decision_bar_opposite_extreme_minus_plus_tick",
    "mfe": "max_favorable_before_stop_touch__adverse_side_first_same_bar",
    "accounting": "ratio_units:initial_equity=1.0,1_lot=1_notional,no_multiplier,no_fees",
    "flat_sample_band": "configurable_minutes_around_opportunity_minutes",
    "initial_state": "first_bar_is_first_decision_point,flat,peak=1.0",
    "reversal_condition_2": "stop_reached_first_or_scan_end_without_exceeding_current_bar",
    "board_state_prev_day": "daily_file_1d_prev_trading_day_over_segment_first_open",
    "board_state_today": "segment_bars_cumulative_extrema_through_decision_bar",
    "daily_trend_extremes": "daily_tp_csv_confirmed_date_lt_trade_date__recent_1up_1down_over_segment_first_open",
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
    """独立重算的决策 K 线价格比值行（v4「现价」行；不调用状态序列化实现，避免自证）。"""
    values = (
        _ratio_or_na(bar.open, reference.open, precision),
        _ratio_or_na(bar.high, reference.open, precision),
        _ratio_or_na(bar.low, reference.open, precision),
        _ratio_or_na(bar.close, reference.open, precision),
    )
    return f"现价: o={values[0]} h={values[1]} l={values[2]} c={values[3]}"


def _independent_vol_line(bar: Bar, reference: Bar, precision: int) -> str:
    """独立重算的成交量/持仓量归一化比值行（v4「联动」行）。"""
    values = (
        _ratio_or_na(bar.volume, reference.volume, precision),
        _ratio_or_na(bar.open_oi, reference.open_oi, precision),
        _ratio_or_na(bar.close_oi, reference.close_oi, precision),
    )
    return f"联动: v={values[0]} oi_open={values[1]} oi_close={values[2]}"


def _independent_daily_line(
    prev_daily: tuple[float, float, float],
    reference: Bar,
    precision: int,
    trend_up_extreme: TrendExtreme,
    trend_dn_extreme: TrendExtreme,
) -> str:
    """独立重算日线行（v5：prev = 上一交易日日线值 ÷ 片段首根开盘；
    trend = 可用折点段极值 ÷ 片段首根开盘 + 段长整数直出）。"""
    prev_high, prev_low, prev_close = prev_daily
    return (
        "日线: "
        f"prev_h={format_ratio_value(ratio_or_none(prev_high, reference.open), precision)}"
        f" prev_l={format_ratio_value(ratio_or_none(prev_low, reference.open), precision)}"
        f" prev_c={format_ratio_value(ratio_or_none(prev_close, reference.open), precision)}"
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


def check_state_leakage(
    records: list[dict[str, Any]],
    *,
    bars_by_segment: Mapping[str, tuple[Bar, ...]],
    price_precision: int,
    prev_daily_by_segment: Mapping[str, tuple[float, float, float]],
    trend_points_by_symbol: Mapping[str, tuple[TurningPoint, ...]],
    symbols_by_segment: Mapping[str, str],
) -> None:
    """状态泄漏抽查：决策点状态只能由 ≤ 决策 K 线的数据计算。

    * 记录里的现价/联动/日线/日内四行必须等于按 ≤ 决策 K 线的数据
      **独立重算**的结果 → 状态若误用下一根/其它根会失败；
    * 日线行 trend 四值必须等于按「确认根日期严格早于决策交易日」过滤后的
      最近 1 up + 1 down 折点独立重算（T 日及之后确认的折点若被误用即被检出）；
    * 状态文本中不得出现决策 K 线及其邻根的**绝对**价格/量，也不得出现上一交易日
      日线的绝对价格与该记录可用折点的绝对极值价（它们只能以比值出现）。
    """
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
            ("联动", _independent_vol_line(bar, reference, price_precision)),
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
    """独立重算日内行（today = 前缀累计极值；``bar=`` 取 state_id 解析出的序号，
    与序列化侧的 ``bar.index`` 交叉锁定）。"""
    today_high = max(bar.high for bar in bars_prefix)
    today_low = min(bar.low for bar in bars_prefix)
    return (
        f"日内: bar={bar_index}"
        f" today_h={format_ratio_value(ratio_or_none(today_high, reference.open), precision)}"
        f" today_l={format_ratio_value(ratio_or_none(today_low, reference.open), precision)}"
    )


def check_no_absolute_values(state_text: str, bar: Bar, *, precision: int) -> None:
    """状态文本不得包含该根 K 线的**绝对价格**（绝对数只留在行情层）。

    只检查价格 token（含固定小数位形式与浮点 ``repr`` 形式）：成交量/持仓量为小整数，
    与状态里 ``日内`` 行的 ``bar=<序号>`` 等数字 token 容易假阳性，且序列化器在结构上
    只输出比值行（由字节级夹具测试锁定），故不参与 token 检查。
    """
    tokens = _absolute_price_tokens((bar.open, bar.high, bar.low, bar.close), precision)
    for token in sorted(tokens):
        if re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.])", state_text):
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
) -> dict[str, Any]:
    """构造审计文件内容（不含墙钟时间，保证双跑逐字节一致）。

    v5 起：新增顶层 section ``trend_extremes``（缺折点跳过清单 + 1d 转折点 CSV 版本），
    ``AUDIT_SCHEMA`` 保持 v1（payload 为纯增量变更；v5 演进标记由
    ``frozen_decisions.state_template`` 承载）。
    """
    per_segment: list[dict[str, Any]] = []
    for segment in segments:
        outcome = outcomes.get(segment.segment_id)
        if outcome is None:
            # board_state / 日线折点跳过的片段不产出 per_segment 审计（分别在
            # board_state.skipped_segments 与 trend_extremes.skipped_segments 记录）
            continue
        per_segment.append(
            {
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
        )

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
    "FROZEN_DECISIONS",
    "GOLD_LABEL_KINDS",
    "RECORD_REQUIRED_FIELDS",
    "build_audit_payload",
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
