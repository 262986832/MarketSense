"""``dataset.market_episode``：行情 episode 训练数据生成流水线（mechanics + policy 两层）。

分层（DESIGN 冻结取舍）：

* :mod:`dataset.account` —— 账户域：确定性回放账户执行语义（决策 K 线成交模型、
  止损锚定、比值化记账、t-1 盯市、死亡/收盘强平契约）；仅依赖 ``dataset.errors``；
* :mod:`dataset.market_episode.replay` —— 数据装载 + 状态行构造
  + 账户域 re-export 兼容 shim（迁移自原 mechanics 层账户引擎）；
* :mod:`dataset.market_episode.labels` —— policy：盈亏比规则真值标签 + (b) 采样；
* :mod:`dataset.market_episode.segments` —— 片段清单 / 品种 tick 配置 / episode 参数；
* :mod:`dataset.market_episode.nanojev_records` —— NanoJev 记录映射、状态序列化、落盘；
* :mod:`dataset.market_episode.audit` —— 确定性双跑、跨 split 隔离、泄漏抽查、计数汇总。

入口：``python -m dataset episode-generate --segments <manifest.jsonl> [--output-dir DIR]
[--config FILE]``（详见 ``dataset/README.md``）。
"""

from __future__ import annotations

from dataset.market_episode.audit import (
    AUDIT_SCHEMA,
    FROZEN_DECISIONS,
    build_audit_payload,
    check_audit_consistency,
    check_records,
    check_split_isolation,
    check_state_leakage,
    parse_state_id,
    verify_determinism,
)
from dataset.market_episode.labels import (
    ACTION_CLOSE,
    ACTION_HOLD,
    ACTION_OPEN_LONG,
    ACTION_OPEN_SHORT,
    ACTION_REVERSE,
    ACTION_STAY_FLAT,
    DecisionPoint,
    DeathEvent,
    SegmentOutcome,
    apply_flat_band,
    evaluate_segment,
    holding_action,
    reversal_conditions,
    reward_risk,
    select_open_action,
)
from dataset.market_episode.nanojev_records import (
    AUDIT_FILENAME,
    FLAT_CRITERIA,
    GOLD_LABEL_KIND,
    HELD_CRITERIA,
    QUESTION_ID,
    STATE_SCHEMA,
    BoardStateValues,
    GenerationResult,
    build_record,
    generate_dataset,
    render_state,
)
from dataset.market_episode.replay import (
    LONG,
    SHORT,
    Bar,
    MarkReport,
    Position,
    PositionSnapshot,
    ReplayAccount,
    TradeEvent,
    bars_from_frame,
    load_segment_bars,
)
from dataset.market_episode.segments import (
    DEFAULT_EPISODE_OUTPUT_DIR,
    EPISODE_PERIOD,
    REQUIRED_SPLIT_ROLES,
    SEGMENT_SCHEMA,
    SPLIT_ROLES,
    SYMBOLS_SCHEMA,
    EpisodeConfig,
    EpisodeParams,
    Segment,
    load_episode_config,
    load_segments,
    load_symbols_config,
    validate_segments,
)

__all__ = [
    # segments（清单/配置/校验）
    "SEGMENT_SCHEMA",
    "SYMBOLS_SCHEMA",
    "EPISODE_PERIOD",
    "SPLIT_ROLES",
    "REQUIRED_SPLIT_ROLES",
    "DEFAULT_EPISODE_OUTPUT_DIR",
    "EpisodeConfig",
    "EpisodeParams",
    "Segment",
    "load_episode_config",
    "load_segments",
    "load_symbols_config",
    "validate_segments",
    # replay（mechanics）
    "LONG",
    "SHORT",
    "Bar",
    "MarkReport",
    "Position",
    "PositionSnapshot",
    "ReplayAccount",
    "TradeEvent",
    "bars_from_frame",
    "load_segment_bars",
    # labels（policy）
    "ACTION_CLOSE",
    "ACTION_HOLD",
    "ACTION_OPEN_LONG",
    "ACTION_OPEN_SHORT",
    "ACTION_REVERSE",
    "ACTION_STAY_FLAT",
    "DecisionPoint",
    "DeathEvent",
    "SegmentOutcome",
    "apply_flat_band",
    "evaluate_segment",
    "holding_action",
    "reversal_conditions",
    "reward_risk",
    "select_open_action",
    # records（NanoJev 契约）
    "AUDIT_FILENAME",
    "FLAT_CRITERIA",
    "HELD_CRITERIA",
    "GOLD_LABEL_KIND",
    "QUESTION_ID",
    "STATE_SCHEMA",
    "BoardStateValues",
    "GenerationResult",
    "build_record",
    "generate_dataset",
    "render_state",
    # audit
    "AUDIT_SCHEMA",
    "FROZEN_DECISIONS",
    "build_audit_payload",
    "check_audit_consistency",
    "check_records",
    "check_split_isolation",
    "check_state_leakage",
    "parse_state_id",
    "verify_determinism",
]
