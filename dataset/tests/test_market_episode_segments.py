"""T1：片段清单 schema / 品种 tick 配置 / 加载与校验。

覆盖：合法清单加载成功；缺字段、坏时间、跨 split 重叠、缺 train/dev/test 角色、
非 1m 周期、越界时间段被拒绝；tick 配置缺失或非法被拒绝；配置解析优先级。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dataset.config import ENV_DATA_DIR
from dataset.errors import ConfigError, DatasetError
from dataset.market_episode.segments import (
    DEFAULT_EPISODE_OUTPUT_DIR,
    SEGMENT_SCHEMA,
    SPLIT_ROLES,
    EpisodeParams,
    load_episode_config,
    load_episode_params,
    load_segments,
    load_symbols_config,
    validate_segments,
)
from dataset.tests.market_episode_fixtures import (
    SYMBOL,
    TICK_SIZE,
    build_workspace,
    segment_record,
    write_manifest,
    write_symbols_config,
)

_ROWS = [(100, 101, 99, 100)] * 9
_SYMBOLS = {SYMBOL: TICK_SIZE}


def _valid_records(count: int = 3) -> list[dict]:
    roles = ["train", "dev", "test"]
    return [
        segment_record(f"seg-{role}", role, start_index=3 * index, end_index=3 * index + 2)
        for index, role in enumerate(roles[:count])
    ]


def test_load_segments_parses_valid_manifest(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path / "segments.jsonl", _valid_records())

    segments = load_segments(manifest)

    assert [segment.segment_id for segment in segments] == ["seg-train", "seg-dev", "seg-test"]
    assert segments[0].split_role == "train"
    assert segments[0].symbol == SYMBOL
    assert segments[0].period == "1m"
    assert segments[0].start < segments[0].end
    assert set(SPLIT_ROLES) >= {segment.split_role for segment in segments}
    assert segments[2].notes is None


def test_load_segments_keeps_notes_and_optional_fields(tmp_path: Path) -> None:
    record = segment_record("seg-train", "train", start_index=0, end_index=1, notes="震荡片段")
    manifest = write_manifest(tmp_path / "segments.jsonl", [record])

    segments = load_segments(manifest)

    assert segments[0].notes == "震荡片段"


def test_load_segments_rejects_missing_field(tmp_path: Path) -> None:
    record = _valid_records()[0]
    record.pop("split_role")
    manifest = write_manifest(tmp_path / "segments.jsonl", [record])

    with pytest.raises(DatasetError, match="缺少必需字段"):
        load_segments(manifest)


def test_load_segments_rejects_empty_value(tmp_path: Path) -> None:
    record = _valid_records()[0]
    record["symbol"] = "  "
    manifest = write_manifest(tmp_path / "segments.jsonl", [record])

    with pytest.raises(DatasetError, match="必须为非空字符串"):
        load_segments(manifest)


def test_load_segments_rejects_bad_schema_marker(tmp_path: Path) -> None:
    record = _valid_records()[0]
    record["schema"] = "marketsense.segment.v0"
    manifest = write_manifest(tmp_path / "segments.jsonl", [record])

    with pytest.raises(DatasetError, match="schema 标记"):
        load_segments(manifest)


def test_load_segments_rejects_unknown_field(tmp_path: Path) -> None:
    record = _valid_records()[0]
    record["tick_size"] = 1
    manifest = write_manifest(tmp_path / "segments.jsonl", [record])

    with pytest.raises(DatasetError, match="未知字段"):
        load_segments(manifest)


def test_load_segments_rejects_non_one_minute_period(tmp_path: Path) -> None:
    record = _valid_records()[0]
    record["period"] = "5m"
    manifest = write_manifest(tmp_path / "segments.jsonl", [record])

    with pytest.raises(DatasetError, match="period 必须为 '1m'"):
        load_segments(manifest)


def test_load_segments_rejects_unknown_split_role(tmp_path: Path) -> None:
    record = _valid_records()[0]
    record["split_role"] = "eval"
    manifest = write_manifest(tmp_path / "segments.jsonl", [record])

    with pytest.raises(DatasetError, match="split_role"):
        load_segments(manifest)


def test_load_segments_rejects_bad_timestamp_and_reversed_range(tmp_path: Path) -> None:
    record = _valid_records()[0]
    record["start_ts"] = "not-a-time"
    manifest = write_manifest(tmp_path / "segments.jsonl", [record])
    with pytest.raises(DatasetError, match="不是合法时间"):
        load_segments(manifest)

    reversed_record = _valid_records()[0]
    reversed_record["start_ts"], reversed_record["end_ts"] = (
        reversed_record["end_ts"],
        reversed_record["start_ts"],
    )
    manifest = write_manifest(tmp_path / "segments_bad.jsonl", [reversed_record])
    with pytest.raises(DatasetError, match="start_ts 必须不晚于 end_ts"):
        load_segments(manifest)


def test_load_segments_rejects_duplicate_id_and_duplicate_json_key(tmp_path: Path) -> None:
    records = _valid_records()
    records[1]["segment_id"] = records[0]["segment_id"]
    manifest = write_manifest(tmp_path / "segments.jsonl", records)
    with pytest.raises(DatasetError, match="segment_id 重复"):
        load_segments(manifest)

    duplicated = tmp_path / "dup_key.jsonl"
    duplicated.write_text(
        json.dumps(_valid_records()[0])[:-1] + ', "symbol": "OTHER.sym"}', encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="JSON 解析失败"):
        load_segments(duplicated)


def test_load_segments_rejects_empty_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("\n\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="片段清单为空"):
        load_segments(manifest)


def test_load_segments_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="片段清单不存在"):
        load_segments(tmp_path / "nope.jsonl")


def test_validate_segments_accepts_covering_manifest(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, _ROWS)

    validate_segments(load_segments(workspace.manifest), data_dir=workspace.data_dir, symbols=_SYMBOLS)


def test_load_segments_accepts_single_bar_segment(tmp_path: Path) -> None:
    """端点含：``start == end`` 是合法的单根 K 线片段。"""
    manifest = write_manifest(
        tmp_path / "one_bar.jsonl",
        [segment_record("seg-train", "train", start_index=0, end_index=0)],
    )

    segments = load_segments(manifest)

    assert segments[0].start == segments[0].end


def test_validate_segments_rejects_missing_train_dev_test(tmp_path: Path) -> None:
    records = [
        segment_record("seg-a", "train", start_index=0, end_index=2),
        segment_record("seg-b", "train", start_index=3, end_index=5),
        segment_record("seg-c", "train", start_index=6, end_index=8),
    ]
    workspace = build_workspace(tmp_path, _ROWS)
    manifest = write_manifest(tmp_path / "only_train.jsonl", records)
    segments = load_segments(manifest)

    with pytest.raises(DatasetError, match="train/dev/test"):
        validate_segments(segments, data_dir=workspace.data_dir, symbols=_SYMBOLS)


def test_validate_segments_rejects_cross_split_overlap(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, _ROWS)
    records = [
        segment_record("seg-a", "train", start_index=0, end_index=4),
        segment_record("seg-b", "dev", start_index=4, end_index=5),  # 与 seg-a 在 bar 4 重叠
        segment_record("seg-c", "test", start_index=6, end_index=8),
    ]
    manifest = write_manifest(tmp_path / "overlap.jsonl", records)
    segments = load_segments(manifest)

    with pytest.raises(DatasetError, match="不同 split 片段时间重叠"):
        validate_segments(segments, data_dir=workspace.data_dir, symbols=_SYMBOLS)


def test_validate_segments_allows_same_role_overlap(tmp_path: Path) -> None:
    """冻结口径只约束「不同 split_role」不重叠；同角色重叠不拒绝。"""
    workspace = build_workspace(tmp_path, _ROWS)
    records = [
        segment_record("seg-a", "train", start_index=0, end_index=4),
        segment_record("seg-b", "train", start_index=4, end_index=5),
        segment_record("seg-c", "dev", start_index=6, end_index=6),
        segment_record("seg-d", "test", start_index=7, end_index=8),
    ]
    manifest = write_manifest(tmp_path / "same_role.jsonl", records)

    validate_segments(
        load_segments(manifest), data_dir=workspace.data_dir, symbols=_SYMBOLS
    )


def test_validate_segments_rejects_symbol_without_tick_size(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, _ROWS)
    segments = load_segments(workspace.manifest)

    with pytest.raises(DatasetError, match="品种配置缺少 tick_size"):
        validate_segments(segments, data_dir=workspace.data_dir, symbols={"OTHER.sym": 1.0})


def test_validate_segments_rejects_segment_outside_data_range(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, _ROWS)
    records = [
        segment_record("seg-train", "train", start_index=0, end_index=1),
        segment_record("seg-dev", "dev", start_index=2, end_index=2),
        segment_record("seg-test", "test", start_index=3, end_index=99),  # 超出数据范围
    ]
    manifest = write_manifest(tmp_path / "out_of_range.jsonl", records)

    with pytest.raises(DatasetError, match="超出数据范围"):
        validate_segments(
            load_segments(manifest), data_dir=workspace.data_dir, symbols=_SYMBOLS
        )


def test_validate_segments_rejects_segment_after_data_end(tmp_path: Path) -> None:
    """片段落在数据范围之外（数据尚未交付到该时段）必须报错。"""
    workspace = build_workspace(tmp_path, _ROWS)
    records = [
        segment_record("seg-train", "train", start_index=0, end_index=5),
        segment_record("seg-dev", "dev", start_index=6, end_index=6),
    ]
    later = segment_record("seg-test", "test", start_index=3, end_index=3)
    later["start_ts"] = "2024-01-05 09:00:00"
    later["end_ts"] = "2024-01-05 09:05:00"
    records.append(later)
    manifest = write_manifest(tmp_path / "beyond.jsonl", records)

    with pytest.raises(DatasetError, match="超出数据范围"):
        validate_segments(
            load_segments(manifest), data_dir=workspace.data_dir, symbols=_SYMBOLS
        )


def test_validate_segments_rejects_missing_ohlcv_file(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, _ROWS)
    segments = load_segments(workspace.manifest)
    empty_dir = tmp_path / "no_data"

    with pytest.raises(DatasetError, match="读取 K 线失败"):
        validate_segments(segments, data_dir=empty_dir, symbols=_SYMBOLS)


def test_load_symbols_config_accepts_example_shape(tmp_path: Path) -> None:
    path = write_symbols_config(tmp_path / "symbols.yaml", tick_size=5)

    table = load_symbols_config(path)

    assert table == {SYMBOL: 5.0}


def test_load_symbols_config_rejects_invalid(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="品种配置不存在"):
        load_symbols_config(tmp_path / "missing.yaml")

    bad_schema = tmp_path / "bad_schema.yaml"
    bad_schema.write_text(yaml.safe_dump({"symbols": {SYMBOL: {"tick_size": 1}}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="schema 标记"):
        load_symbols_config(bad_schema)

    unknown_field = tmp_path / "unknown.yaml"
    unknown_field.write_text(
        yaml.safe_dump(
            {
                "schema": "marketsense.symbols.v1",
                "symbols": {SYMBOL: {"tick_size": 1, "multiplier": 5}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="未知字段"):
        load_symbols_config(unknown_field)

    bad_tick = tmp_path / "bad_tick.yaml"
    bad_tick.write_text(
        yaml.safe_dump(
            {"schema": "marketsense.symbols.v1", "symbols": {SYMBOL: {"tick_size": 0}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="tick_size 必须为正数"):
        load_symbols_config(bad_tick)


def test_load_episode_params_defaults_and_overrides() -> None:
    assert EpisodeParams().as_dict() == {
        "drawdown_threshold": 0.05,
        "reward_risk_threshold": 3.0,
        "price_precision": 6,
        "flat_sample_band_minutes": 2,
        "event_lookforward_open": 2,
        "event_lookforward_exit": 10,
        "breakthrough_window": 20,
        "breakthrough_period": "1m",
        "linkage_symbols": [],
    }

    params = load_episode_params(
        {
            "drawdown_threshold": 0.02,
            "reward_risk_threshold": 2.5,
            "price_precision": 4,
            "flat_sample_band_minutes": 0,
            "event_lookforward_open": 5,
            "event_lookforward_exit": 0,
        },
        where="测试配置段",
    )

    assert params == EpisodeParams(
        drawdown_threshold=0.02,
        reward_risk_threshold=2.5,
        price_precision=4,
        flat_sample_band_minutes=0,
        event_lookforward_open=5,
        event_lookforward_exit=0,
    )


@pytest.mark.parametrize(
    "section, message",
    [
        ({"unknown": 1}, "未知配置项"),
        ({"drawdown_threshold": 1.5}, "必须小于 1"),
        ({"drawdown_threshold": 0}, "必须为正数"),
        ({"reward_risk_threshold": -1}, "必须为正数"),
        ({"price_precision": 13}, "必须 ≤ 12"),
        ({"flat_sample_band_minutes": -1}, "必须为非负整数"),
        ({"event_lookforward_open": -1}, "必须为非负整数"),
        ({"event_lookforward_exit": -1}, "必须为非负整数"),
    ],
)
def test_load_episode_params_rejects_invalid(section: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_episode_params(section, where="测试配置段")


def test_load_episode_config_resolves_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(ENV_DATA_DIR, raising=False)
    workspace = build_workspace(tmp_path, _ROWS, episode={"reward_risk_threshold": 4.0})

    config = load_episode_config(workspace.config_path)

    assert config.data_dir == workspace.data_dir
    assert config.output_dir == Path(DEFAULT_EPISODE_OUTPUT_DIR)
    assert config.symbols_path == workspace.symbols_path
    assert config.params.reward_risk_threshold == 4.0
    assert config.config_path == workspace.config_path


def test_load_episode_config_applies_overrides_and_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(ENV_DATA_DIR, raising=False)
    workspace = build_workspace(tmp_path, _ROWS)
    config = load_episode_config(
        workspace.config_path, output_dir=tmp_path / "explicit", data_dir=tmp_path / "explicit_ohlcv"
    )
    assert config.output_dir == tmp_path / "explicit"
    assert config.data_dir == tmp_path / "explicit_ohlcv"

    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "envdata"))
    config = load_episode_config(workspace.config_path)
    assert config.data_dir == tmp_path / "envdata" / "ohlcv"


def test_load_episode_config_rejects_missing_config_and_symbols(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MARKETSENSE_DATASET_CONFIG", raising=False)
    with pytest.raises(ConfigError, match="配置文件不存在"):
        load_episode_config(tmp_path / "nope.yaml")

    config_path = tmp_path / "only_dataset.yaml"
    config_path.write_text(
        yaml.safe_dump({"dataset": {"output_dir": "data"}}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="品种配置不存在"):
        load_episode_config(config_path)


def test_load_episode_config_rejects_unknown_episode_key(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, _ROWS, episode={"band_minutes": 3})

    with pytest.raises(ConfigError, match="未知配置项"):
        load_episode_config(workspace.config_path)


def test_example_templates_exist_and_parse() -> None:
    root = Path(__file__).resolve().parents[1] / "config"
    segments = root / "segments.example.jsonl"
    symbols = root / "symbols.example.yaml"

    records = [
        json.loads(line) for line in segments.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert records
    for record in records:
        assert record["schema"] == SEGMENT_SCHEMA
        assert record["period"] == "1m"
        assert record["split_role"] in SPLIT_ROLES
    assert {record["split_role"] for record in records} >= {"train", "dev", "test"}
    assert load_symbols_config(symbols)  # example 的 tick 配置可被解析
