"""标准 OHLCV 落盘/读取与来源指纹。

* 单合约 + 单周期 → 一个文件（``{symbol}_{period}.csv``） + 同名 ``.json`` sidecar；
* ``source_data_version = tianqin|symbol|period|start|end|rows=N|first|last|sha256=…``；
* sidecar **不含墙钟时间**，保证重复运行逐字节一致；
* ``load_ohlcv`` 忠实读取：不排序、不去重、不修值（合法性校验归
  :mod:`dataset.validator`），仅做结构收敛。

移植自 ``reference/perception/src/marksense/data/storage.py`` 与
``reference/perception/src/marksense/data/loader.py``（两者合并到本文件；
输出格式固定 CSV，D-02）。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Mapping

import pandas as pd

from dataset.errors import DatasetError, DataLoadError
from dataset.ohlcv import OHLCV_COLUMNS, OI_COLUMNS, TIMEZONE

#: 落盘格式（D-02：本次仅 CSV）
OUTPUT_FORMAT: Final[str] = "csv"
#: sidecar 后缀与数据源标识
SIDECAR_SUFFIX: Final[str] = ".json"
PROVIDER_NAME: Final[str] = "tianqin"
_HASH_CHUNK_SIZE: Final[int] = 1 << 20  # 1 MiB，仅影响读取性能
_HASH_DIGEST_LENGTH: Final[int] = 16


@dataclass(frozen=True)
class LoadedOHLCV:
    """一次加载的结果：标准 OHLCV DataFrame 与可追溯元数据。

    :param df: 标准 OHLCV（timestamp tz-aware；**保持文件原序**）
    :param symbol: 合约代码
    :param period: 周期名
    :param path: 数据文件路径
    :param source_data_version: sidecar 中的溯源标识（无 sidecar 时为 None）
    :param row_count_declared: sidecar 声明的行数（无 sidecar 时为 None）
    """

    df: pd.DataFrame
    symbol: str
    period: str
    path: Path
    source_data_version: str | None
    row_count_declared: int | None


def ohlcv_filename(symbol: str, period: str) -> str:
    """落盘文件名：``{symbol}_{period}.csv``。"""
    return f"{symbol}_{period}.{OUTPUT_FORMAT}"


def file_sha256(path: Path) -> str:
    """计算文件 SHA-256（来源指纹的一部分）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_data_version(
    *,
    symbol: str,
    period: str,
    start: str | None,
    end: str | None,
    row_count: int,
    first_timestamp: str,
    last_timestamp: str,
    file_sha256: str,
) -> str:
    """构造 ``source_data_version``（口径固定，不含墙钟时间）。"""
    return (
        f"{PROVIDER_NAME}|{symbol}|{period}|{start or '-'}|{end or '-'}"
        f"|rows={row_count}|{first_timestamp}|{last_timestamp}"
        f"|sha256={file_sha256[:_HASH_DIGEST_LENGTH]}"
    )


def _require_non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{name} 必须为非空字符串")
    return value


def ensure_directory(path: Path) -> None:
    """创建输出目录（含父级）；失败统一收敛为 :class:`DatasetError`。

    裸 ``OSError``（``NotADirectoryError``/``PermissionError``/``FileExistsError`` 等）
    会绕过 CLI 的错误处理，导致 traceback + 非约定退出码（CLI 契约：stderr 给
    ``错误：…`` 且退出码非 0）。

    :raises DatasetError: 目录不可创建/不可用（消息只含输出路径与异常类型，不含凭证）
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatasetError(
            f"输出目录不可用（无法创建）: {path} ({type(exc).__name__}: {exc})"
        ) from None


def _stage_text(path: Path, text: str) -> Path:
    """把 ``text`` 写入 ``path`` 同目录的临时文件（``utf-8``，不翻译换行）。

    :return: 临时文件路径（调用方负责 ``os.replace`` 提交或删除）
    :raises DatasetError: 临时文件写入失败（磁盘/权限）
    """
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise DatasetError(f"落盘失败（临时文件）: {path} ({type(exc).__name__}: {exc})") from exc
    return tmp


def _replace_atomic(tmp: Path, final: Path) -> None:
    """同目录原子替换（``os.replace``）；失败报明确错误。"""
    try:
        os.replace(tmp, final)
    except OSError as exc:
        raise DatasetError(f"落盘失败: {final} ({type(exc).__name__}: {exc})") from exc


def commit_data_with_sidecar(
    path: Path,
    data_text: str,
    *,
    build_meta: Callable[[str], Mapping[str, object]],
) -> str:
    """原子落盘「数据文件 + sidecar，成对出现」；返回数据文件内容的 SHA-256。

    :param build_meta: 接收数据文件摘要，返回 sidecar 元数据（``file_sha256`` 由调用方填入）

    流程：两个文件先各写一份同目录临时文件 → 先 ``os.replace`` sidecar → 再
    ``os.replace`` 数据文件。数据文件**最后提交**，因此「数据文件存在」蕴含「sidecar
    已就位」；任一步失败都不会覆盖既有文件、也不留下半成品（临时文件在 ``finally``
    中清理，替换成功后路径已不存在，清理为幂等空操作）。
    """
    data_tmp = _stage_text(path, data_text)
    try:
        digest = file_sha256(data_tmp)
        sidecar = path.with_suffix(SIDECAR_SUFFIX)
        sidecar_tmp = _stage_text(
            sidecar,
            json.dumps(
                build_meta(digest), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
        )
        try:
            _replace_atomic(sidecar_tmp, sidecar)
            _replace_atomic(data_tmp, path)
        finally:
            sidecar_tmp.unlink(missing_ok=True)
    finally:
        data_tmp.unlink(missing_ok=True)
    return digest


def read_csv_table(path: Path) -> pd.DataFrame:
    """读取 CSV 表格；**任何**读取失败统一收敛为 :class:`DataLoadError`。

    与参考实现 ``reference/perception/src/marksense/data/loader.py::_read_raw`` 对齐：
    空文件（``pandas.errors.EmptyDataError``）、异编码（``UnicodeDecodeError``）、
    权限/缺失（``OSError``）等均包成 ``DataLoadError``，不让底层异常类型外泄。
    ``float_precision="round_trip"``：pandas 默认快速浮点解析可差 1 ulp，开启后
    写入的 float64 文本可无损读回（转折点相对值等需逐字节一致比较的场景依赖此保证）。
    """
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except Exception as exc:  # noqa: BLE001 — 读取失败一律收敛为 DataLoadError
        raise DataLoadError(f"文件读取失败: {path} ({type(exc).__name__}: {exc})") from None


def save_ohlcv(
    df: pd.DataFrame,
    *,
    symbol: str,
    period: str,
    output_dir: str | Path,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """把标准 OHLCV 落盘为 CSV，并写入 ``.json`` 来源指纹 sidecar。

    :param df: 已标准化的 OHLCV（含 ``timestamp`` 列，升序）
    :param symbol: 合约代码（``交易所.合约``）
    :param period: 周期名（如 ``1m``）
    :param output_dir: 落盘目录（不存在则创建）
    :param start/end: 拉取区间（写入 sidecar，可为空）
    :return: 落盘数据文件路径
    :raises DatasetError: 空数据 / symbol、period 非法 / 输出目录不可创建
    """
    symbol = _require_non_empty(symbol, "symbol")
    period = _require_non_empty(period, "period")
    if df is None or df.empty:
        raise DatasetError("拒绝落盘空 OHLCV")

    output_dir_path = Path(output_dir)
    ensure_directory(output_dir_path)
    path = output_dir_path / ohlcv_filename(symbol, period)

    first_timestamp = str(df["timestamp"].iloc[0])
    last_timestamp = str(df["timestamp"].iloc[-1])
    row_count = int(len(df))

    def _meta(digest: str) -> Mapping[str, object]:
        return {
            "provider": PROVIDER_NAME,
            "symbol": symbol,
            "period": period,
            "output_format": OUTPUT_FORMAT,
            "start_dt": start,
            "end_dt": end,
            "row_count": row_count,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "file_sha256": digest,
            "source_data_version": build_source_data_version(
                symbol=symbol,
                period=period,
                start=start,
                end=end,
                row_count=row_count,
                first_timestamp=first_timestamp,
                last_timestamp=last_timestamp,
                file_sha256=digest,
            ),
        }

    # 先写临时文件再 os.replace：CSV 与 sidecar 成对出现，失败不留半成品
    commit_data_with_sidecar(path, df.to_csv(index=False), build_meta=_meta)
    return path


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """结构收敛：列校验 + 类型转换；结构不符抛 :class:`DataLoadError`。

    OHLC 的缺失值**原样保留**（NaN 传递），由校验层判定。
    """
    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise DataLoadError(f"OHLCV 文件缺少必需列: {missing}")

    try:
        ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    except (TypeError, ValueError) as exc:
        raise DataLoadError(f"timestamp 无法解析为时间: {exc}") from exc
    # 与 standardize_klines 产出一致：统一为纳秒分辨率，避免 us/ns 混用
    ts = ts.dt.tz_convert(TIMEZONE).astype(f"datetime64[ns, {TIMEZONE}]")

    data: dict[str, pd.Series] = {"timestamp": ts}
    for col in ("open", "high", "low", "close"):
        try:
            data[col] = df[col].astype("float64")
        except (TypeError, ValueError) as exc:
            raise DataLoadError(f"{col} 无法转换为浮点数: {exc}") from exc

    volume = pd.to_numeric(df["volume"], errors="coerce")
    if bool(volume.isna().any()):
        raise DataLoadError("volume 存在缺失或非数值")
    if bool((volume % 1 != 0).any()):
        raise DataLoadError("volume 必须为整数，存在小数值")
    data["volume"] = volume.astype("int64")

    for col in OI_COLUMNS:  # 持仓量固定列，与 volume 同规整
        oi = pd.to_numeric(df[col], errors="coerce")
        if bool(oi.isna().any()):
            raise DataLoadError(f"{col} 存在缺失或非数值")
        if bool((oi % 1 != 0).any()):
            raise DataLoadError(f"{col} 必须为整数，存在小数值")
        data[col] = oi.astype("int64")

    extra = [c for c in df.columns if c not in OHLCV_COLUMNS]
    converted = pd.DataFrame(data)
    for col in extra:  # 未识别的额外列原样保留
        converted[col] = df[col].reset_index(drop=True)
    return converted


def _read_sidecar(path: Path) -> dict[str, object]:
    """读取同名 sidecar；不存在返回空映射，损坏抛 :class:`DataLoadError`。"""
    sidecar = path.with_suffix(SIDECAR_SUFFIX)
    if not sidecar.is_file():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DataLoadError(f"来源指纹文件损坏: {sidecar} ({exc})") from exc
    if not isinstance(payload, dict):
        raise DataLoadError(f"来源指纹文件必须是 JSON 对象: {sidecar}")
    return payload


def _declared_row_count(sidecar: Mapping[str, object], path: Path) -> int | None:
    """读取 sidecar 声明的行数；字段缺失/为 ``null`` 返回 ``None``，非法值抛错。"""
    raw = sidecar.get("row_count")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise DataLoadError(
            f"来源指纹 row_count 非法: {path.with_suffix(SIDECAR_SUFFIX)} ({raw!r})"
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise DataLoadError(
            f"来源指纹 row_count 非法: {path.with_suffix(SIDECAR_SUFFIX)} ({raw!r})"
        ) from exc


def _verify_sidecar_digest(sidecar: Mapping[str, object], path: Path) -> None:
    """重算文件 SHA-256 并与 sidecar 声明比对（显式开启时）。"""
    declared = sidecar.get("file_sha256")
    if not isinstance(declared, str) or not declared:
        raise DataLoadError(
            f"来源指纹缺少 file_sha256，无法校验完整性: {path.with_suffix(SIDECAR_SUFFIX)}"
        )
    actual = file_sha256(path)
    if actual != declared:
        raise DataLoadError(
            f"文件摘要与来源指纹不一致: {path}"
            f"（实际 sha256={actual[:_HASH_DIGEST_LENGTH]}…，"
            f"sidecar={declared[:_HASH_DIGEST_LENGTH]}…）"
        )


def load_ohlcv(
    symbol: str,
    period: str,
    *,
    data_dir: str | Path,
    verify_sha256: bool = False,
) -> LoadedOHLCV:
    """读取 ``{symbol}_{period}.csv`` 并做结构收敛（保持文件原序）。

    :param verify_sha256: 是否重算文件 SHA-256 并与 sidecar 比对（默认关闭；
        开启时代价为一次全文件哈希）
    :raises DataLoadError: 文件不存在 / 读取失败 / 缺列 / 类型无法收敛 /
        sidecar 损坏 / 行数或摘要与 sidecar 声明不一致
    """
    symbol = _require_non_empty(symbol, "symbol")
    period = _require_non_empty(period, "period")
    path = Path(data_dir) / ohlcv_filename(symbol, period)
    if not path.is_file():
        raise DataLoadError(f"OHLCV 文件不存在: {path}")

    df = _coerce_columns(read_csv_table(path))
    sidecar = _read_sidecar(path)
    row_count_declared = _declared_row_count(sidecar, path)
    if row_count_declared is not None and row_count_declared != len(df):
        raise DataLoadError(
            f"行数与来源指纹不一致: {path}（sidecar row_count={row_count_declared}，"
            f"实际 {len(df)} 行）"
        )
    if verify_sha256:
        _verify_sidecar_digest(sidecar, path)

    return LoadedOHLCV(
        df=df,
        symbol=symbol,
        period=period,
        path=path,
        source_data_version=sidecar.get("source_data_version"),
        row_count_declared=row_count_declared,
    )
