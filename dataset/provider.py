"""DataProvider 抽象与天勤量化实现（api_factory 依赖注入）。

职责边界：``TianQinProvider`` 只负责「连天勤 → 取 K 线 → 标准化 → 落盘」，
不计算任何业务指标。两条取数路径：

* ``fetch_history``：``get_kline_data_series`` 任意历史区间（需天勤专业版权限）；
* ``fetch_recent``：``get_kline_serial`` 最近 N 根已收盘 K 线（上限 8964 根），
  内部经 :func:`dataset.ohlcv.clean_serial_klines` 剔除未收盘 K 线（防泄漏红线）。

实时订阅（``subscribe``）不移植（本次非目标）。

移植自 ``reference/perception/src/marksense/data/provider.py``（去掉 ``subscribe``
与心跳相关逻辑；「多取 1 根以凑满 N 根已收盘」由参考脚本 ``turning_points.py``
移入 Provider，见 ``fetch_recent``）。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Final

import pandas as pd

from dataset.config import DatasetConfig, require_credentials
from dataset.errors import DatasetError, ProviderNotConnectedError
from dataset.ohlcv import clean_serial_klines, parse_date_bound, standardize_klines
from dataset.periods import resolve_duration_seconds
from dataset.storage import save_ohlcv
from dataset.validator import validate_ohlcv

#: 创建底层 API 的工厂；测试注入桩实现，生产使用 :func:`_default_api_factory`
ApiFactory: Final = Callable[[DatasetConfig], Any]

#: ``get_kline_serial`` 单次请求最大长度（tqsdk 平台限制，非策略参数）
SERIAL_MAX_DATA_LENGTH: Final[int] = 8964
#: 等待 K 线序列就绪的默认超时（秒）
SERIAL_READY_TIMEOUT_SECONDS: Final[float] = 60.0


def validate_data_length(data_length: int) -> None:
    """校验 ``get_kline_serial`` 的 ``data_length``（平台限制，非策略参数）。"""
    if (
        not isinstance(data_length, int)
        or isinstance(data_length, bool)
        or not 1 <= data_length <= SERIAL_MAX_DATA_LENGTH
    ):
        raise DatasetError(
            f"data_length 必须为 1~{SERIAL_MAX_DATA_LENGTH} 的整数，实际: {data_length!r}"
        )


def _default_api_factory(config: DatasetConfig) -> Any:
    """生产环境 API 工厂：凭证缺失时明确报错（不回显凭证），tqsdk 延迟导入。"""
    require_credentials(config)
    from tqsdk import TqApi, TqAuth  # 延迟导入：测试环境无需触网

    return TqApi(auth=TqAuth(config.account, config.password))


class DataProvider(ABC):
    """数据源抽象基类。"""

    @abstractmethod
    def connect(self) -> None:
        """建立数据源连接。"""

    @abstractmethod
    def fetch_history(self, symbol: str, period: str, start: str, end: str) -> pd.DataFrame:
        """拉取 ``[start, end]`` 历史 K 线，返回标准 OHLCV DataFrame。"""

    @abstractmethod
    def fetch_recent(
        self, symbol: str, period: str, data_length: int = SERIAL_MAX_DATA_LENGTH
    ) -> pd.DataFrame:
        """拉取最近 ``data_length`` 根**已收盘** K 线，返回标准 OHLCV DataFrame。"""

    @abstractmethod
    def close(self) -> None:
        """释放连接（连接是稀缺资源）。"""


class TianQinProvider(DataProvider):
    """基于 tqsdk 的 K 线数据源。"""

    def __init__(
        self, config: DatasetConfig, api_factory: ApiFactory | None = None
    ) -> None:
        self._config = config
        self._api_factory: ApiFactory = api_factory or _default_api_factory
        self._api: Any | None = None

    def connect(self) -> None:
        """建立天勤连接；幂等，认证/网络异常统一包装为明确错误。"""
        if self._api is not None:
            return
        try:
            self._api = self._api_factory(self._config)
        except DatasetError:
            raise
        except Exception as exc:
            raise DatasetError(f"天勤连接/认证失败: {exc}") from exc

    def fetch_history(
        self, symbol: str, period: str, start: str, end: str
    ) -> pd.DataFrame:
        """拉取 ``[start, end]`` 历史 K 线并标准化。

        :param symbol: 合约代码（``交易所.合约``，如 ``DCE.v2701``）
        :param period: 周期名（``1m``/``5m``/``15m``/``1h``/``1d``）
        :param start: ISO 日期/时间字符串（含）
        :param end: ISO 日期/时间字符串（含）
        """
        if self._api is None:
            raise ProviderNotConnectedError("调用 fetch_history 前必须先 connect()")
        duration_seconds = resolve_duration_seconds(period)
        start_dt = parse_date_bound(start, "start")
        end_dt = parse_date_bound(end, "end")
        raw = self._api.get_kline_data_series(
            symbol=symbol,
            duration_seconds=duration_seconds,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        return standardize_klines(raw, include_oi=self._config.include_oi)

    def fetch_and_save(
        self, symbol: str, period: str, start: str, end: str
    ) -> Path:
        """``fetch_history → 校验 → 落盘``（区间模式）。"""
        df = self.fetch_history(symbol, period, start, end)
        validate_ohlcv(df).raise_if_invalid()
        return save_ohlcv(
            df,
            symbol=symbol,
            period=period,
            output_dir=self._ohlcv_dir(),
            start=start,
            end=end,
        )

    def fetch_recent(
        self,
        symbol: str,
        period: str,
        data_length: int = SERIAL_MAX_DATA_LENGTH,
        *,
        as_of: pd.Timestamp | None = None,
        wait_timeout_seconds: float = SERIAL_READY_TIMEOUT_SECONDS,
    ) -> pd.DataFrame:
        """``get_kline_serial`` 拉取最近 ``data_length`` 根**已收盘** K 线。

        因序列末根是「当前未收盘」K 线（交易时段），实现上多取 1 根再截取最后
        ``data_length`` 根，保证凑满 N 根已收盘。

        **边界（不静默）**：``data_length == SERIAL_MAX_DATA_LENGTH``（8964，平台上限）
        时无法再多取 1 根，末根未收盘被剔除后实际至多返回 8963 根；CLI 在返回根数
        少于请求根数时输出 stderr 告警（``dataset/cli.py::_warn_if_bars_shortfall``）。

        :param as_of: 判定已收盘的时间基准；缺省当前北京时间（测试可注入）
        :param wait_timeout_seconds: 等待序列就绪的超时（秒）
        """
        if self._api is None:
            raise ProviderNotConnectedError("调用 fetch_recent 前必须先 connect()")
        validate_data_length(data_length)
        duration_seconds = resolve_duration_seconds(period)
        request_length = (
            data_length + 1 if data_length < SERIAL_MAX_DATA_LENGTH else data_length
        )
        raw = self._api.get_kline_serial(
            symbol=symbol,
            duration_seconds=duration_seconds,
            data_length=request_length,
        )
        self._wait_serial_ready(raw, wait_timeout_seconds)
        df = clean_serial_klines(
            raw,
            duration_seconds=duration_seconds,
            as_of=as_of,
            include_oi=self._config.include_oi,
        )
        if len(df) > data_length:
            df = df.tail(data_length)
        return df.reset_index(drop=True)

    def fetch_recent_and_save(
        self,
        symbol: str,
        period: str,
        data_length: int = SERIAL_MAX_DATA_LENGTH,
        *,
        as_of: pd.Timestamp | None = None,
    ) -> Path:
        """``fetch_recent → 校验 → 落盘``（最近 N 根模式）。

        sidecar 的 ``start_dt``/``end_dt`` 取实际数据首末时间（该路径无请求区间概念）。
        """
        df = self.fetch_recent(symbol, period, data_length, as_of=as_of)
        validate_ohlcv(df).raise_if_invalid()
        return save_ohlcv(
            df,
            symbol=symbol,
            period=period,
            output_dir=self._ohlcv_dir(),
            start=str(df["timestamp"].iloc[0]),
            end=str(df["timestamp"].iloc[-1]),
        )

    def _ohlcv_dir(self) -> Path:
        """OHLCV 落盘子目录（``<output_dir>/ohlcv``）。"""
        return Path(self._config.output_dir) / "ohlcv"

    def _wait_serial_ready(self, klines: pd.DataFrame, timeout: float) -> None:
        """等待序列就绪（至少一根有效 K 线）；超时/断线报明确错误。"""
        if "id" not in klines.columns:
            raise DatasetError("K 线序列缺少必需列: id")
        ids = pd.to_numeric(klines["id"], errors="coerce")
        if bool((ids >= 0).any()):
            return

        wait_update = getattr(self._api, "wait_update", None)
        if wait_update is None:
            raise DatasetError("K 线序列中无有效数据行")
        loop_deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= loop_deadline:
                raise DatasetError(f"等待天勤K线序列就绪超时（{timeout}s）")
            try:
                # tqsdk wait_update 的 deadline 为绝对 unix 时间戳
                wait_update(deadline=min(loop_deadline, time.time() + 1.0))
            except Exception as exc:
                raise DatasetError(f"等待天勤K线数据时连接中断: {exc}") from exc
            ids = pd.to_numeric(klines["id"], errors="coerce")
            if bool((ids >= 0).any()):
                return

    def close(self) -> None:
        """释放天勤连接；幂等。"""
        if self._api is None:
            return
        self._api.close()
        self._api = None
