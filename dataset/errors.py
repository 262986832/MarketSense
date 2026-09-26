"""dataset 子应用异常体系。

所有数据准备相关错误继承 :class:`DatasetError`，便于 CLI 统一捕获与报告
（AGENTS.md 要求明确异常、错误消息不得泄露凭证）。

移植自早期版本数据层的 errors 模块（基类由
``DataSourceError`` 更名为 ``DatasetError``，语义与子类划分保持一致）。
"""

from __future__ import annotations


class DatasetError(Exception):
    """数据准备错误基类（配置、取数、标准化、校验、落盘、读取）。"""


class ConfigError(DatasetError):
    """配置错误（文件缺失、缺键、非法取值）。"""


class UnknownPeriodError(DatasetError):
    """未知周期（不在五档周期映射表内）。"""


class ProviderNotConnectedError(DatasetError):
    """未连接数据源即请求行情数据。"""


class DataLoadError(DatasetError):
    """落盘 OHLCV / 转折点文件读取错误（文件不存在 / 缺列 / 类型无法收敛）。"""


class ValidationError(DatasetError):
    """OHLCV 合法性校验失败（唯一/升序/非空有限/volume/OHLC 关系）。"""
