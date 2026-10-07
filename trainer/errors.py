"""trainer 包内错误类型（trainer 不 import dataset，错误类型独立定义）。"""

from __future__ import annotations

__all__ = ["TrainerError"]


class TrainerError(Exception):
    """trainer 契约违规（数据/配置/校验失败）；CLI 捕获后以退出码 1 报告。"""
