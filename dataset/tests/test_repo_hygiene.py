"""仓库卫生（AC-9）：数据文件与本地凭证不入库，本地凭证路径被 ``.gitignore`` 覆盖。

纯只读检查（不写仓库、不 staging、不 commit）。``git`` 不可用或项目根不是 git
仓库时整体 ``skip``（例如分发副本），不误报失败。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(
    _GIT is None or not (_PROJECT_ROOT / ".git").exists(),
    reason="需要可用的 git 且项目根为 git 仓库",
)

#: 不得入库的文件形态（数据产物 / 本地凭证 / 密钥）
_FORBIDDEN_SUFFIXES = (".csv", ".local.yaml", ".local.yml", ".key", ".pem")
#: 数据产物根目录（``data/``、``datasets/`` 均在 .gitignore 中）
_FORBIDDEN_PREFIXES = ("data/", "datasets/")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_GIT, "-C", str(_PROJECT_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _forbidden(paths: list[str]) -> list[str]:
    """从 git 返回的路径列表中挑出「数据 / 凭证」类条目。"""
    offenders: list[str] = []
    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        if path.startswith(_FORBIDDEN_PREFIXES) or path.endswith(_FORBIDDEN_SUFFIXES):
            offenders.append(path)
    return offenders


def test_local_credential_file_is_ignored_by_git() -> None:
    """本地凭证文件（模板复制而来）必须被 ``.gitignore`` 覆盖。"""
    result = _git("check-ignore", "-v", "dataset/config/tianqin.local.yaml")

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "dataset/config/*.local.yaml" in result.stdout

    yml = _git("check-ignore", "-v", "dataset/config/tianqin.local.yml")
    assert yml.returncode == 0, (yml.stdout, yml.stderr)
    assert "dataset/config/*.local.yml" in yml.stdout


def test_data_output_directories_are_ignored_by_git() -> None:
    """K 线 / 转折点产物路径必须被 ``.gitignore`` 覆盖（含 CSV 与 sidecar）。"""
    paths = [
        "data/ohlcv/DCE.v2701_1d.csv",
        "data/ohlcv/DCE.v2701_1d.json",
        "data/turning_points/DCE.v2701_1d.csv",
    ]

    result = _git("check-ignore", "-v", *paths)

    assert result.returncode == 0, (result.stdout, result.stderr)
    for path in paths:
        assert path in result.stdout, result.stdout


def test_no_data_or_credential_files_are_tracked_or_staged() -> None:
    """已跟踪与已暂存的文件中不得出现数据产物或凭证（源码级 ``.yaml`` 模板除外）。"""
    tracked = _git("ls-files").stdout.splitlines()
    staged = _git("diff", "--cached", "--name-only").stdout.splitlines()

    assert _forbidden(tracked) == []
    assert _forbidden(staged) == []


def test_example_config_is_trackable_and_has_no_credentials() -> None:
    """入库模板必须可提交且凭证留空（真实凭证只进被忽略的 ``.local.yaml``）。"""
    example = "dataset/config/tianqin.example.yaml"

    listed = set(_git("ls-files").stdout.splitlines())
    untracked = set(_git("ls-files", "--others", "--exclude-standard").stdout.splitlines())
    assert example in listed | untracked  # 已跟踪或可跟踪（未被忽略）
    ignored = _git("check-ignore", example)
    assert ignored.returncode == 1  # 未被忽略 → 可入库
    assert "account: \"\"" in (_PROJECT_ROOT / example).read_text(encoding="utf-8")
    assert _forbidden([example]) == []
