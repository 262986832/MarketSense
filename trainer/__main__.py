"""``python -m trainer`` 入口（见 :mod:`trainer.cli`）。"""

from __future__ import annotations

import sys

from trainer.cli import main

if __name__ == "__main__":
    sys.exit(main())
