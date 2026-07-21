from __future__ import annotations

from pathlib import Path


def get_PyInstaller_tests() -> list[str]:
    return [str(Path(__file__).parent)]
