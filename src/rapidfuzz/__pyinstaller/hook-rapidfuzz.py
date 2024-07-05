# Pyinstaller hook to successfully freeze: https://pyinstaller.readthedocs.io/en/stable/hooks.html
from __future__ import annotations

from PyInstaller.utils.hooks import collect_submodules


def filterUnneededImports(name):
    if "__pyinstaller" in name:
        return False

    return True


hiddenimports = collect_submodules("rapidfuzz", filter=filterUnneededImports)
