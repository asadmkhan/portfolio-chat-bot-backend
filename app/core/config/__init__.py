from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

def _load_legacy_settings_module() -> ModuleType:
    """Load existing app/core/config.py to keep backward compatibility."""
    legacy_path = Path(__file__).resolve().parent.parent / "config.py"
    module_name = "app.core._legacy_settings"

    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy settings module at '{legacy_path}'.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_legacy_settings = _load_legacy_settings_module()
Settings = _legacy_settings.Settings
settings = _legacy_settings.settings

__all__ = ["Settings", "settings"]
