from copy import deepcopy
import os
from typing import Any, Dict


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_yaml_with_encodings(path: str) -> Dict[str, Any]:
    import yaml
    # Try UTF-8 first; then UTF-8 with BOM; then cp1252 as last resort.
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("yaml", b"", 0, 1, f"Could not decode {path} with utf-8/utf-8-sig/cp1252")


def load_config(path: str) -> Dict[str, Any]:
    cfg_task = _load_yaml_with_encodings(path)

    # Optional base.yaml merge
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    if os.path.basename(path) != "base.yaml" and os.path.exists(base_path):
        cfg_base = _load_yaml_with_encodings(base_path)
        return deep_update(deepcopy(cfg_base), cfg_task)
    return cfg_task
