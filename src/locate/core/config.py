from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Config:
    def __init__(self, d: dict[str, Any]):
        self.raw = d
        self.api_alarm_verbose = bool(d.get("api", {}).get("alarm_verbose", True))
        g = d.get("geofence", {})
        self.gf_lat0 = float(g.get("lat0", 0.0))
        self.gf_lon0 = float(g.get("lon0", 0.0))
        self.gf_radius_m = float(g.get("radius_m", 500.0))
        self.gf_debounce_sec = int(g.get("debounce_sec", 10))


def load_config(path: str | Path) -> Config:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return Config(d)
