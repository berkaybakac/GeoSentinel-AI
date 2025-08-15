from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..core.config import Config, load_config
from ..core.geofence import DebouncedGeofence, GeofenceParams

# -------------------- Pydantic şemaları --------------------


class DetectIn(BaseModel):
    device_id: str
    timestamp: datetime
    lat: float
    lon: float
    speed: float = Field(..., description="m/s")


class Alarm(BaseModel):
    code: int
    label: Literal["GEOFENCE_EXIT", "MODEL_ANOMALY", "SPEED_ANOMALY", "ROUTE_JUMP"]
    source: Literal["GEOFENCE", "MODEL", "RULE"]
    window_sec: int | None = None
    # MODEL durumunda faydalı alanlar (geofence için None bırakılır)
    score: float | None = None
    threshold: float | None = None


class DetectOutAlarm(BaseModel):
    device_id: str
    timestamp: str
    location: dict
    anomaly_reason: str
    alarm: Alarm


class DetectOutNormal(BaseModel):
    device_id: str
    timestamp: str
    anomaly: bool
    # verbose modda ek bilgi vereceğiz
    distance_m: float | None = None


# -------------------- App & Config --------------------

app = FastAPI(title="Locate – AI Anomali Tespit Modülü")

CFG_PATH = Path("configs/config.json")
cfg: Config = load_config(CFG_PATH)

# Env ile mod seçimi (öncelik env'de):
# API_MODE = "minimal" | "verbose"
api_mode_env = os.getenv("API_MODE", "").strip().lower()
if api_mode_env in {"minimal", "verbose"}:
    ALARM_VERBOSE = api_mode_env == "verbose"
else:
    # config.api.alarm_verbose sahasıyla eşleşsin
    ALARM_VERBOSE = bool(cfg.api_alarm_verbose)

gf = DebouncedGeofence(
    GeofenceParams(
        lat0=cfg.gf_lat0,
        lon0=cfg.gf_lon0,
        radius_m=cfg.gf_radius_m,
        debounce_sec=cfg.gf_debounce_sec,
    )
)


# -------------------- Endpoints --------------------


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "1.0",
        "mode": "verbose" if ALARM_VERBOSE else "minimal",
        "geofence": {
            "lat0": cfg.gf_lat0,
            "lon0": cfg.gf_lon0,
            "radius_m": cfg.gf_radius_m,
            "debounce_sec": cfg.gf_debounce_sec,
        },
    }


@app.post("/detect", response_model=DetectOutAlarm | DetectOutNormal)
def detect(inp: DetectIn):
    """
    Akış:
      1) Şema doğrulama (Pydantic)
      2) Geofence + debounce
      3) (ileride) ML model skoru
    """
    # Geofence kontrolü
    triggered, distance_m = gf.check(inp.device_id, inp.lat, inp.lon)

    # Zamanı ISO8601 UTC stringe çevir
    ts_str = inp.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    if triggered:
        # ---- Dokümantasyon 5.2 uyumlu alarm JSON ----
        return DetectOutAlarm(
            device_id=inp.device_id,
            timestamp=ts_str,
            location={"lat": inp.lat, "lon": inp.lon},
            anomaly_reason="GEOFENCE_EXIT",
            alarm=Alarm(
                code=1000,
                label="GEOFENCE_EXIT",
                source="GEOFENCE",
                window_sec=cfg.gf_debounce_sec,
            ),
        )

    # Alarm yok → minimal ya da verbose normal çıktı
    if ALARM_VERBOSE:
        return DetectOutNormal(
            device_id=inp.device_id,
            timestamp=ts_str,
            anomaly=False,
            distance_m=distance_m,
        )
    else:
        return DetectOutNormal(
            device_id=inp.device_id,
            timestamp=ts_str,
            anomaly=False,
        )
