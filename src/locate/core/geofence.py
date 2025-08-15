# src/locate/core/geofence.py
from __future__ import annotations

from dataclasses import dataclass
from time import time

from ..utils.geo import haversine_m


@dataclass
class GeofenceParams:
    lat0: float
    lon0: float
    radius_m: float
    debounce_sec: int = 10


class DebouncedGeofence:
    """
    Cihaz bazlı kısa süreli bellekle (in-memory) debounce uygular.
    state: device_id -> outside_since_ts (epoch saniye) veya None

    Not:
      - now_ts verilirse olay zamanına göre; verilmezse sistem saatine göre hesap yapar.
      - Replay/test senaryolarında her zaman now_ts geçmek önerilir.
    """

    def __init__(self, params: GeofenceParams):
        self.p = params
        self.state: dict[str, float | None] = {}

    def check(
        self, device_id: str, lat: float, lon: float, now_ts: float | None = None
    ) -> tuple[bool, float]:
        """
        Returns:
          (triggered, distance_m)
          triggered=True ise debounce penceresi dolmuş ve alarm üretilmeli.
        """
        now_ts = time() if now_ts is None else now_ts
        d = haversine_m(self.p.lat0, self.p.lon0, lat, lon)
        outside = d > self.p.radius_m

        since = self.state.get(device_id)
        if outside:
            if since is None:
                self.state[device_id] = now_ts  # yeni dışarıya çıktı
                return (False, d)
            else:
                if now_ts - since >= self.p.debounce_sec:
                    # alarm zamanı, fakat state'i içeride olana kadar koru
                    return (True, d)
                else:
                    return (False, d)
        else:
            # içeri döndü; state temizle
            if since is not None:
                self.state[device_id] = None
            return (False, d)
