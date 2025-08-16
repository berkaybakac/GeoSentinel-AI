from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import atan2, cos, radians, sin

from ..utils.geo import haversine_m  # var: Haversine (metre)


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    φ1, φ2 = radians(lat1), radians(lat2)
    dλ = radians(lon2 - lon1)
    y = sin(dλ) * cos(φ2)
    x = cos(φ1) * sin(φ2) - sin(φ1) * cos(φ2) * cos(dλ)
    θ = atan2(y, x)
    deg = (θ * 180.0 / 3.141592653589793) % 360.0
    return deg


def _ang_diff_deg(a: float, b: float) -> float:
    """[-180, +180] aralığında açısal fark."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


@dataclass
class _Last:
    ts: datetime
    lat: float
    lon: float
    speed: float
    bearing: float | None


class OnlineFeatureState:
    """
    Cihaz bazında son noktayı saklar; bir sonraki ölçüm geldiğinde
    (speed, |accel|, |turn_rate|, dist_center_m) özelliklerini döndürür.
    """

    def __init__(self, lat0: float, lon0: float):
        self._last: dict[str, _Last] = {}
        self._lat0 = lat0
        self._lon0 = lon0

    def add_and_features(
        self, device_id: str, ts: datetime, lat: float, lon: float, speed: float
    ) -> list[float] | None:
        prev = self._last.get(device_id)
        brg = None
        if prev is not None:
            dt = (ts - prev.ts).total_seconds()
            if dt <= 0:
                # zaman geri gitmiş; sadece state'i güncelle
                brg = _bearing_deg(prev.lat, prev.lon, lat, lon)
                self._last[device_id] = _Last(ts, lat, lon, speed, brg)
                return None
            brg = _bearing_deg(prev.lat, prev.lon, lat, lon)
            accel = (speed - prev.speed) / dt
            turn_rate = 0.0 if prev.bearing is None else abs(_ang_diff_deg(brg, prev.bearing)) / dt
            dist_center = haversine_m(self._lat0, self._lon0, lat, lon)
            feats = [speed, abs(accel), abs(turn_rate), dist_center]
            self._last[device_id] = _Last(ts, lat, lon, speed, brg)
            return feats
        # ilk nokta → sadece state kur
        self._last[device_id] = _Last(ts, lat, lon, speed, None)
        return None
