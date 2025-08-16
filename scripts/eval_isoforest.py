# src/locate/ml/features.py
from __future__ import annotations

import numpy as np
import pandas as pd

# Var olan yardımcıyı kullan: geofence de aynı modülü kullanıyor
from locate.utils.geo import haversine_m

__all__ = ["build_features"]  # dışa açık semboller


def _bearing_rad(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Büyük daire (great-circle) başlık açısı (radyan).
    Kaynak: standard arctan2 formülü.
    """
    # deg -> rad
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(x, y)  # [-pi, pi]


def _ang_diff(a2, a1) -> np.ndarray:
    """Açı farkını (-pi, pi] aralığında normalize et."""
    d = a2 - a1
    return (d + np.pi) % (2 * np.pi) - np.pi


def build_features(
    df: pd.DataFrame,
    lat0: float,
    lon0: float,
    window_sec: int | None = 5,
) -> pd.DataFrame:
    """
    Giriş beklenen kolonlar: device_id, timestamp(ISO8601), lat, lon, speed(m/s).
    Çıkış: yalnızca sayısal özelliklerden oluşan DataFrame.
    """
    if not {"device_id", "timestamp", "lat", "lon"}.issubset(df.columns):
        raise ValueError("Eksik zorunlu kolon(lar): device_id/timestamp/lat/lon")

    g = df.copy()

    # tip & sıralama
    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
    g = g.sort_values(["device_id", "timestamp"], kind="stable")

    # Eğer 'speed' yoksa 0 kabul (m/s). Varsa m/s olduğu projede garanti.
    if "speed" not in g.columns:
        g["speed"] = 0.0

    # Grup bazında farklar
    g["lat_prev"] = g.groupby("device_id")["lat"].shift(1)
    g["lon_prev"] = g.groupby("device_id")["lon"].shift(1)
    g["t_prev"] = g.groupby("device_id")["timestamp"].shift(1)

    # dt (saniye)
    dt = (g["timestamp"] - g["t_prev"]).dt.total_seconds().astype("float32")
    dt = dt.replace([np.inf, -np.inf], np.nan)

    # iki nokta arası mesafe (metre)
    d_m = haversine_m(
        g["lat_prev"].fillna(g["lat"]), g["lon_prev"].fillna(g["lon"]), g["lat"], g["lon"]
    )
    d_m = d_m.astype("float32")

    # speed_mps_kaynak: veriden (speed); ek olarak d/dt ile "gözlenen hız"
    obs_speed = (d_m / dt).replace([np.inf, -np.inf], np.nan)
    speed = g["speed"].astype("float32").fillna(obs_speed).fillna(0.0)

    # ivme (m/s^2)
    speed_prev = g.groupby("device_id")["speed"].shift(1).astype("float32")
    accel = ((speed - speed_prev) / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # başlık açısı (radyan) ve dönüş hızı (rad/s)
    bearing = _bearing_rad(
        g["lat_prev"].fillna(g["lat"]), g["lon_prev"].fillna(g["lon"]), g["lat"], g["lon"]
    )
    bearing_prev = g.groupby("device_id")["lat"].shift(1)
    # bearing_prev'ı tekrar hesaplamak yerine bir önceki bearing'i kaydır:
    bearing_prev = pd.Series(bearing).shift(1)
    turn_rate = (
        (_ang_diff(bearing, bearing_prev) / dt)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )

    # merkeze uzaklık ve radyal hız
    dist_center = haversine_m(lat0, lon0, g["lat"], g["lon"]).astype("float32")
    dist_center_prev = dist_center.shift(1)
    radial_speed = (
        ((dist_center - dist_center_prev) / dt)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )

    # Örnekleme ~adımı tahmini (medyan dt); pencere boyuna çevir
    if window_sec and window_sec > 0:
        dt_med = dt.median()
        win = int(max(1, round(window_sec / dt_med))) if pd.notna(dt_med) and dt_med > 0 else 1
    else:
        win = 1

    # rolling özellikler (cihaz bazında)
    def roll_mean(s: pd.Series) -> pd.Series:
        return (
            s.groupby(g["device_id"])
            .rolling(win, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    speed_ma = roll_mean(speed)
    accel_ma = roll_mean(accel)
    radial_speed_ma = roll_mean(radial_speed)

    features: pd.DataFrame = (
        pd.DataFrame(
            {
                "speed": speed.values,
                "accel": accel.values,
                "turn_rate": turn_rate.values,
                "dist_center": dist_center.values,
                "radial_speed": radial_speed.values,
                "speed_ma": speed_ma.values,
                "accel_ma": accel_ma.values,
                "radial_speed_ma": radial_speed_ma.values,
            },
            index=g.index,
        )
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    return features.astype("float32")
