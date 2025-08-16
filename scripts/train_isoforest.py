import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from locate.utils.geo import haversine_m


def bearing_deg(lat1, lon1, lat2, lon2):
    import numpy as np

    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dλ = np.radians(lon2 - lon1)
    y = np.sin(dλ) * np.cos(φ2)
    x = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(dλ)
    θ = np.arctan2(y, x)
    return (np.degrees(θ) + 360.0) % 360.0


def ang_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def build_features(df: pd.DataFrame, lat0: float, lon0: float) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["device_id", "timestamp"], kind="stable")
    # cihaz bazında öteleme
    for c in ["lat", "lon", "speed", "timestamp"]:
        df[f"prev_{c}"] = df.groupby("device_id")[c].shift(1)
    dt = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()
    brg = bearing_deg(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])
    brg_prev = df.groupby("device_id")[brg.name].shift(1)
    accel = (df["speed"] - df["prev_speed"]) / dt
    turn_rate = abs(ang_diff_deg(brg, brg_prev)) / dt
    dist_center = df.apply(
        lambda r: haversine_m(lat0, lon0, float(r["lat"]), float(r["lon"])), axis=1
    )
    X = pd.DataFrame(
        {
            "speed": df["speed"],
            "accel_abs": abs(accel),
            "turn_rate_abs": turn_rate,
            "dist_center_m": dist_center,
        }
    )
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="JSONL (lines=True)")
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    lat0 = cfg["geofence"]["lat0"]
    lon0 = cfg["geofence"]["lon0"]

    df = pd.read_json(args.inp, lines=True)
    X = build_features(df, lat0, lon0)

    # scaler
    scaler = None
    if cfg.get("features", {}).get("scaler", "standard") == "standard":
        scaler = StandardScaler().fit(X.values)
        Xv = scaler.transform(X.values)
    else:
        Xv = X.values

    # model
    mcfg = cfg.get("model", {})
    model = IsolationForest(
        n_estimators=mcfg.get("n_estimators", 256),
        max_samples=mcfg.get("max_samples", 1024),
        contamination=mcfg.get("contamination", 0.10),  # hedef FAR ~10%
        random_state=mcfg.get("random_state", 42),
    ).fit(Xv)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    dump(
        {"model": model, "scaler": scaler, "features": list(X.columns)},
        Path(args.outdir) / "isoforest.joblib",
    )
    (Path(args.outdir) / "isoforest.meta.json").write_text(
        json.dumps(
            {"features": list(X.columns), "count": int(len(X))}, ensure_ascii=False, indent=2
        )
    )
    print(f"[OK] model -> {Path(args.outdir) / 'isoforest.joblib'} | n={len(X)}")


if __name__ == "__main__":
    main()
