# scripts/prepare_dbra24.py
import argparse
import json
from datetime import UTC  # <-- eklendi
from pathlib import Path

import pandas as pd
from dateutil import parser as dtp


def to_iso8601_utc(ts):
    """
    Her zaman ISO 8601 UTC (Z) döndürür.
    - Naive timestamp -> UTC varsayılır.
    - Aware timestamp -> UTC'ye dönüştürülür.
    """
    dt = dtp.parse(str(ts))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="DBRA24 kaynak dosya (CSV/Parquet)")
    ap.add_argument("--mapping", default="configs/mapping_dbra24.json")
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--out", default="data/processed/dbra24_test.jsonl")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    mp = json.loads(Path(args.mapping).read_text(encoding="utf-8"))

    src = Path(args.csv)
    df = pd.read_parquet(src) if src.suffix.lower() == ".parquet" else pd.read_csv(src)

    rename = {
        mp["timestamp"]: "timestamp",
        mp["lat"]: "lat",
        mp["lon"]: "lon",
        mp["speed_kmh"]: "speed_kmh",
        mp["device_id"]: "device_id",
    }
    rename = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=rename)

    for c in ["timestamp", "lat", "lon", "device_id"]:
        if c not in df.columns:
            raise ValueError(f"Eksik zorunlu kolon: {c}")

    # km/h -> m/s
    if "speed_kmh" in df.columns:
        df["speed"] = df["speed_kmh"] / 3.6
    elif "speed" not in df.columns:
        df["speed"] = 0.0

    # temel temizlik
    df = df.dropna(subset=["timestamp", "lat", "lon", "device_id"])

    # opsiyonel hız filtresi
    max_kmh = cfg.get("filters", {}).get("max_speed_kmh", None)
    if max_kmh is not None:
        df = df[df["speed"] <= (max_kmh / 3.6)]

    # Zamanı UTC-Z'ye normalle
    df["timestamp"] = df["timestamp"].map(to_iso8601_utc)
    df = df.sort_values(by=["device_id", "timestamp"], kind="stable")

    # etiketler (varsa)
    labels = mp.get("labels", {})
    out_cols = ["device_id", "timestamp", "lat", "lon", "speed"]
    for k in ["geofence", "route", "event"]:
        col = labels.get(k)
        if col and col in df.columns:
            out_cols.append(col)

    # JSONL yaz
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for _, r in df[out_cols].iterrows():
            rec = {
                "device_id": str(r["device_id"]),
                "timestamp": r["timestamp"],
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "speed": float(r["speed"]),
            }
            if labels.get("geofence") in out_cols:
                rec["label_geofence"] = bool(r.get(labels["geofence"], False))
            if labels.get("route") in out_cols:
                rec["label_route"] = bool(r.get(labels["route"], False))
            if labels.get("event") in out_cols:
                rec["label_event"] = bool(r.get(labels["event"], False))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] -> {out} | Kayıt: {len(df)}")


if __name__ == "__main__":
    main()
