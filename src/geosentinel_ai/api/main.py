import os

from fastapi import FastAPI
from pydantic import BaseModel

from geosentinel_ai.utils.geo import in_radius

app = FastAPI(title="GeoSentinel-AI", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


class Point(BaseModel):
    lat: float
    lon: float


@app.post("/geofence/check")
def geofence_check(pt: Point):
    center_lat = float(os.getenv("GEOFENCE_CENTER_LAT", "41.015137"))
    center_lon = float(os.getenv("GEOFENCE_CENTER_LON", "28.979530"))
    radius_m = float(os.getenv("GEOFENCE_RADIUS_M", "1000"))
    inside = in_radius(center_lat, center_lon, pt.lat, pt.lon, radius_m)
    return {"inside": inside, "radius_m": radius_m}
