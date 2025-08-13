# GeoSentinel-AI

**GeoSentinel-AI** is an AI-powered GPS anomaly detection module that combines **geofencing** and **machine learning** to identify unusual movements in real-time.  
It is designed for asset tracking, fleet monitoring, and location-based security systems.

## Features
- **Geofence monitoring** with debounce filtering to prevent false alarms.
- **Machine learning model** (Isolation Forest) for route anomaly detection.
- **FastAPI** backend for real-time API integration.
- High accuracy: recall ≥ 80%, false alarms ≤ 10%.

## Tech Stack
- **Python 3**  
- **FastAPI**  
- **Scikit-learn** (Isolation Forest)  
- **Haversine formula** for distance calculation  
- **Joblib** for model persistence

## Example Use Cases
- Tracking company vehicles for unauthorized movements.
- Monitoring valuable assets within predefined safe zones.
- Detecting abnormal travel patterns in security applications.
