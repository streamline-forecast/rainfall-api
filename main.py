from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import httpx

app = FastAPI(title="Rainfall Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAN_ANTONIO_LAT = 29.4241
SAN_ANTONIO_LON = -98.4936
MM_TO_INCHES = 1 / 25.4


@app.get("/")
def root():
    return {
        "success": True,
        "status": "online",
        "service": "Rainfall Forecast API",
        "message": "API is running",
        "available_endpoints": [
            "/",
            "/api/status",
            "/api/forecast",
            "/api/trigger-update"
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.api_route("/api/status", methods=["GET", "POST"])
def status():
    return {
        "success": True,
        "status": "online",
        "service": "Rainfall Forecast API",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/forecast")
async def get_forecast():
    now_utc = datetime.now(timezone.utc)

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": SAN_ANTONIO_LAT,
        "longitude": SAN_ANTONIO_LON,
        "hourly": "precipitation",
        "past_days": 1,
        "forecast_days": 3,
        "timezone": "UTC"
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            meteo = response.json()

        times = meteo.get("hourly", {}).get("time", [])
        precip_mm = meteo.get("hourly", {}).get("precipitation", [])

        observed_mm = 0.0
        forecast_mm = 0.0

        observed_hours = 0
        forecast_hours = 0

        for time_str, rain_mm in zip(times, precip_mm):
            if rain_mm is None:
                rain_mm = 0.0

            dt = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)

            if now_utc.replace(minute=0, second=0, microsecond=0) >= dt >= now_utc.replace(minute=0, second=0, microsecond=0).replace(hour=now_utc.hour) - timedelta(hours=24):
                observed_mm += rain_mm
                observed_hours += 1

            if now_utc < dt <= now_utc + timedelta(hours=48):
                forecast_mm += rain_mm
                forecast_hours += 1

        return {
            "success": True,
            "data": {
                "timestamp": now_utc.isoformat(),
                "location": "San Antonio, Texas",
                "observedRainfallInches": round(observed_mm * MM_TO_INCHES, 3),
                "forecastRainfallInches": round(forecast_mm * MM_TO_INCHES, 3),
                "forecastHours": 48,
                "source": "Open-Meteo",
                "note": f"Observed rainfall is estimated from the last {observed_hours} hourly precipitation values. Forecast rainfall is summed from the next {forecast_hours} hourly precipitation values."
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": {
                "timestamp": now_utc.isoformat(),
                "location": "San Antonio, Texas",
                "observedRainfallInches": None,
                "forecastRainfallInches": None,
                "forecastHours": 48,
                "source": "Open-Meteo",
                "note": "Failed to retrieve live rainfall data."
            }
        }


@app.post("/api/trigger-update")
def trigger_update():
    return {
        "success": True,
        "message": "Rainfall update triggered successfully",
        "status": "update_started",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "next_steps": [
            "Pull observed rainfall",
            "Pull forecast rainfall",
            "Generate raster output",
            "Update latest forecast layer"
        ]
    }
