import gzip
import os
import tempfile
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Rainfall Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_LAT = 29.4241
DEFAULT_LON = -98.4936
DEFAULT_LOCATION = "San Antonio, Texas"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MRMS_BASE_URL = "https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/"
MM_TO_INCHES = 1 / 25.4


@app.get("/")
async def root():
    return {
        "success": True,
        "status": "online",
        "service": "Rainfall Forecast API",
        "docs": "/docs",
        "available_endpoints": [
            "/",
            "/api/status",
            "/api/forecast",
            "/api/trigger-update",
            "/api/mrms/latest",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.api_route("/api/status", methods=["GET", "POST"])
async def get_status():
    return {
        "success": True,
        "status": "online",
        "service": "Rainfall Forecast API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def fetch_rainfall(lat: float, lon: float):
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    observed_start = now_utc - timedelta(hours=24)
    forecast_end = now_utc + timedelta(hours=48)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "past_days": 1,
        "forecast_days": 3,
        "timezone": "UTC",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(OPEN_METEO_URL, params=params)
        response.raise_for_status()
        data = response.json()

    times = data.get("hourly", {}).get("time", [])
    precipitation_mm = data.get("hourly", {}).get("precipitation", [])

    observed_mm = 0.0
    forecast_mm = 0.0
    observed_hours = 0
    forecast_hours = 0

    for time_str, rain_mm in zip(times, precipitation_mm):
        if rain_mm is None:
            rain_mm = 0.0

        dt = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)

        if observed_start <= dt <= now_utc:
            observed_mm += rain_mm
            observed_hours += 1

        if now_utc < dt <= forecast_end:
            forecast_mm += rain_mm
            forecast_hours += 1

    return {
        "observedRainfallInches": round(observed_mm * MM_TO_INCHES, 3),
        "forecastRainfallInches": round(forecast_mm * MM_TO_INCHES, 3),
        "forecastHours": forecast_hours,
        "observedWindowHours": observed_hours,
    }


@app.get("/api/forecast")
async def get_forecast(
    lat: float = Query(default=DEFAULT_LAT),
    lon: float = Query(default=DEFAULT_LON),
    location: str = Query(default=DEFAULT_LOCATION),
):
    try:
        rainfall = await fetch_rainfall(lat, lon)

        return {
            "success": True,
            "data": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "location": location,
                "observedRainfallInches": rainfall["observedRainfallInches"],
                "forecastRainfallInches": rainfall["forecastRainfallInches"],
                "forecastHours": rainfall["forecastHours"],
                "source": "Open-Meteo",
                "note": (
                    f"Observed rainfall is estimated from the last "
                    f"{rainfall['observedWindowHours']} hourly precipitation values. "
                    f"Forecast rainfall is summed from the next "
                    f"{rainfall['forecastHours']} hourly precipitation values. "
                    "Data provided by Open-Meteo."
                ),
            },
        }

    except httpx.HTTPError as error:
        return {
            "success": False,
            "error": f"Open-Meteo request failed: {str(error)}",
            "data": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "location": location,
                "observedRainfallInches": None,
                "forecastRainfallInches": None,
                "forecastHours": 48,
                "source": "Open-Meteo",
                "note": "Failed to retrieve live rainfall data.",
            },
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
            "data": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "location": location,
                "observedRainfallInches": None,
                "forecastRainfallInches": None,
                "forecastHours": 48,
                "source": "Open-Meteo",
                "note": "Unexpected error retrieving rainfall data.",
            },
        }


@app.post("/api/trigger-update")
async def trigger_update():
    return {
        "success": True,
        "message": "Update triggered. Open-Meteo data is fetched live on each /api/forecast request.",
        "status": "update_started",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "next_steps": [
            "Pull observed rainfall",
            "Pull forecast rainfall",
            "Retrieve MRMS raster data",
            "Generate raster output",
            "Update latest forecast layer",
        ],
    }


@app.get("/api/mrms/latest")
async def get_latest_mrms_product():
    product_name = "MultiSensor_QPE_01H_Pass2"
    latest_filename = f"{product_name}.latest.grib2.gz"
    download_url = f"{MRMS_BASE_URL}{latest_filename}"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(download_url)
            response.raise_for_status()

        gz_path = os.path.join(tempfile.gettempdir(), latest_filename)

        with open(gz_path, "wb") as file:
            file.write(response.content)

        compressed_file_size_bytes = os.path.getsize(gz_path)

        grib2_filename = latest_filename.replace(".gz", "")
        grib2_path = os.path.join(tempfile.gettempdir(), grib2_filename)

        with gzip.open(gz_path, "rb") as source:
            with open(grib2_path, "wb") as target:
                target.write(source.read())

        grib2_file_size_bytes = os.path.getsize(grib2_path)

        os.remove(gz_path)

        return {
            "success": True,
            "product": product_name,
            "source_url": download_url,
            "downloaded_filename": latest_filename,
            "grib2_filename": grib2_filename,
            "compressed_file_size_bytes": compressed_file_size_bytes,
            "grib2_file_size_bytes": grib2_file_size_bytes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "Latest MRMS 1-hour observed rainfall product downloaded and decompressed to temporary server storage.",
        }

    except httpx.HTTPStatusError as error:
        return {
            "success": False,
            "product": product_name,
            "source_url": download_url,
            "error": f"HTTP error downloading MRMS data: {error.response.status_code}",
            "note": "Check the MRMS source URL or server availability.",
        }

    except httpx.RequestError as error:
        return {
            "success": False,
            "product": product_name,
            "source_url": download_url,
            "error": f"Network error downloading MRMS data: {str(error)}",
            "note": "Check network connectivity or MRMS server availability.",
        }

    except Exception as error:
        return {
            "success": False,
            "product": product_name,
            "source_url": download_url,
            "error": str(error),
            "note": "Unexpected error retrieving or decompressing MRMS data.",
        }
