"""
Rainfall API - FastAPI backend
Deployed on Render

Provides:
- Live forecast rainfall from Open-Meteo
- NOAA MRMS realtime gridded rainfall retrieval
- Phase 3A overlay preparation endpoint
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import httpx
import gzip
import os
import re
import tempfile

app = FastAPI(title="Rainfall API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DEFAULT CONFIG
# =========================================================

DEFAULT_LAT = 29.4241
DEFAULT_LON = -98.4936
DEFAULT_LOCATION = "San Antonio, Texas"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

MRMS_BASE_URL = "https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/"
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2"


# =========================================================
# ROOT
# =========================================================

@app.get("/api/mrms/latest")
async def mrms_latest():
    try:
        latest_filename = f"MRMS_{MRMS_PRODUCT}.latest.grib2.gz"
        source_url = f"{MRMS_BASE_URL}{latest_filename}"

        gz_path = f"/tmp/{latest_filename}"
        grib2_filename = latest_filename.replace(".gz", "")
        grib2_path = f"/tmp/{grib2_filename}"

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(source_url)
            response.raise_for_status()

            with open(gz_path, "wb") as file:
                file.write(response.content)

        gz_size = os.path.getsize(gz_path)

        with gzip.open(gz_path, "rb") as gz_in:
            with open(grib2_path, "wb") as grib_out:
                grib_out.write(gz_in.read())

        grib2_size = os.path.getsize(grib2_path)
        os.remove(gz_path)

        return {
            "success": True,
            "product": MRMS_PRODUCT,
            "source_url": source_url,
            "downloaded_filename": latest_filename,
            "grib2_filename": grib2_filename,
            "compressed_file_size_bytes": gz_size,
            "grib2_file_size_bytes": grib2_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "Downloaded and decompressed latest NOAA MRMS 1-hour observed rainfall grid.",
        }

    except httpx.HTTPError as error:
        return {
            "success": False,
            "error": f"HTTP error fetching MRMS data: {str(error)}",
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
        }

# =========================================================
# STATUS
# =========================================================

@app.api_route("/api/status", methods=["GET", "POST"])
async def get_status():
    return {
        "success": True,
        "status": "online",
        "service": "Rainfall Forecast API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =========================================================
# OPEN-METEO FORECAST
# =========================================================

async def fetch_rainfall(lat: float, lon: float):
    """
    Fetch hourly precipitation from Open-Meteo.
    """

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "past_days": 1,
        "forecast_days": 2,
        "timezone": "UTC",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(OPEN_METEO_URL, params=params)
        response.raise_for_status()
        data = response.json()

    times = data["hourly"]["time"]
    precip_mm = data["hourly"]["precipitation"]

    now_utc = datetime.now(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    now_str = now_utc.strftime("%Y-%m-%dT%H:%M")

    observed_mm = 0.0
    forecast_mm = 0.0
    observed_hours = 0
    forecast_hours_counted = 0

    for t_str, mm in zip(times, precip_mm):

        if mm is None:
            continue

        if t_str <= now_str:
            observed_mm += mm
            observed_hours += 1

        else:
            if forecast_hours_counted < 48:
                forecast_mm += mm
                forecast_hours_counted += 1

    MM_TO_INCHES = 1 / 25.4

    return {
        "observedRainfallInches": round(observed_mm * MM_TO_INCHES, 2),
        "forecastRainfallInches": round(forecast_mm * MM_TO_INCHES, 2),
        "forecastHours": min(forecast_hours_counted, 48),
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
                    f"Observed: last "
                    f"{rainfall['observedWindowHours']}h total. "
                    f"Forecast: next "
                    f"{rainfall['forecastHours']}h total."
                ),
            },
        }

    except httpx.HTTPError as error:

        return {
            "success": False,
            "error": f"Open-Meteo request failed: {str(error)}",
            "data": None,
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error),
            "data": None,
        }


# =========================================================
# UPDATE TRIGGER
# =========================================================

@app.post("/api/trigger-update")
async def trigger_update():

    return {
        "success": True,
        "message": (
            "Update triggered. "
            "Open-Meteo data is fetched live on each request."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =========================================================
# MRMS LATEST DOWNLOAD
# =========================================================

@app.get("/api/mrms/latest")
async def mrms_latest():

    try:

        # Fetch directory listing
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
        ) as client:

            directory_response = await client.get(MRMS_BASE_URL)
            directory_response.raise_for_status()

        # Parse timestamped filenames
        pattern = re.compile(
            r'href__=\"(MRMS_' +
            re.escape(MRMS_PRODUCT) +
            r'_\d{2}\.\d{2}_\d{8}-\d{6}\.grib2\.gz)\"'
        )

        matches = pattern.findall(directory_response.text)

        if not matches:
            return {
                "success": False,
                "error": "No MRMS files found in directory listing.",
            }

        newest_filename = sorted(matches)[-1]

        source_url = MRMS_BASE_URL + newest_filename

        gz_path = f"/tmp/{newest_filename}"

        grib2_filename = newest_filename.replace(".gz", "")

        grib2_path = f"/tmp/{grib2_filename}"

        # Download file
        async with httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
        ) as client:

            download_response = await client.get(source_url)
            download_response.raise_for_status()

            with open(gz_path, "wb") as file:
                file.write(download_response.content)

        gz_size = os.path.getsize(gz_path)

        # Decompress
        with gzip.open(gz_path, "rb") as gz_in:
            with open(grib2_path, "wb") as grib_out:
                grib_out.write(gz_in.read())

        grib2_size = os.path.getsize(grib2_path)

        # Cleanup .gz
        os.remove(gz_path)

        return {
            "success": True,
            "product": MRMS_PRODUCT,
            "source_url": source_url,
            "downloaded_filename": newest_filename,
            "grib2_filename": grib2_filename,
            "file_size_bytes": grib2_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Downloaded and decompressed latest NOAA MRMS "
                "1-hour observed rainfall grid."
            ),
        }

    except httpx.HTTPError as error:

        return {
            "success": False,
            "error": f"HTTP error fetching MRMS data: {str(error)}",
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error),
        }


# =========================================================
# MRMS OVERLAY PREP
# =========================================================

@app.get("/api/mrms/overlay")
async def mrms_overlay():

    """
    Phase 3A:
    Download and verify MRMS GRIB2 file for future overlay conversion.
    """

    try:

        latest_filename = (
            f"MRMS_{MRMS_PRODUCT}.latest.grib2.gz"
        )

        download_url = f"{MRMS_BASE_URL}{latest_filename}"

        # Download file
        async with httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
        ) as client:

            response = await client.get(download_url)
            response.raise_for_status()

        gz_path = os.path.join(
            tempfile.gettempdir(),
            latest_filename,
        )

        with open(gz_path, "wb") as file:
            file.write(response.content)

        gz_size = os.path.getsize(gz_path)

        # Decompress
        grib2_filename = latest_filename.replace(".gz", "")

        grib2_path = os.path.join(
            tempfile.gettempdir(),
            grib2_filename,
        )

        with gzip.open(gz_path, "rb") as gz_in:
            with open(grib2_path, "wb") as grib_out:
                grib_out.write(gz_in.read())

        grib2_size = os.path.getsize(grib2_path)

        # Cleanup .gz
        os.remove(gz_path)

        return {
            "success": True,
            "product": MRMS_PRODUCT,
            "download_url": download_url,
            "grib2_filename": grib2_filename,
            "compressed_file_size_bytes": gz_size,
            "grib2_file_size_bytes": grib2_size,
            "grib2_file_exists": os.path.exists(grib2_path),
            "conversion_status": "pending_gdal",
            "recommended_next_step": (
                "Phase 3B: Convert GRIB2 to GeoTIFF or PNG overlay."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": (
                "GRIB2 file downloaded and verified successfully."
            ),
        }

    except httpx.HTTPStatusError as error:

        return {
            "success": False,
            "conversion_status": "failed",
            "error": (
                f"HTTP error downloading MRMS data: "
                f"{error.response.status_code}"
            ),
        }

    except httpx.RequestError as error:

        return {
            "success": False,
            "conversion_status": "failed",
            "error": f"Network error: {str(error)}",
        }

    except Exception as error:

        return {
            "success": False,
            "conversion_status": "failed",
            "error": str(error),
        }
