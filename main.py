from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

app = FastAPI(title="Rainfall Forecast API")

# Allow Base44 and browser testing to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/api/status")
def status():
    return {
        "success": True,
        "status": "online",
        "service": "Rainfall Forecast API",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/forecast")
def get_forecast():
    return {
        "success": True,
        "data": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "location": "San Antonio, Texas",
            "observedRainfallInches": 2.15,
            "forecastRainfallInches": 3.42,
            "forecastHours": 48,
            "source": "placeholder_test_data",
            "note": "This is test data. Replace with NOAA/MRMS/HRRR logic later."
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
