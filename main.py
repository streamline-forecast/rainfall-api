from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()

# Allow Base44 frontend requests
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
        "status": "online",
        "service": "Rainfall Forecast API"
    }

@app.get("/api/forecast")
def get_forecast():

    # Replace this with real rainfall logic later
    rainfall_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "location": "San Antonio",
        "observedRainfall": 2.15,
        "forecastRainfall": 3.42,
        "forecastHours": 48
    }

    return {
        "success": True,
        "data": rainfall_data
    }