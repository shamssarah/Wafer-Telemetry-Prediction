# routes/image.py
from fastapi import APIRouter
from api.models.telemetry import TelemetryRequest,TelemetryResponse,SingleTelemetryRequest
from api.pipelines.telemetry import telemetry_pipeline
import os
from collections import defaultdict 
import pandas as pd

from datetime import datetime


router = APIRouter()

counters = defaultdict(int)

LOG_DIR = "../data/streams"
ARCHIVE_DIR = "../data/streams/archive"

# create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

@router.post("/ingest/stream")
def ingest_stream(payload: SingleTelemetryRequest):
    
    log_path = f"{LOG_DIR}/{payload.chamber_id}.csv"
    
    # append reading
    with open(log_path, 'a') as f:
        f.write(f"{payload.gas_flow},{payload.temp},{payload.pressure}\n")

    counters[payload.chamber_id] += 1

       
    if counters[payload.chamber_id] >= 200:
        counters[payload.chamber_id] = 0  # reset counter
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.read_csv(log_path, names=['gas_flow', 'temp', 'pressure'])
        os.rename(log_path, f"{ARCHIVE_DIR}/{payload.chamber_id}_{timestamp}.csv")
        result = telemetry_pipeline(TelemetryRequest(
            gas_flow=df['gas_flow'].tolist(),
            temp=df['temp'].tolist(),
            pressure=df['pressure'].tolist()
        ))
        return {"status": "inference_run", "result": result}
    
    return {"status": "received", "count": counters[payload.chamber_id]}

@router.post("/ingest/telemetry", response_model=TelemetryResponse)
async def ingest_telemetry(file: TelemetryRequest):
    result = telemetry_pipeline(file)
    return result