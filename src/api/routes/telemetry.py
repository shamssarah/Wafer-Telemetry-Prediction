# routes/image.py
from fastapi import APIRouter
from api.models.telemetry import TelemetryRequest,TelemetryResponse
from api.pipelines.telemetry import telemetry_pipeline

router = APIRouter()

@router.post("/ingest/telemetry", response_model=TelemetryResponse)
async def ingest_telemetry(file: TelemetryRequest):
    result = telemetry_pipeline(file)
    return result