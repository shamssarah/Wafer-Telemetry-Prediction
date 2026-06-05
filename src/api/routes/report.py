from fastapi import APIRouter
from api.models.telemetry import TelemetryResponse
from api.models.image import ImageResponse
from api.models.report import FaultReport
from api.pipelines.report import report_pipeline

router = APIRouter()
@router.post("/report", response_model=FaultReport)
def generate_report(telemetry_result: TelemetryResponse, image_result: ImageResponse):
    result = report_pipeline(telemetry_result, image_result)
    return result