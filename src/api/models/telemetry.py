from pydantic import BaseModel
from typing import List

class SingleTelemetryRequest(BaseModel):
    chamber_id: str
    gas_flow: float
    temp:     float
    pressure: float

class TelemetryRequest (BaseModel):

    gas_flow : List [float]
    temp : List [float]
    pressure : List [float]

class TelemetryResponse (BaseModel):

    alert : bool # detected
    alert_step : int | None
    lead_time : int | None
    affected_parameter : str