from pydantic import BaseModel
from typing import List

class TelemetryRequest (BaseModel):

    gas_flow : List [float]
    temp : List [float]
    pressure : List [float]

class TelemetryResponse (BaseModel):

    alert : bool # detected
    alert_step : int | None
    lead_time : int | None