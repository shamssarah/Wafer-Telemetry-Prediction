# main.py
from fastapi import FastAPI
from api.routes import telemetry, image, status, report

app = FastAPI()

app.include_router(telemetry.router)
app.include_router(image.router)
app.include_router(status.router)
app.include_router(report.router)