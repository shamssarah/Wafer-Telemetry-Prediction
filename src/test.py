# tests/test_integration.py
from fastapi.testclient import TestClient
from api.main import app

from api.models.image import ImageResponse
from api.models.telemetry import TelemetryResponse
import time

 
client = TestClient(app)


def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
   
def test_telemetry():
    # use 200 steps of fake normal data
    payload = {
        "gas_flow": [100.0] * 200,
        "temp":     [340.0] * 200,
        "pressure": [2.0]   * 200,
    }
    response = client.post("/ingest/telemetry", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "alert"      in data
    assert "alert_step" in data
    assert "lead_time"  in data
    assert isinstance(data["alert"], bool)

def test_image():
    import numpy as np
    from PIL import Image
    import io

    # create a fake 32x32 wafer image
    img_array = np.zeros((32, 32), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    response = client.post(
        "/ingest/image",
        files={"file": ("wafer.png", buf, "image/png")}
    )
    assert response.status_code == 200

    data = response.json()
    assert "anomaly"               in data
    assert "reconstruction_error"  in data
    assert isinstance(data["anomaly"], bool)

def test_report():
    tele = TelemetryResponse(alert=True, alert_step=84, lead_time=76, affected_parameter="pressure")
    img  = ImageResponse(anomaly=True, reconstruction_error=0.73)
    
    response = client.post("/report", json={
        "telemetry_result": tele.model_dump(),
        "image_result":     img.model_dump()
    })
    assert response.status_code == 200
    
    data = response.json()
    print(data)
    assert data["mode"]     in [1, 2]
    assert data["severity"] in ["low", "medium", "high"]
    assert data["urgency"]  in ["immediate", "scheduled", "monitor"]