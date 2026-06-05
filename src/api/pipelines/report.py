from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import os
import json
from api.models.report import FaultReport

load_dotenv()

def get_model():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel("gemini-2.5-flash")

# TEMPLATE FOR REPORT GENERATION
def rule_based_report(alert, anomaly, lead_time, affected_parameter):
    if alert and anomaly:
        return {"severity": "high", "urgency": "immediate", 
                "recommended_action": "Halt chamber and inspect"}
    elif alert and not anomaly:
        return {"severity": "medium", "urgency": "scheduled",
                "recommended_action": "Schedule inspection within 24hrs"}
    


def report_pipeline(telemetry_response, image_response):
    model  = get_model()
    mode = 2 if telemetry_response.alert and image_response.anomaly else 1
    
    prompt = f"""
    Generate a fault report as JSON only. No prose.
    
    Sensor data: alert={telemetry_response.alert}, affected_parameter={telemetry_response.affected_parameter}, lead_time={telemetry_response.lead_time}
    Wafer data:  anomaly={image_response.anomaly}, reconstruction_error={image_response.reconstruction_error}
    Mode: {mode} (1=preventive, 2=confirmed fault)
    
    Return only this JSON:
    {{
        "fault_type": <infer from signals>,
        "affected_parameter": "{telemetry_response.affected_parameter}",
        "lead_time_ms" : "{telemetry_response.lead_time}"
        "severity": <medium if mode 1, high if mode 2>,
        "confidence": <0.6 if mode 1, 0.9 if mode 2>,
        "recommended_action": <based on severity>,
        "urgency": <scheduled if mode 1, immediate if mode 2>,
        "mode": {mode}
    }}
    """
    response = model.generate_content(prompt)
    raw = response.text.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    return FaultReport(**json.loads(clean))
        
