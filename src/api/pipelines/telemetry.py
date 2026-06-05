import pandas as pd
import joblib
import torch
import numpy as np
import sys
sys.path.append('src')

from data_loader import feature_engineering,SCALER_PATH,FEATURE_COLS
from TimeSeriesModel import FaultLSTM,MODEL_PATH,crosses_threshold,PHASE3_START

from api.models.telemetry import TelemetryRequest

def loadModel(device):
    model  = FaultLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model.eval()  # set to eval mode here so you don't have to elsewhere
    return model

def evaluate (model,input,scaler, window_size=30, forecast_steps=10):
    alert_step = None
    with torch.no_grad():
        for t in range(len(input) - window_size - forecast_steps + 1):
            window = input [t : t + window_size].unsqueeze(0)
            forecast = model(window).squeeze(0).cpu().numpy()
            actual = input [t + window_size : t + window_size + forecast_steps].numpy()

            forecast_raw = scaler.inverse_transform (forecast)
            actual_raw = scaler.inverse_transform (actual)

            if crosses_threshold (forecast_raw,actual_raw):
                alert_step = t + window_size
                break
            
    lead_time = PHASE3_START - alert_step if alert_step is not None else None
    return {
        "alert":      alert_step is not None,
        "alert_step": alert_step,
        "lead_time":  lead_time,
    }
        

def telemetry_pipeline (payload : TelemetryRequest):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.DataFrame([{
        'gas_flow': payload.gas_flow,
        'temp':     payload.temp,
        'pressure': payload.pressure,
    }])
    
    scaler = joblib.load(SCALER_PATH)
    df, _  = feature_engineering(df, scaler=scaler, fit=False)

    model = loadModel(device=device)

    seq = np.stack([df.iloc[0][col] for col in FEATURE_COLS],axis=1)
    seq = seq [~np.isnan(seq).any(axis=1)]
    seq = torch.tensor(seq,dtype=torch.float32)

    result = evaluate (model,seq,scaler)
    return result




