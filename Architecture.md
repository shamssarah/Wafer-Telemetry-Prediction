## Architecture

```
[Python Telemetry Generator]
    Phase 1: Normal baseline (Gaussian noise around stable values)
    Phase 2: Degradation drift (sigmoidal interpolation → fault threshold)
    Phase 3: Fault state (threshold crossed → wafer defect appears)
    Pushes timestamped JSON stream per chamber
              │
              ▼
[LSTM Predictor]                              [CNN Autoencoder]
    Trained on Phase 1 normal data                Trained on defect-free wafers
    Forecasts next N readings                     Anomaly via reconstruction error
    Fires alert when trajectory                   Confirms defect has manifested
    trends toward fault threshold
              │                                          │
              └──────────────┬───────────────────────────┘
                             ▼
                     [FastAPI Backend]
                       /ingest/telemetry
                       /ingest/image
                       /status
                             │
                    Both signals aligned?
                             │
                             ▼
                       [LLM Agent]
                    Mode 1 — Predictive Alert:
                      telemetry trend flagged,
                      no wafer defect yet →
                      preventive intervention report

                    Mode 2 — Confirmed Fault:
                      telemetry + wafer image both flag →
                      full failure analysis report

                    Output: structured JSON
                    { chamber_id, fault_type, affected_parameter,
                      lead_time_ms, severity, confidence,
                      recommended_action, urgency }

```


## Tech Stack

| Layer | Technology |
|-------|------------|
| Data Generation | Python |
| ML / Vision | Python, PyTorch |
| Telemetry Prediction | Python, PyTorch (LSTM) |
| Backend | FastAPI |
| Agent | LLM API |
| Stream Emitter | C++ (stdlib) |

