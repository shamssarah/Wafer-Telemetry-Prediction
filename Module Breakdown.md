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
                             │
                             ▼
                    [Docker + GCP Cloud Run]
                      Benchmarked under concurrent streams
```

---

## Modules

### Module 1 — Data Foundation
> Define and generate both data streams. The synthetic telemetry must be temporally structured so the LSTM has something meaningful to learn.

- [X] Source WM-811K wafer map dataset and inspect class distribution
- [X] Define telemetry schema: `gas_flow_sccm`, `vacuum_pressure_torr`, `temperature_c`, `timestamp`, `chamber_id`, `phase_label`
- [X] Implement three-phase synthetic generator:
  - Phase 1: Gaussian noise around stable baseline per parameter
  - Phase 2: Sigmoidal drift from baseline toward fault threshold, independent onset per parameter, noise added
  - Phase 3: Values at or beyond fault threshold
- [X] Label phase boundaries in output — needed to evaluate LSTM lead time
- [X] Correlate Phase 3 timestamps with defective wafer images for Module 2
- [X] Document baseline values, fault thresholds, and degradation parameters

---


### Module 2 — CNN Autoencoder (Wafer Anomaly Detection)
> Confirm faults visually. Trained on defect-free wafers, detects anomalies via reconstruction error at Phase 3 timestamps.

- [X] Filter WM-811K to defect-free wafers only for training
- [X] Build convolutional autoencoder in PyTorch (encoder → bottleneck → decoder)
- [X] Train and validate reconstruction error on held-out clean wafers
- [X] Evaluate anomaly signal: reconstruction error on clean vs defective wafers
- [X] Align defective wafer samples to Phase 3 timestamps from Module 1
- [X] Visualise reconstruction error heatmaps for spatial fault localisation
- [ ] *(Stretch)* Port to VAE for probabilistic anomaly scoring

---

### Module 3 — LSTM Telemetry Predictor
> Predict faults before they occur. Trained on Phase 1 normal data, forecasts trajectory and fires alert when heading toward fault threshold.

- [ ] Engineer time-series features: rolling mean, rolling std, delta per parameter
- [ ] Train LSTM on Phase 1 normal telemetry sequences
- [ ] Implement forecasting: predict next N readings from current window
- [ ] Define alert trigger: forecast trajectory crosses threshold within time horizon
- [ ] Evaluate lead time: how many timesteps before Phase 3 does the alert fire?
- [ ] Document lead time as a key project metric

---

### Module 4 — FastAPI Backend
> Unified ingestion API wiring both models.

- [ ] Set up FastAPI project structure
- [ ] `POST /ingest/telemetry` — accepts sensor payload, returns forecast + alert flag + estimated lead time
- [ ] `POST /ingest/image` — accepts wafer image, returns reconstruction error + anomaly flag
- [ ] `GET /status` — health check
- [ ] Integration tests for both endpoints

---

### Module 5 — LLM Agent & Structured Report
> Correlates signals and produces structured output across both modes.

- [ ] Define report JSON schema: `chamber_id`, `fault_type`, `affected_parameter`, `lead_time_ms`, `severity`, `confidence`, `recommended_action`, `urgency`, `mode`
- [ ] Mode 1 trigger: telemetry alert fires, no wafer flag yet → preventive report
- [ ] Mode 2 trigger: telemetry + wafer image both flag → confirmed fault report
- [ ] Engineer system prompt with schema, fault taxonomy, and mode distinction
- [ ] Validate output conforms to schema across fault scenarios
- [ ] *(Stretch)* Multi-turn agent requests additional telemetry context before filing

---

### Module 6 — C++ Stream Emitter
> Lightweight edge layer that reads pre-generated telemetry and emits it as a high-frequency stream to the FastAPI backend, simulating real sensor hardware.

- [ ] Read pre-generated telemetry file (Phase 1 → 2 → 3 sequence)
- [ ] Emit records row-by-row at configurable frequency via HTTP POST
- [ ] Output matches telemetry schema exactly
- [ ] Configurable emission rate and chamber ID
- [ ] Document build instructions

> **Note:** C++ depth is demonstrated separately via the interpreter project. This component exists to show the hardware-software boundary and is intentionally scoped small.

---

### Module 7 — Integration, Docker & Write-Up
> Containerise, benchmark, and produce the portfolio artifact.

- [ ] Dockerize FastAPI service; confirm C++ emitter connects end-to-end
- [ ] Deploy to GCP Cloud Run
- [ ] Benchmark concurrent telemetry streams; document latency
- [ ] Write architecture diagram
- [ ] Document design decisions: prediction vs detection, sigmoidal degradation, autoencoder vs U-Net, agent dual-mode design

---

## Timeline

| Module | Target |
|--------|--------|
| 1 — Data Foundation | ✅ Done |
| 2 — CNN Autoencoder | ✅ Done |
| 3 — LSTM Predictor | Week of July 7 |
| 4 — FastAPI Backend | Week of July 21 |
| 5 — LLM Agent | Week of August 4 |
| 6 — C++ Emitter | Week of August 18 |
| 7 — Integration & Write-Up | Week of August 25 |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Data Generation | Python |
| ML / Vision | Python, PyTorch |
| Telemetry Prediction | Python, PyTorch (LSTM) |
| Backend | FastAPI |
| Agent | LLM API |
| Stream Emitter | C++ (stdlib) |
| Containerisation | Docker |
| Cloud | GCP Cloud Run |
| Benchmarking | Locust or k6 |

---

## Dataset Sources

- **WM-811K** — 811,457 wafer maps with labelled defect patterns
- **Synthetic telemetry** — three-phase procedurally generated sensor streams with sigmoidal degradation and injected fault scenarios

---

## Design Decisions

**Why prediction on telemetry rather than anomaly detection?**
Anomaly detection on sensor readings tells you the chamber is already outside normal range — the fault has already begun. An LSTM forecasting the trajectory can fire an alert while readings are still within bounds, providing lead time before any wafer is affected. That lead time is the entire value proposition.

**Why sigmoidal degradation rather than linear interpolation?**
Real chamber degradation is slow at first, accelerates through the middle, and levels off at the fault state. A linear ramp is easier to predict and doesn't reflect physical reality. Sigmoidal drift makes the prediction problem harder and the reported lead time more meaningful.

**Why a CNN Autoencoder and not a classifier?**
Labelled defect data in real fabs is sparse and proprietary. An autoencoder trained on defect-free wafers learns a compressed representation of normality — it struggles to reconstruct anomalous patterns at inference, and that reconstruction error is the anomaly signal. No labelled defects required.

**Why not U-Net?**
U-Net skip connections make reconstruction too easy even for anomalous inputs, suppressing the reconstruction error signal. U-Net is right for pixel-level segmentation with labelled masks; wrong here.
