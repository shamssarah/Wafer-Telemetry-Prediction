# DigitalThread: Agentic Metrology & Root-Cause Inspector

> Dual-signal semiconductor tool health monitoring — wafer imagery + predictive telemetry → early fault detection → automated failure analysis reports

---

## Overview

In semiconductor manufacturing, a single tool failure can cost hundreds of thousands of dollars per hour in downtime. Most monitoring systems are reactive — an anomaly is only confirmed when a defective wafer surfaces at the end of a process run, long after the upstream fault began.

Physical defects on wafers are downstream consequences of chamber conditions that degraded earlier. By the time a wafer scan flags contamination or pattern failure, the process window has already been violated. Catching the fault at the wafer is too late.

DigitalThread attacks the problem upstream. It monitors two independent signals from a simulated semiconductor processing chamber: sensor telemetry (gas flow, vacuum pressure, temperature) and wafer surface imagery. An LSTM trained on normal chamber behaviour forecasts future sensor trajectories, flagging degradation trends before they cross a fault threshold. A CNN Autoencoder detects when a defect has already manifested on the wafer surface. When both signals align, an LLM agent correlates them and generates a structured failure analysis report identifying the responsible parameter and recommended intervention.

The system operates across two modes that together cover the full fault lifecycle: **predictive alert** (telemetry forecasts a fault before it occurs) and **confirmed fault** (wafer image corroborates what the telemetry predicted).

---

## Design Decisions

**Prediction, not detection.** Most anomaly detection systems tell you something went wrong. The telemetry model forecasts that something *will* go wrong and how much lead time remains before the fault threshold is crossed. Lead time is a reportable metric.

**Two signals with distinct roles.** The LSTM and the CNN Autoencoder are doing different jobs — one predicts, one confirms. The LLM agent has meaningful correlation work to do rather than routing a single flag.

**Sigmoidal degradation.** Real chamber wear is slow at first, accelerates through the middle, and plateaus at fault state. The synthetic data mirrors this rather than a linear ramp, making the prediction problem harder and the evaluation more honest.

**Rule-based fallback.** A hardcoded template would be more reliable in production. The LLM is used here as a design pattern demonstration — in a real system it would sit alongside deterministic rules as an optional enrichment layer.

---

## Lead Time Results

| Fault Type | Avg Lead Time (steps) |
|------------|----------------------|
| Loc        | 91.5                 |
| Near-full  | 87.6                 |
| Scratch    | 88.6                 |
| Donut      | 85.9                 |
| Random     | 80.5                 |
| Edge-Ring  | 78.5                 |
| Center     | 78.4                 |
| Edge-Loc   | 76.2                 |

---

## Data Sources

- **WM-811K** — wafer defect map dataset, downloaded from Kaggle
- **Synthetic telemetry** — generated via sigmoidal degradation profiles per fault class; not included in repo, generate locally

---
## Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

```python
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
```

### Data
```bash
mkdir data
```

1. Download the WM-811K dataset from [Kaggle](https://www.kaggle.com/datasets/mohammedfariskhan/wm811k-clean-subset) and save to `data/original_raw/`
2. Run the data update script — migrates to current pandas format and constructs `data/raw/`:
```bash
python scripts/update_data.py
```
3. Generate synthetic telemetry — creates all files in `data/synthetic/`:
```bash
python scripts/telemetry_generator.py
```

### Train Models
```bash
cd src
python ImageModel.py
python TimeSeriesModel.py --mode train
```

### Run Server
```bash
uvicorn api.main:app --reload
```

### C++ Emitter
See `emitter/README.md` for build instructions.
