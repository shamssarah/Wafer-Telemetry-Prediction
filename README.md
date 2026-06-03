# DigitalThread: Agentic Metrology & Root-Cause Inspector

> Dual-signal semiconductor tool health monitoring — wafer imagery + predictive telemetry → early fault detection → automated failure analysis reports

---

## Overview

In semiconductor manufacturing, a single tool failure can cost hundreds of thousands of dollars per hour in downtime. Most monitoring systems are reactive — an anomaly is only confirmed when a defective wafer surfaces at the end of a process run, long after the upstream fault began.

The core challenge is that physical defects on wafers are downstream consequences of chamber conditions that degraded earlier. By the time a wafer scan flags contamination or pattern failure, the process window has already been violated. Catching the fault at the wafer is too late.

DigitalThread attacks the problem upstream. It monitors two independent streams from a simulated semiconductor processing chamber: sensor telemetry (gas flow, vacuum pressure, temperature) and wafer surface imagery. On the telemetry side, an LSTM trained on normal chamber behaviour forecasts future sensor trajectories — flagging when a degradation trend is heading toward a fault threshold before it gets there. On the imagery side, a CNN Autoencoder detects when a defect has already manifested on the wafer surface. When the predicted trajectory and the visual confirmation align, an LLM agent correlates both signals and generates a structured failure analysis report identifying the specific parameter responsible and the recommended intervention.

The system operates across two modes that together cover the full fault lifecycle: **predictive alert** (telemetry forecasts a fault before it occurs) and **confirmed fault** (wafer image corroborates what the telemetry predicted).

---
## Data Sources
- WM-811K: downloaded manually from Kaggle (API unavailable)
- UCR Wafer: downloaded manually from UCR Time Series Archive

---

## What I am trying to achieve

**Prediction, not just detection.** Most anomaly detection systems tell you something went wrong. The telemetry model here forecasts that something *will* go wrong and how much lead time remains before the fault threshold is crossed. Lead time is a reportable metric.

**Two signals with distinct roles.** The LSTM and the CNN Autoencoder are doing different jobs — one predicts, one confirms. The agent has meaningful work to do correlating them rather than just routing a single flag.

**Sigmoidal degradation is physically realistic.** Real chamber degradation is slow at first, accelerates through the middle, and plateaus at the fault state. The synthetic data mirrors this rather than using a simple linear ramp, which makes the LSTM's prediction problem harder and the evaluation more honest.

---
