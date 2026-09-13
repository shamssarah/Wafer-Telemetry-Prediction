# =============================================================================
# TELEMETRY SYNTHESIS PARAMETERS
# =============================================================================
# Phase 3 fault endpoints are domain-informed estimates derived from
# semiconductor process engineering literature and physical reasoning:
# each defect morphology (Center, Edge-Ring, Donut, etc.) is associated
# with known root causes (gas flow imbalance, temperature gradients,
# pressure events) and the parameter deviations reflect those causal
# relationships.
#
# UCR Wafer Dataset informed Phase 1 std values only — it does not
# contain per-defect telemetry or physical units and was not used
# for fault profile generation.
# -----------------------------
# =============================================================================

# -----------------------------------------------------------------------------
# PHASE 1 — Normal Baseline (stable chamber operation) - Extracted from UCR Wafer Dataset
# -----------------------------------------------------------------------------
# Temperature : 350°C  ± 3     | typical CVD process window
# Gas Flow    : 100 sccm ± 2   | standard precursor flow rate
# Pressure    : 1.2 Torr ± 0.02| typical LPCVD chamber pressure
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# PHASE 2 — Degradation Drift
# -----------------------------------------------------------------------------
# Steps       : 250 total
# Drift window: 40-60 steps 
# Transition  : sigmoidal with steepness k=0.15, midpoint t=50
# Each parameter drifts independently toward its fault-class endpoint.
# Sigmoidal chosen over linear to reflect real chamber degradation:
# slow onset → accelerating → plateau at fault state.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# PHASE 2.5 — Adding Noise - Leading up to Drift
# -----------------------------------------------------------------------------
# Steps       : 250 total
# Drift window: 10-30 steps
# Transition  : 

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# PHASE 3 — Fault State endpoints per defect class
# Defect class → likely physical cause → parameter deviation
# -----------------------------------------------------------------------------
# Center    → gas over-supply to wafer center
#             temp: 365°C, gas_flow: 150 sccm, pressure: 1.8 Torr
#
# Edge-Ring → edge cooling effect, temperature drop at periphery
#             temp: 325°C, gas_flow: 90 sccm,  pressure: 2.2 Torr
#
# Edge-Loc  → localised edge temperature drop
#             temp: 330°C, gas_flow: 92 sccm,  pressure: 2.0 Torr
#
# Donut     → insufficient gas flow, symmetric ring pattern
#             temp: 345°C, gas_flow: 70 sccm,  pressure: 3.5 Torr
#
# Loc       → localised pressure burst / contamination event
#             temp: 350°C, gas_flow: 100 sccm, pressure: 6.0 Torr
#
# Random    → general process instability, elevated variance on all params
#             temp: 345°C, gas_flow: 105 sccm, pressure: 2.5 Torr
#
# Scratch   → mechanical event, vacuum spike
#             temp: 350°C, gas_flow: 100 sccm, pressure: 4.5 Torr
#
# Near-full → severe process deviation, large temperature drop
#             temp: 290°C, gas_flow: 80 sccm,  pressure: 4.0 Torr
# -----------------------------------------------------------------------------


import pandas
import random
import os
import gc
import numpy as np

import numpy as np
import pandas as pd

TEST_DATA_DIR = "../data/raw/test_data.pkl"
VALIDATION_DATA_DIR = "../data/raw/val_data.pkl"
TRAIN_DATA_DIR = "../data/raw/train_split.pkl"

INITIAL_TRAINING_DATA = "../data/raw/train_1_split.pkl"


TOTAL_LEN = 200
SLOPE_MIN, SLOPE_MAX = 50, 70       # how many steps the drift takes to complete
TAIL_BUFFER = 20                    # steps left after the ramp, so there's a noisy "past threshold" tail
ONSET_MIN = 50                      # earliest step the drift can start
LEAD_MIN, LEAD_MAX = 15, 30         # how many steps before onset the noise starts climbing

BASELINE = {
    'temp':     (350, 3),    # °C
    'gas_flow': (100, 2),    # sccm
    'pressure': (1.2, 0.02)  # torr
}
NOISE_STD = {
    'temp':     3.0,
    'gas_flow': 4.0,
    'pressure': 0.05,
}

FAULT_PROFILES = {
    'Center':    {'temp': (365, 5),  'gas_flow': (150, 6),  'pressure': (1.8, 0.1)},
    'Edge-Ring': {'temp': (325, 4),  'gas_flow': (90,  3),  'pressure': (2.2, 0.2)},
    'Edge-Loc':  {'temp': (330, 5),  'gas_flow': (92,  3),  'pressure': (2.0, 0.2)},
    'Donut':     {'temp': (345, 4),  'gas_flow': (70,  3),  'pressure': (3.5, 0.3)},
    'Loc':       {'temp': (350, 4),  'gas_flow': (100, 2),  'pressure': (6.0, 0.5)},
    'Random':    {'temp': (345, 10), 'gas_flow': (105, 9),  'pressure': (2.5, 0.4)},
    'Scratch':   {'temp': (350, 3),  'gas_flow': (100, 2),  'pressure': (4.5, 0.4)},
    'Near-full': {'temp': (290, 6),  'gas_flow': (80,  5),  'pressure': (4.0, 0.3)},
}


def generate_telemetry_baseline(rng):
    gas_flow    = rng.normal(*BASELINE['gas_flow'])
    temperature = rng.normal(*BASELINE['temp'])
    pressure    = rng.normal(*BASELINE['pressure'])
    return gas_flow, temperature, pressure


def generate_telemetry_fault_data(fault, rng):
    profile = FAULT_PROFILES[fault]
    gas_flow    = rng.normal(*profile['gas_flow'])
    temperature = rng.normal(*profile['temp'])
    pressure    = rng.normal(*profile['pressure'])
    return gas_flow, temperature, pressure


def sigmoid_progress(t, t_mid, k):
    """0 -> 1 smooth curve, centered at t_mid, steepness k."""
    return 1 / (1 + np.exp(-k * (t - t_mid)))

def sigmoid_drift(t, y_start, y_end, t_mid, k):
    curve = sigmoid_progress(t, t_mid, k)
    return y_start + (y_end - y_start) * curve


def build_synthetic_dataset(data, file_suffix, seed=0):
    rng = np.random.default_rng(seed)  # private random state -> reproducible with a fixed seed
    synthetic_telemetry_data = []
    t = np.arange(TOTAL_LEN)

    for id_counter, (_, row) in enumerate(data.iterrows()):  # fixes: id_counter now increments for every row
        s_gas, s_temp, s_pressure = generate_telemetry_baseline(rng)

        if row.failureType == 'none':
            data_point = {
                "id": id_counter,
                "waferMap":    row.waferMap,
                "failureCode": row.failureCode,
                "failureType": row.failureType,
                "temp":        rng.normal(s_temp, NOISE_STD['temp'], TOTAL_LEN),
                "gas_flow":    rng.normal(s_gas, NOISE_STD['gas_flow'], TOTAL_LEN),
                "pressure":    rng.normal(s_pressure, NOISE_STD['pressure'], TOTAL_LEN),
                "onset": None, "slope": None, "lead": None,
            }
        else:
            e_gas, e_temp, e_pressure = generate_telemetry_fault_data(row.failureType, rng)

            # --- randomized timing, per sample ---
            # -----------------------------------------------------------------------------------------------------------------------
            # To prevent the forecasting model from memorizing the drift pattern
            # The TOTAL_LEN has increased from 100 -> 200 to allow a greater range of randomization
            # of where the drift occurs. The length of the drift has also been randomized to prevent the memorization.
            # Also lead time was added - the flunctuattion of the sensor reading before thye completely degrade.
            # Assumption were made as real telemetry data is not publicly available, hence the patterns present (noise assumption) 
            # represented in the synthetic data may not exist in real telemetry data. 
            # However, this is a proof of concept, that having two models can reduce risk of achieving defective wafers.
            # -----------------------------------------------------------------------------------------------------------------------
            slope = int(rng.integers(SLOPE_MIN, SLOPE_MAX + 1)) # how big/long is the drift
            onset_max = TOTAL_LEN - slope - TAIL_BUFFER # When the drift should end by
            onset = int(rng.integers(ONSET_MIN, onset_max)) # When the drift begins
            lead = int(rng.integers(LEAD_MIN, LEAD_MAX + 1)) #


            t_mid = onset + slope / 2
            k = 8 / slope

            noise_t_mid = onset - lead / 2
            noise_k = 8 / lead
            noise_progress = sigmoid_progress(t, noise_t_mid, noise_k)
            noise_scale = 1.0 + noise_progress  # 1x calm -> 2x noisy

            values = {
                "gas_flow": (s_gas, e_gas),
                "temp": (s_temp, e_temp),
                "pressure": (s_pressure, e_pressure),
            }

            data_point = {
                "id": id_counter,
                "waferMap":    row.waferMap,
                "failureCode": row.failureCode,
                "failureType": row.failureType,
                "onset": onset, "slope": slope, "lead": lead,  # ground truth for later lead-time eval
            }

            for key, (start_point, end_point) in values.items():
                base_drift = sigmoid_drift(t, start_point, end_point, t_mid, k)
                noise = rng.normal(0, NOISE_STD[key], size=TOTAL_LEN) * noise_scale
                data_point[key] = base_drift + noise

        synthetic_telemetry_data.append(data_point)

    synthetic_telemetry_data_df = pd.DataFrame(synthetic_telemetry_data)
    synthetic_telemetry_data_df.to_pickle(f"../data/synthetic/{file_suffix}")
    return synthetic_telemetry_data_df


if __name__ == "__main__":
    # Load the training data
    with open(INITIAL_TRAINING_DATA, 'rb') as file:
        data = pandas.read_pickle(file)

    for file_path in [TRAIN_DATA_DIR, VALIDATION_DATA_DIR, TEST_DATA_DIR]:
        with open(file_path, 'rb') as file:
            data = pandas.read_pickle(file)
        generate_synthetic_telemetry(data, os.path.basename(file_path))
        del data
        gc.collect()
        print (f"Synthetic telemetry data generated and saved to ../data/synthetic/{os.path.basename(file_path)}")