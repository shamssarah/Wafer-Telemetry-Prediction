# =============================================================================
# TELEMETRY SYNTHESIS PARAMETERS
# =============================================================================
# All physical values are domain-informed estimates for a generic CVD/etch
# semiconductor processing chamber. No real telemetry dataset exists in the
# public domain with physical units — baselines are grounded in typical
# process ranges documented in semiconductor manufacturing literature.
# Variability (std) is informed by the UCR Wafer Dataset (2018), which
# confirmed tight steady-state variability (~5-8% of signal range) in
# normal semiconductor tool operation.
# =============================================================================

# -----------------------------------------------------------------------------
# PHASE 1 — Normal Baseline (stable chamber operation)
# -----------------------------------------------------------------------------
# Temperature : 350°C  ± 3     | typical CVD process window
# Gas Flow    : 100 sccm ± 2   | standard precursor flow rate
# Pressure    : 1.2 Torr ± 0.02| typical LPCVD chamber pressure
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# PHASE 2 — Degradation Drift
# -----------------------------------------------------------------------------
# Steps       : 100 total
# Drift window: steps 20–80 (60% of sequence)
# Transition  : sigmoidal with steepness k=0.15, midpoint t=50
# Each parameter drifts independently toward its fault-class endpoint.
# Sigmoidal chosen over linear to reflect real chamber degradation:
# slow onset → accelerating → plateau at fault state.
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

# Set the seed
random.seed(42)
# RAW_DATA_DIR = "../data/raw"
TEST_DATA_DIR = "../data/raw/test_data.pkl"
VALIDATION_DATA_DIR = "../data/raw/val_data.pkl"
TRAIN_DATA_DIR = "../data/raw/train_split.pkl"

INITIAL_TRAINING_DATA = "../data/raw/train_1_split.pkl"


BASELINE = {
    'temp':     (350, 3),    # °C
    'gas_flow': (100, 2),    # sccm
    'pressure':   (1.2, 0.02)  # torr
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

def generate_telemetry_baseline ():
    gas_flow    = np.random.normal(BASELINE['gas_flow'][0], BASELINE['gas_flow'][1] )   # sccm
    temperature = np.random.normal(BASELINE['temp'][0], BASELINE['temp'][1] )   # celsius
    pressure    = np.random.normal(BASELINE['pressure'][0], BASELINE['pressure'][1] )   # torr
    return gas_flow, temperature, pressure

def generate_telemetry_fault_data(fault):
    gas_flow = np.random.normal(FAULT_PROFILES[fault]['gas_flow'][0], FAULT_PROFILES[fault]['gas_flow'][1])    
    temperature = np.random.normal(FAULT_PROFILES[fault]['temp'][0], FAULT_PROFILES[fault]['temp'][1])
    pressure = np.random.normal(FAULT_PROFILES[fault]['pressure'][0], FAULT_PROFILES[fault]['pressure'][1])
    return gas_flow,temperature, pressure

def sigmoid_drift(t, y_start, y_end, t_mid, k):
    # Calculate the sigmoid curve and scale to [0, 1]
    sigmoid_curve = 1 / (1 + np.exp(-k * (t - t_mid)))
    # Map the curve to your specific start and end points
    return y_start + (y_end - y_start) * sigmoid_curve


def generate_synthetic_telemetry (data, file_suffix):
    synthetic_telemetry_data = []
    id_counter = 0
    for _, row in data.iterrows():

        if row.failureType == 'none':
            data_point = {
                "id": id_counter,
                "waferMap":    row.waferMap,
                "failureCode": row.failureCode,
                "failureType": row.failureType,
                "temp":        np.random.normal(350, 3),
                "gas_flow":    np.random.normal(100, 2),
                "pressure":    np.random.normal(1.2, 0.02),
                "phase":       1   # always Phase 1, no drift
            }
        else:
            s_gas, s_temp, s_pressure = generate_telemetry_baseline()
            e_gas, e_temp, e_pressure = generate_telemetry_fault_data(row.failureType)

            values = {
                "gas_flow": (s_gas, e_gas),
                "temp": (s_temp, e_temp),
                "pressure": (s_pressure, e_pressure)
            }
            t = np.linspace(0, 100, 100)
            phase = np.where(t < 20, 1, np.where(t < 80, 2, 3))


            data_point = {
                "id": id_counter,
                "waferMap": row.waferMap,
                "failureCode": row.failureCode,
                "failureType": row.failureType,
                "temp":None, 
                "gas_flow": None,
                "pressure": None,
                'phase': phase        
            }

            for key, (start_point, end_point) in values.items():    
                total_steps = 100
                steepness = 0.15          # Adjust for faster/slower transitions
                t = np.linspace(0, total_steps, total_steps)
                t_mid = total_steps / 2   # Set midpoint to halfway through the steps
                data_point[key] = sigmoid_drift(t, start_point, end_point, t_mid, steepness)

            id_counter += 1
        synthetic_telemetry_data.append(data_point)
       
    synthetic_telemetry_data_df = pandas.DataFrame(synthetic_telemetry_data)
    synthetic_telemetry_data_df.to_pickle(f"../data/synthetic/{file_suffix}")



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