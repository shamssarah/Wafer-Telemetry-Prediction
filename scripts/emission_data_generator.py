import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_pickle("../data/synthetic/test_data.pkl")

    records = []
    for _, row in df.iterrows():
        for t in range(len(row['gas_flow'])):
            records.append({
                'gas_flow':     row['gas_flow'][t],
                'temp':         row['temp'][t],
                'pressure':     row['pressure'][t],
                'phase':        row['phase'][t],
                'failure_type': row['failureType']
            })

    pd.DataFrame(records).to_csv("../data/synthetic/telemetry_stream.csv", index=False)