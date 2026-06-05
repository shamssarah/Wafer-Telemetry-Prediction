
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import joblib
import argparse

import numpy as np
import pandas as pd

from data_loader import load_time_series_data



#        failureCode failureType
#             0         Center
#             1         Donut
#             2         Edge-Loc
#             3         Edge-Ring
#             4         Loc
#             5         Near-full
#             6         Random
#             7         Scratch
#             8         none

TEST_DATA_DIR = "../data/synthetic/test_data.pkl"
VALIDATION_DATA_DIR = "../data/synthetic/val_data.pkl"
TRAINING_DATA_DIR = "../data/synthetic/train_split.pkl"
MODEL_PATH = "../data/models/timeseries_model.pth"
FAULT_THRESHOLDS = {
    'temp':     360.0,
    'gas_flow': 130.0,
    'pressure': 3.0,
}
PHASE3_START = 160
SCALER_PATH = "../data/models/scaler.pkl"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class FaultLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, forecast_steps=10):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.input_size     = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, input_size * forecast_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.fc(out[:, -1, :])              # last timestep → (B, input_size * forecast_steps)
        return out.view(-1, self.forecast_steps, self.input_size)  # (B, forecast_steps, input_size)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader,
                num_epochs=20, lr=0.001,
                model_path="../data/models/timeseries_model.pth"):

    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    criterion    = nn.MSELoss()
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        # train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x,y = x.to(DEVICE),y.to(DEVICE)
            outputs    = model(x)
            loss       = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item()
        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x,y = x.to(DEVICE),y.to(DEVICE)
                outputs   = model(x)
                val_loss += criterion(outputs, y).item()

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss/len(train_loader):.4f}  "
              f"Val Loss:   {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  ↳ saved best model")

    return model
    

# errors that are just noise vs actual drift
ERROR_THRESHOLDS = {
    'gas_flow': 15.0,   # NOISE_STD is 4, so 15 = clearly drifting
    'pressure': 0.3,    # NOISE_STD is 0.05
    'temp':     10.0,   # NOISE_STD is 3
}

def crosses_threshold(forecast_raw, actual_raw):
    col_map = {'gas_flow': 0, 'pressure': 1, 'temp': 2}
    for sensor, col_idx in col_map.items():
        errors = np.abs(actual_raw[:, col_idx] - forecast_raw[:, col_idx])
        if np.mean(errors) > ERROR_THRESHOLDS[sensor]:  # mean over forecast steps
            return True
    return False

def predict(model, data_loader, window_size = 30, forecast_steps=10):
    model.eval()
    results= []
    scaler = joblib.load(SCALER_PATH)
    with torch.no_grad():
        for full_seq,failure_code,failure_type,wafer_id in data_loader:
            for i in range(full_seq.shape[0]):
                seq = full_seq[i]
                alert_step = None
                for t in range(len(seq) - window_size - forecast_steps + 1):
                    window   = seq[t:t + window_size]
                    actual   = seq[t + window_size:t + window_size + forecast_steps]
                    
                    forecast     = model(window.unsqueeze(0)).squeeze(0).cpu().numpy()
                    forecast_raw = scaler.inverse_transform(forecast)
                    actual_raw   = scaler.inverse_transform(actual.cpu().numpy())
                    # print (forecast_raw[:,0],actual_raw[:,0])
                    # print (crosses_threshold(forecast_raw,actual_raw))
                    if crosses_threshold(forecast_raw, actual_raw):
                        alert_step = t + window_size
                        break

                lead_time = PHASE3_START - alert_step if alert_step is not None else None

                results.append({
                    'wafer_id':     wafer_id[i].item(),
                    'failure_type': failure_type[i],
                    'alert_step':   alert_step,
                    'lead_time':    lead_time,
                    'detected':     alert_step is not None,
                })

    return pd.DataFrame(results)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Train or evaluate the LSTM model for wafer fault detection.")
    argument_parser.add_argument('--mode', choices=['train', 'predict'], required=True, help="Whether to train the model or make predictions.")
    argument_parser.add_argument('--train_path','-tr', default = TRAINING_DATA_DIR, type=str, help="Path to the telemetry data CSV file.")
    argument_parser.add_argument('--validation_path','-val', default = VALIDATION_DATA_DIR, type=str, help="Path to the validation telemetry data CSV file.")
    argument_parser.add_argument('--test_path','-te', default = TEST_DATA_DIR, type=str, help="Path to the test telemetry data CSV file.")
    argument_parser.add_argument('--model_path','-m', default = MODEL_PATH, type=str, help="Path to save the trained model or load a model for prediction.")
    argument_parser.add_argument('--epochs', '-e', type=int, default=20, help="Number of training epochs.")
    argument_parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training.")
    argument_parser.add_argument('--batch_size', '-b', type=int, default=32, help="Batch size for training.")
    args = argument_parser.parse_args()
    
    # Assigining args to variables for easier reference
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr

    MODEL_PATH = args.model_path
    TRAIN_DATA_PATH = args.train_path
    VAL_DATA_PATH = args.validation_path
    TEST_DATA_PATH = args.test_path

    if args.mode == 'train':
        # returns a PyTorch Dataset object with engineered features and labels
        train_data = load_time_series_data(args.train_path,target_class='none',sample_n=2000)
        val_data = load_time_series_data(args.validation_path,target_class='none',sample_n=500)
    

        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model, optimizer, loss function
        model = FaultLSTM().to(DEVICE) # adjust input_size based on engineered features
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        # Train the model
        print (f"Training has began...")
        train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,model_path=MODEL_PATH)
        # Save the trained model

    elif args.mode == 'predict':
        print ("Loading Model: ", MODEL_PATH)
        model = FaultLSTM().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE))
        test_data   = load_time_series_data(args.test_path, inference=True,sample_n=20)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        results_df  = predict(model, test_loader)
        print(results_df.groupby('failure_type')['lead_time'].mean())  # avg lead time per fault class
            
    