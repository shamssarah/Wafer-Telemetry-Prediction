import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import ast

SCALER_PATH = "../data/models/scaler.pkl"

FEATURE_COLS = [
    'gas_flow', 'pressure', 'temp',
    'gas_flow_rolling_mean', 'gas_flow_rolling_std', 'gas_flow_delta',
    'pressure_rolling_mean', 'pressure_rolling_std', 'pressure_delta',
    'temp_rolling_mean',     'temp_rolling_std',     'temp_delta',
]

class SmartResizePad:
    def __init__(self, target_size=32, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):

        width, height = img.size # 25x27, 27x25, 26x26
        max_dim = max(width, height) 
        if max_dim > self.target_size:
            img = F.resize(img, (self.target_size, self.target_size))
            width, height = img.size # 32x32
    
        # Calculate the padding needed to reach the target size
        pad_w = self.target_size - width
        pad_h = self.target_size - height
        
        # Split the padding equally on both sides
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        return img
    
def feature_engineering(df, scaler=False, fit=False):
    window = 10
    sensor_cols = ['gas_flow', 'pressure', 'temp']

    for col in sensor_cols:
        df[f'{col}_rolling_mean'] = df[col].apply(lambda x: pd.Series(x).rolling(window).mean().to_numpy())
        df[f'{col}_rolling_std']  = df[col].apply(lambda x: pd.Series(x).rolling(window).std().to_numpy())
        df[f'{col}_delta']        = df[col].apply(lambda x: pd.Series(x).diff().to_numpy())


    # build flat (n_wafers * seq_len, n_features) array for scaler
    rows = []
    for i in range(len(df)):
        seq = np.stack([np.array(df.iloc[i][col]) for col in FEATURE_COLS], axis=1)
        seq = seq[~np.isnan(seq).any(axis=1)]
        rows.append(seq)
    flat = np.vstack(rows)  # (n_wafers * seq_len, 12)


    if fit:
        scaler = StandardScaler()
        scaler.fit(flat)
        joblib.dump(scaler, "../data/models/scaler.pkl")

    # scale and put back into df as lists
    for i in range(len(df)):
        seq = np.stack([np.array(df.iloc[i][col]) for col in FEATURE_COLS], axis=1)
        seq = seq[~np.isnan(seq).any(axis=1)]
        scaled = scaler.transform(seq)  # (seq_len, 12)
        for j, col in enumerate(FEATURE_COLS):
            df.at[df.index[i], col] = scaled[:, j]

    return df, scaler

class WaferDataset(torch.utils.data.Dataset):
    def __init__(self,dataframe,target_class="none",transform=True):

        if target_class:
            self.data = dataframe[dataframe.failureType == target_class].reset_index(drop=True)
        else:
            self.data = dataframe.reset_index(drop=True)
        
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        wafer_map = self.data.iloc[idx].waferMap
        label = self.data.iloc[idx].failureCode
        wafer_id = self.data.iloc[idx].id  # or whatever your ID column is called
        
        img = Image.fromarray(wafer_map, mode='L') # Convert to PIL Image
        if self.transform:
            img = self.transform(img)
        else:
            # Safe fallback if you ever instantiate it without transforms
            img = torch.tensor(wafer_map, dtype=torch.float32).unsqueeze(0)
        return img,label,wafer_id
    
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, window_size=30, forecast_steps=10,
                 target_class='none',  inference=False):
        
        if target_class:
            self.data = dataframe[dataframe.failureType == target_class].reset_index(drop=True)
        else:
            self.data = dataframe.reset_index(drop=True)

        self.window_size    = window_size
        self.forecast_steps = forecast_steps
        self.inference      = inference

        # pre-build windows at init time so __getitem__ is just a lookup
        self.windows = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            seq = np.stack([np.array(row[col]) for col in FEATURE_COLS], axis=1)
            seq = seq[~np.isnan(seq).any(axis=1)]

            if inference:
                # return full sequence + metadata
                self.windows.append((seq, row['failureCode'], row['failureType'], row['id']))
            else:
                # sliding window (x, y) pairs
                for i in range(len(seq) - window_size - forecast_steps + 1):
                    x = seq[i            : i + window_size]
                    y = seq[i + window_size : i + window_size + forecast_steps]
                    self.windows.append((x, y))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if self.inference:
            seq, code, ftype, wid = self.windows[idx]
            return torch.tensor(seq, dtype=torch.float32), code, ftype, wid
        else:
            x, y = self.windows[idx]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_image_data(data_path,target_class=None):

    with open (data_path,"rb") as file:
        data = pd.read_pickle(file)


    preprocess = transforms.Compose([
        SmartResizePad(target_size=32, fill=0), # Your custom conditional logic
        transforms.ToTensor(),                  # Convert to tensor
        transforms.Normalize(mean=[0.0], std=[1.0]) # Add any other standard transforms
    ])

    dataset = WaferDataset(
            dataframe=data,
            target_class=target_class,
            transform=preprocess
        )
    return dataset
   
    
def load_time_series_data(data_path, target_class=None, inference=False, sample_n=None, fit_scaler=False):
    with open (data_path,"rb") as file:
        data = pd.read_pickle(file)

    if target_class and sample_n:
        pool = data[data.failureType == target_class]
        data = pool.sample(n=min(sample_n, len(pool)), random_state=42)
    
    scaler = joblib.load(SCALER_PATH) if not fit_scaler else None
    data, scaler = feature_engineering(data, scaler=scaler, fit=fit_scaler)  # unpack tuple

    dataset = TimeSeriesDataset(
        dataframe=data,
        target_class=target_class,
        inference=inference
    )
    return dataset