import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import ast
from skimage.measure import block_reduce

import math

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

        width, height = img.size
        max_dim = max(width, height)
        if max_dim > self.target_size:
            block_h = max(1, math.ceil(height / self.target_size))
            block_w = max(1, math.ceil(width / self.target_size))
            img_arr = self.downsize_binary_preserve_foreground(np.array(img), block_h=block_h, block_w=block_w)
            img = F.to_pil_image(img_arr)
            width, height = img.size
            # fall through to padding below, don't return early
        else:
            scale = self.target_size / max_dim
            new_w, new_h = round(width * scale), round(height * scale)
            img = F.resize(img, (new_h, new_w), interpolation=F.InterpolationMode.NEAREST)
            width, height = new_w, new_h
        # Pad the shorter dimension to reach target_size
        pad_w = self.target_size - width
        pad_h = self.target_size - height
        
        # Split the padding equally on both sides
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        return img
    
    def downsize_binary_preserve_foreground(self,img_array, block_h, block_w):
        """
        Downsize a binary image while preserving foreground pixels.
        If any pixel in the block is 1, the resulting pixel will be 1.
        """
        return block_reduce(img_array, block_size=(block_h, block_w), func=np.max)


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

    def __init__(self, dataframe, include_classes=None, label_encoder=None, transform=True):
        if include_classes:
            self.data = dataframe[dataframe.failureType.isin(include_classes)].reset_index(drop=True)
        else:
            self.data = dataframe.reset_index(drop=True)
        self.label_encoder = label_encoder
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
                 include_classes=None,  inference=False):
        
        if include_classes:
            self.data = dataframe[dataframe.failureType.isin(include_classes)].reset_index(drop=True)
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


def load_image_data(data_path,include_classes=None):

    with open (data_path,"rb") as file:
        data = pd.read_pickle(file)


    preprocess = transforms.Compose([
        SmartResizePad(target_size=32, fill=0), # Your custom conditional logic
        transforms.ToTensor(),                  # Convert to tensor
        transforms.Normalize(mean=[0.0], std=[1.0]) # Add any other standard transforms
    ])

    dataset = WaferDataset(
            dataframe=data,
            include_classes=include_classes,
            transform=preprocess
        )
    return dataset





   
    
def load_time_series_data(data_path, include_classes=None, inference=False, sample_n=None, fit_scaler=False):
    with open (data_path,"rb") as file:
        data = pd.read_pickle(file)

    if include_classes and sample_n:
        pool = data[data.failureType.isin(include_classes)]
        data = pool.sample(n=min(sample_n, len(pool)), random_state=42)
    
    scaler = joblib.load(SCALER_PATH) if not fit_scaler else None
    data, scaler = feature_engineering(data, scaler=scaler, fit=fit_scaler)  # unpack tuple

    dataset = TimeSeriesDataset(
        dataframe=data,
        include_classes=include_classes,
        inference=inference
    )
    return dataset