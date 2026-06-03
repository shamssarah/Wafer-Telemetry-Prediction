import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd

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
    


def load_data(data_path,target_class=None):
    with open(data_path, 'rb') as file:    
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
   