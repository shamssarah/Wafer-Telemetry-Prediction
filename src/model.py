import os

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from data_loader import load_data



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

class PrintShape(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"{self.name} shape: {x.shape}")
        return x
    
class WaferAutoEncoder(torch.nn.Module):
    def __init__(self,in_channels=1,
                 kernel_size=3,
                 stride=2,
                 padding=1):
        super(WaferAutoEncoder, self).__init__()


        self.enc_1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.enc_2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=kernel_size, stride=stride, padding=padding)
        self.enc_3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=kernel_size, stride=stride, padding=padding) # 4

        self.dec_1 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=padding)
        self.dec_2 = nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=padding)
        self.dec_3 = nn.ConvTranspose2d(in_channels=16, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=padding)

        self.encoder = nn.Sequential(
            self.enc_1,
            # PrintShape("After enc_1"),
            nn.ReLU(),
            self.enc_2,
            # PrintShape("After enc_2"),
            nn.ReLU(),
            self.enc_3,
            # PrintShape("After enc_3"),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            self.dec_1,
            # PrintShape("After dec_1"),
            nn.ReLU(),
            self.dec_2,
            # PrintShape("After dec_2"),
            nn.ReLU(),
            self.dec_3,
            # PrintShape("After dec_3"),
            nn.Sigmoid() # Use sigmoid to ensure output is between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
       

def train_autoencoder(model, train_loader, val_loader, criterion, optimizer, epochs):
        # training loop
    best_val_loss = np.inf
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch,_,_ in train_loader:
            # inputs = 
            outputs = model(batch)
            batch_loss = criterion(outputs, batch)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) 
            train_loss += batch_loss.item()
        # print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")
    
        # validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch,_,_ in val_loader:
                inputs = batch 
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_weights.pth")
        # torch.save(model.state_dict(), "model_weights.pth")
    return model

def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch 
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    return predictions

def reconstruction_errors(model, dataloader, device):
    model.eval()
    all_scores  = []
    all_labels  = []
    all_pixels  = []
    all_ids     = []
    all_originals = []
    with torch.no_grad():
        for batch,labels,wafer_ids in dataloader:
            imgs = batch.to(device)
            recon = model(imgs)
            pixel_error = (imgs - recon) ** 2
            wafer_score = pixel_error.mean(dim=[1, 2, 3])
            # batch_mean = wafer_score.mean().item()
            all_pixels.extend(pixel_error.cpu().numpy())
            all_scores.extend(wafer_score.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(wafer_ids.cpu().numpy())
            all_originals.extend(imgs.cpu().numpy())   # store original imgs too

    # concatenate everything across all batches into single tensors
    all_scores = np.array(all_scores)         # shape (N,)
    all_labels = np.array(all_labels)         # shape (N,)
    all_pixels = np.array(all_pixels)         # shape (N, 1, 32, 32)
    all_ids    = np.array(all_ids)            # shape (N,)
    all_originals = np.array(all_originals)   # shape (N, 1, 32, 32)

    # return it
    return all_scores, all_labels, all_pixels, all_ids, all_originals


def visualize_reconstruction_errors(clean_scores, defective_scores):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(clean_scores, bins=50, alpha=0.5, label='Clean Wafers')
    plt.hist(defective_scores, bins=50, alpha=0.5, label='Defective Wafers')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.show()# break down by defect type instead of just clean vs defective

def visualize_reconstruction_errors_by_defect_type(all_scores, all_labels):
    label_names = {0:"Center", 1:"Donut", 2:"Edge-Loc", 3:"Edge-Ring", 
                4:"Loc", 5:"Near-full", 6:"Random", 7:"Scratch", 8:"none"}

    plt.figure(figsize=(12, 6))
    for code, name in label_names.items():
        mask = all_labels == code
        if mask.sum() > 0:
            plt.hist(all_scores[mask], bins=50, alpha=0.4, label=name)

    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error by Defect Type')
    plt.legend()
    plt.show()

def visualize_heatmaps(all_pixels, all_labels, all_scores, all_originals, n_samples=5):
    import matplotlib.pyplot as plt
    
    label_names = {0:"Center", 1:"Donut", 2:"Edge-Loc", 3:"Edge-Ring", 
                   4:"Loc", 5:"Near-full", 6:"Random", 7:"Scratch", 8:"none"}
    
    # pick one example per defect type
    fig, axes = plt.subplots(len(label_names), 3, figsize=(10, len(label_names) * 3))
    # fig.suptitle("Reconstruction Error Heatmaps by Defect Type")

    for row, (code, name) in enumerate(label_names.items()):
        mask = np.where(all_labels == code)[0]
        if len(mask) == 0:
            continue
        
        # pick the sample with highest reconstruction error for that class
        idx = mask[np.argmax(all_scores[mask])]

        heatmap    = all_pixels[idx, 0]       # shape (32, 32)

        # column 1 — original wafer
        axes[row, 0].set_title("Original") if row == 0 else None
        axes[row, 0].imshow(all_originals[idx, 0], cmap='gray', interpolation='nearest')
        axes[row, 0].text(0.5, -0.15, name, transform=axes[row, 0].transAxes,
                  ha='center', fontsize=9)
        axes[row, 0].axis('off')

        # column 2 — error heatmap  ← shift everything right by one column
        axes[row, 1].set_title("Error Heatmap") if row == 0 else None
        im = axes[row, 1].imshow(heatmap, cmap='hot', interpolation='nearest')
        axes[row, 1].axis('off')
        plt.colorbar(im, ax=axes[row, 1])

        # column 3 — histogram
        axes[row, 2].set_title("Pixel Error Dist.") if row == 0 else None
        axes[row, 2].hist(heatmap.flatten(), bins=30)
        axes[row, 2].set_xlabel("MSE")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Train a wafer autoencoder.")
    argument_parser.add_argument("--train_data_path","-train", type=str, default=TRAINING_DATA_DIR, help="Path to the training data pickle file.")
    argument_parser.add_argument("--val_data_path","-val", type=str, default=VALIDATION_DATA_DIR, help="Path to the validation data pickle file.")
    argument_parser.add_argument("--test_data_path","-test", type=str, default=TEST_DATA_DIR, help="Path to the test data pickle file.")
    argument_parser.add_argument("--epochs","-e", type=int, default=10, help="Number of epochs to train the autoencoder.")
    argument_parser.add_argument("--batch_size","-b", type=int, default=32, help="Batch size for training.")
    argument_parser.add_argument("--learning_rate","-lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    args = argument_parser.parse_args()

    # arguments transformed into variables for easier access
    TRAIN_DATA_PATH = args.train_data_path
    VAL_DATA_PATH = args.val_data_path  
    TEST_DATA_PATH = args.test_data_path
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate

    # load data
    train_data = load_data(TRAIN_DATA_PATH,target_class="none")
    val_data = load_data(VAL_DATA_PATH,target_class="none")
    test_data = load_data(TEST_DATA_PATH)

    # confirm loaders are using the right datasets
    print(f"Train samples: {len(train_data)}")   # should be 22037
    print (train_data.data['failureType'].value_counts())
    print(f"Val samples:   {len(val_data)}")     # should be none only
    print (val_data.data['failureType'].value_counts())
    print(f"Test samples:  {len(test_data)}")    # should be 12450 (all classes)
    print (test_data.data['failureType'].value_counts())
    
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # initialize model, loss function, and optimizer
    model = WaferAutoEncoder().to("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop

    if os.path.exists("best_model_weights.pth"):
        model.load_state_dict(torch.load("best_model_weights.pth"))
        print("Loaded saved model weights")
    else:
        model = train_autoencoder(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

    # prediction loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_scores, all_labels, all_pixels, all_ids,all_originals = reconstruction_errors(model, test_loader, device=device)

    clean_mask = all_labels == 8
    defective_mask = all_labels != 8

    clean_scores     = all_scores[clean_mask]
    defective_scores = all_scores[defective_mask]
    clean_pixels     = all_pixels[clean_mask]
    defective_pixels = all_pixels[defective_mask]

    for file_path in [TEST_DATA_DIR]:
        data = pd.read_pickle(file_path)
        data = data.iloc[all_ids].assign(reconstruction_error=all_scores, pixel_error=all_pixels.tolist()) # add new columns to the original dataframe for later analysis
        data.to_pickle(file_path.replace(".pkl", "_with_reconstruction_errors.pkl"))

    # visualize_reconstruction_errors(clean_scores, defective_scores)
    # visualize_reconstruction_errors_by_defect_type(all_scores, all_labels)
    visualize_heatmaps(all_pixels, all_labels, all_scores, all_originals, n_samples=5)
    

    

    