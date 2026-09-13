import os

import torch
from torchinfo import summary
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from data_loader import  load_image_data

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,f1_score



TEST_DATA_DIR = "../data/synthetic/test_data.pkl"
VALIDATION_DATA_DIR = "../data/synthetic/val_data.pkl"
TRAINING_DATA_DIR = "../data/synthetic/train_split.pkl"
MODEL_PATH = "../data/models/basic_cnn_weights.pth"


NO_CLASS = 8

class PrintShape(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"{self.name} shape: {x.shape}")
        return x


def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)  # only this halves spatial size
    )
class WaferCNN(torch.nn.Module):
    def __init__(self,in_channels=1,
                 kernel_size=3,
                 stride=2,
                 padding=1):
        super(WaferCNN, self).__init__()

        self.block1 = conv_block(1, 16)   # 1X32x32 -> 16x16X16, 16 channels (32 + 2 - 3) / 1 +1 = 32/2 = 16
        self.block2 = conv_block(16, 8)   # 16x16X16 -> 8x8,  8 channels (16+2-3)/1 + 1 = 16/2 = 8
        self.block3 = conv_block(8, 4)    # 8x8   -> 4x4,  4 channels (8+2-3)/1 + 1 = 8/2 = 4
        self.flatten = nn.Flatten()
        self.fc  = nn.Linear(4 * 4 * 4 , NO_CLASS)  # Adjust the input size based on your image dimensions

        self.model = nn.Sequential(
            self.block1,
            # PrintShape("after block1"),
            self.block2,
            # PrintShape("after block2"),
            self.block3,
            # PrintShape("after block3"),
            self.flatten,
            self.fc,
        )


    def forward(self, x):
        output = self.model(x)
        return output
       

def train_classifier(model, train_loader, val_loader, criterion, optimizer, epochs):
        # training loop
    best_val_loss = np.inf
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels,_ in train_loader:
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) 
            train_loss += batch_loss.item()
        # print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")
    
        # validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels,_ in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
    return model

def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _,_ in dataloader:
            outputs = model(images)
            predictions.append(outputs.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions, axis=0)

def confusion_matrix_plot (all_labels, all_predictions, label_names):
  
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_names.values()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Train a wafer classification model.")
    argument_parser.add_argument("--train_data_path","-train", type=str, default=TRAINING_DATA_DIR, help="Path to the training data pickle file.")
    argument_parser.add_argument("--val_data_path","-val", type=str, default=VALIDATION_DATA_DIR, help="Path to the validation data pickle file.")
    argument_parser.add_argument("--test_data_path","-test", type=str, default=TEST_DATA_DIR, help="Path to the test data pickle file.")
    argument_parser.add_argument("--epochs","-e", type=int, default=10, help="Number of epochs to train the model.")
    argument_parser.add_argument("--batch_size","-b", type=int, default=32, help="Batch size for training.")
    argument_parser.add_argument("--learning_rate","-lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    argument_parser.add_argument("--model-path",default=MODEL_PATH, help="Model path")
    args = argument_parser.parse_args()

    # arguments transformed into variables for easier access
    TRAIN_DATA_PATH = args.train_data_path
    VAL_DATA_PATH = args.val_data_path  
    TEST_DATA_PATH = args.test_data_path
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = {0:"Center", 1:"Donut", 2:"Edge-Loc", 3:"Edge-Ring",
                4:"Loc", 5:"Near-full", 6:"Random", 7:"Scratch"}

    # load data
    train_data = load_image_data(TRAIN_DATA_PATH, include_classes=list(label_names.values()))
    val_data = load_image_data(VAL_DATA_PATH, include_classes=list(label_names.values()))
    test_data = load_image_data(TEST_DATA_PATH, include_classes=list(label_names.values()))
    print (len(train_data))

    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # initialize model, loss function, and optimizer
    model = WaferCNN().to("cuda" if torch.cuda.is_available() else "cpu")

    # summary(model, input_size=(1, 1, 32, 32))

    class_counts = [train_data.data[train_data.data["failureCode"] == i].shape[0] for i in range(NO_CLASS)]

    class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded saved model weights")
    else:
        model = train_classifier(model, train_loader, val_loader, criterion, optimizer, EPOCHS) 

    model = model.to(device)
    all_predictions = predict(model, test_loader)
    all_labels = test_loader.dataset.data['failureCode'].values
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    print(f"Accuracy: {(all_predictions == all_labels).mean()}")
    print(f"F1 Score: {f1}")
    # confusion_matrix_plot(all_labels, all_predictions, label_names)
    print (classification_report(all_labels, all_predictions, target_names=label_names.values()))


    # prediction loop
    