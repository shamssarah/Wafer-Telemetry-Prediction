import torch
from torchvision import transforms
import sys
sys.path.append('src')
#IMPORT LOCAL FUNCTIONS
from data_loader import SmartResizePad
from ImageModel import WaferAutoEncoder,MODEL_PATH

ANOMALY_THRESHOLD = 0.0301

def preprocessImage (img):
    preprocess = transforms.Compose([
        SmartResizePad(target_size=32, fill=0), # Your custom conditional logic
        transforms.ToTensor(),                  # Convert to tensor
        transforms.Normalize(mean=[0.0], std=[1.0]) # Add any other standard transforms
    ])
    return preprocess(img)

def loadModel(device):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = WaferAutoEncoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model.eval()  # set to eval mode here so you don't have to elsewhere
    return model

def reconstruction (model,img):
    recon = model(img)
    pixel_error = (img - recon) ** 2
    wafer_score = pixel_error.mean(dim=[1, 2, 3])
    return wafer_score

def image_pipeline (raw_img):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = preprocessImage(raw_img)
    img   = img.unsqueeze(0)
    model = loadModel (device)

    img = img.to(device)
    with torch.no_grad():  # missing this
        score = reconstruction(model, img)
    
    error = score.item()
    return {
        "anomaly": error > ANOMALY_THRESHOLD,
        "reconstruction_error": error
    }
