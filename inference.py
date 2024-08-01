import os
import torch
import torch.nn as nn
from torchvision import transforms
import albumentations as A
import cv2
import numpy as np
from utils import *
from config import *
from model import UNet

set_seed()

EXP_ID = 18808
MODEL_PATH = f"/scratch/dr/y.aboelwafa/Retina/Retina_Blood_Vessel_Segmentation/checkpoints/pytorch_{EXP_ID}.pth"
INPUT_DIR = "/path/to/input/images"
OUTPUT_DIR = "/path/to/output/masks"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = UNet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

test_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = test_transform(image=image)
    image = augmented['image']
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def save_mask(mask, output_path):
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    cv2.imwrite(output_path, mask)

for image_name in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, image_name)
    output_path = os.path.join(OUTPUT_DIR, image_name)

    image = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        pred = model(image)
        pred = torch.sigmoid(pred)
        save_mask(pred, output_path)

print("Inference completed and masks saved.")