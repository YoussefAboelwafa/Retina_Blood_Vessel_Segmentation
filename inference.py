import os
import torch
from torchvision import transforms
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from utils import *
from model import UNet

set_seed()

EXP_ID = 18808
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = f"/scratch/dr/y.aboelwafa/Retina/Retina_Blood_Vessel_Segmentation/checkpoints/pytorch_{EXP_ID}.pth"

INPUT_DIR = "/scratch/dr/agamal/vm1/datadrive/kaggle-dr"
OUTPUT_DIR = "/scratch/dr/y.aboelwafa/DR/diabetic-retinopathy/datasets/masks"

TRAIN_CSV = "/scratch/dr/y.aboelwafa/DR/diabetic-retinopathy/datasets/kaggle_train.csv"
VAL_CSV = "/scratch/dr/y.aboelwafa/DR/diabetic-retinopathy/datasets/kaggle_val.csv"
TEST_CSV = (
    "/scratch/dr/y.aboelwafa/DR/diabetic-retinopathy/datasets/kaggle_test_public.csv"
)

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

model = UNet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

test_transform = A.Compose(
    [
        A.Resize(512, 512),
    ]
)


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = test_transform(image=image)
    image = augmented["image"]
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    return image


def save_mask(mask, output_path):
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    cv2.imwrite(output_path, mask * 255)


for row in test_df.itertuples():
    image_path = os.path.join(INPUT_DIR, row.path)
    print(f"Processing {image_path}")
    output_path = os.path.join(OUTPUT_DIR, row.path)

    image = preprocess_image(image_path).to(DEVICE)

    try:
        image = preprocess_image(image_path).to(DEVICE)
    except FileNotFoundError as e:
        print(e)
        continue

    with torch.no_grad():
        pred = model(image)
        pred = torch.sigmoid(pred)
        save_mask(pred, output_path)


print("*" * 50)
print("Inference completed on test set")
print("*" * 50)
