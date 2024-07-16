import torch
from model import UNet
from dataset import RetinaDataset
from torch.utils.data import DataLoader
from utils import load_data
import warnings
import segmentation_models_pytorch as smp
import json

warnings.filterwarnings("ignore")

BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
IN_CHANNELS = 3
OUT_CHANNELS = 1
CHECKPOINT_PATH = (
    "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/checkpoints/checkpoint.pth"
)
METRICS_PATH = (
    "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/metrics/metrics.json"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_images, train_masks, test_images, test_masks = load_data("dataset")

train_dataset = RetinaDataset(train_images, train_masks, augment=True)
test_dataset = RetinaDataset(test_images, test_masks)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = smp.losses.DiceLoss(mode="binary")

metrics = {
    "train_loss": [],
    "val_loss": [],
    "train_iou_score": [],
    "val_iou_score": [],
    "train_dice_score": [],
    "val_dice_score": [],
}

for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (image, mask) in enumerate(train_dataloader):
        image = image.to(device=device)
        mask = mask.to(device=device)
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print(f"Epoch {epoch}, batch {batch_idx}, loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        for image, mask in test_dataloader:
            image = image.to(device=device)
            mask = mask.float().to(device=device)
            pred = model(image)
            loss = criterion(pred, mask)
            print(f"Validation loss: {loss.item()}")
