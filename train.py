import torch
from model import UNet
from dataset import RetinaDataset
from torch.utils.data import DataLoader
from utils import load_data
import warnings
import segmentation_models_pytorch as smp
from tqdm import tqdm
import json

warnings.filterwarnings("ignore")

BATCH_SIZE = 4
EPOCHS = 50
LR = 0.001
IN_CHANNELS = 3
OUT_CHANNELS = 1
BASE_DIRECTORY = "dataset"

CHECKPOINT_PATH = (
    "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/checkpoints/checkpoint.pth"
)
METRICS_PATH = (
    "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/metrics/metrics.json"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_images, train_masks, test_images, test_masks = load_data(BASE_DIRECTORY)

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


best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = []
    train_iou_score = []
    val_iou_score = []

    tqdm_train = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Epoch {epoch+1}/{EPOCHS} [Training]",
    )

    for batch_idx, (image, mask) in tqdm_train:
        image = image.to(device=device)
        mask = mask.to(device=device)
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        tqdm_train.set_description(
            f"Epoch {epoch+1}/{EPOCHS} [Training]   Loss: {sum(train_loss)/len(train_loss):.4f}"
        )

    metrics["train_loss"].append(sum(train_loss) / len(train_loss))

    model.eval()
    val_loss = []
    val_iou_score = []
    val_dice_score = []
    tqdm_val = tqdm(
        test_dataloader,
        total=len(test_dataloader),
        desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]",
    )
    with torch.no_grad():
        for image, mask in tqdm_val:
            image = image.to(device=device)
            mask = mask.float().to(device=device)
            pred = model(image)
            loss = criterion(pred, mask)
            val_loss.append(loss.item())
            tqdm_val.set_description(
                f"Epoch {epoch+1}/{EPOCHS} [Validation] Loss: {sum(val_loss)/len(val_loss):.4f}"
            )

    epoch_val_loss = sum(val_loss) / len(val_loss)
    metrics["val_loss"].append(epoch_val_loss)

    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Checkpoint saved at epoch {epoch+1} with loss {best_loss:.4f}")

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)
