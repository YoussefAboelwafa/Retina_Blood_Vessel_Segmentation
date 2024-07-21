from comet_ml import Experiment
import torch
import torch.nn as nn
from model import UNet
from dataset import RetinaDataset
import albumentations as A
from torch.utils.data import DataLoader
from utils import *
from sklearn.model_selection import train_test_split
import warnings
import segmentation_models_pytorch as smp
from datetime import datetime
import argparse

warnings.filterwarnings("ignore")
set_seed(5)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--job_id", type=int)

args = parser.parse_args()

EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch_size

IN_CHANNELS = 3
OUT_CHANNELS = 1

BASE_DIRECTORY = "dataset"

CHECKPOINT_PATH = "/scratch/y.aboelwafa/Retina/Retina_Blood_Vessel_Segmentation/checkpoints/checkpoint"

experiment = Experiment(
    api_key="rwyMmTQC0QDIH0oF5XaSzgmh4",
    project_name="retina-blood-vessel-segmentation",
    workspace="youssefaboelwafa",
)

experiment.set_name(str(args.job_id))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images, masks, _, _ = load_data(BASE_DIRECTORY)

train_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Sharpen(alpha=(0.5, 0.9), lightness=(0.5, 1.0), p=0.3),
        A.Emboss(alpha=(0.1, 0.3), strength=(0.5, 1.0), p=0.3),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(512, 512),
    ]
)

train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.25, random_state=5
)


train_dataset = RetinaDataset(train_images, train_masks, transform=train_transform)
val_dataset = RetinaDataset(val_images, val_masks, transform=test_transform)

model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

if torch.cuda.device_count() > 1:
    print(f"start training using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model.to(device)

best_iou = float("-inf")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

metrics = {
    "train_loss": [],
    "val_loss": [],
    "train_iou_score": [],
    "val_iou_score": [],
}
results = {}

start_time = datetime.now()

for epoch in range(EPOCHS):
    model.train()
    train_loss = []
    train_iou_score = []

    for batch_idx, (image, mask) in enumerate(train_dataloader):
        image = image.to(device=device)
        mask = mask.to(device=device)
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        pred = torch.sigmoid(pred)
        mask = mask.round().long()
        tp, fp, fn, tn = smp.metrics.get_stats(pred, mask, mode="binary", threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
        train_loss.append(loss.item())
        train_iou_score.append(iou_score)

    epoch_train_loss = sum(train_loss) / len(train_loss)
    epoch_train_iou_score = sum(train_iou_score) / len(train_iou_score)

    metrics["train_loss"].append(epoch_train_loss)
    experiment.log_metric("train_loss", epoch_train_loss, step=epoch)
    metrics["train_iou_score"].append(epoch_train_iou_score)
    experiment.log_metric("train_iou_score", epoch_train_iou_score, step=epoch)

    model.eval()
    val_loss = []
    val_iou_score = []

    with torch.no_grad():
        for i, (image, mask) in enumerate(val_dataloader):
            image = image.to(device=device)
            mask = mask.float().to(device=device)
            pred = model(image)
            loss = criterion(pred, mask)
            pred = torch.sigmoid(pred)
            mask = mask.round().long()
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred, mask, mode="binary", threshold=0.5
            )
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            val_loss.append(loss.item())
            val_iou_score.append(iou_score)

    epoch_val_loss = sum(val_loss) / len(val_loss)
    epoch_val_iou_score = sum(val_iou_score) / len(val_iou_score)

    metrics["val_loss"].append(epoch_val_loss)
    experiment.log_metric("val_loss", epoch_val_loss, step=epoch)
    metrics["val_iou_score"].append(epoch_val_iou_score)
    experiment.log_metric("val_iou_score", epoch_val_iou_score, step=epoch)

    results[epoch + 1] = {
        "val_loss": epoch_val_loss,
        "val_iou_score": epoch_val_iou_score,
    }

    if epoch % 50 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, IoU: {epoch_val_iou_score:.2f}% | Val loss: {epoch_val_loss:.5f}, Val IoU: {epoch_val_iou_score:.2f}%"
        )
        print("-" * 50)

    if epoch_val_iou_score > best_iou:
        best_iou = epoch_val_iou_score

        checkpoint_path_with_job_id = f"{CHECKPOINT_PATH}_{str(args.job_id)}.pth"

        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), checkpoint_path_with_job_id)
        else:
            torch.save(model.state_dict(), checkpoint_path_with_job_id)

        print(
            f"Checkpoint saved at epoch {epoch+1} with IOU {best_iou:.4f}",
            flush=True,
        )
        print("-" * 50)


best_epoch = max(results, key=lambda epoch: results[epoch]["val_iou_score"])
best_val_iou_score = results[best_epoch]["val_iou_score"]
best_val_loss = results[best_epoch]["val_loss"]

print("Number of GPUs: ", torch.cuda.device_count())
print("Time taken: ", datetime.now() - start_time)
print(
    f"Best IOU Score: {best_val_iou_score:.4f} with Loss: {best_val_loss:.4f} at Epoch: {best_epoch}"
)
print("Checkpoint path: ", checkpoint_path_with_job_id)
print("Batch size: ", BATCH_SIZE)
print("Learning rate: ", LR)
