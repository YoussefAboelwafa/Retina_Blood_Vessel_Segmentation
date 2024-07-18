from comet_ml import Experiment
import torch
import torch.nn as nn
from model import UNet
from dataset import RetinaDataset
from torch.utils.data import DataLoader
from utils import load_data
from torch.utils.data import random_split
import warnings
import segmentation_models_pytorch as smp
from datetime import datetime
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--e", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--b", type=int, default=4)
parser.add_argument("--job_id", type=int)

args = parser.parse_args()

EPOCHS = args.e
LR = args.lr
BATCH_SIZE = args.b

IN_CHANNELS = 3
OUT_CHANNELS = 1

BASE_DIRECTORY = "dataset"

CHECKPOINT_PATH = (
    "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/checkpoints/checkpoint"
)

experiment = Experiment(
    api_key="rwyMmTQC0QDIH0oF5XaSzgmh4",
    project_name="retina-blood-vessel-segmentation",
    workspace="youssefaboelwafa",
)

experiment.set_name(str(args.job_id))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_images, train_masks, test_images, test_masks = load_data(BASE_DIRECTORY)

dataset = RetinaDataset(train_images, train_masks, augment=True)
train_dataset, val_dataset = random_split(dataset, [60, 20])


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
    print(
        f"Epoch {epoch+1}/{EPOCHS} [Training]   Loss: {epoch_train_loss:.4f}, "
        f"IOU: {epoch_train_iou_score:.4f}, ",
        flush=True,
    )
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
    epoch_iou_score = sum(val_iou_score) / len(val_iou_score)

    print(
        f"Epoch {epoch+1}/{EPOCHS} [Validation] Loss: {sum(val_loss)/len(val_loss):.4f}, "
        f"IOU: {sum(val_iou_score)/len(val_iou_score):.4f}, ",
        flush=True,
    )

    metrics["val_loss"].append(epoch_val_loss)
    experiment.log_metric("val_loss", epoch_val_loss, step=epoch)
    metrics["val_iou_score"].append(epoch_iou_score)
    experiment.log_metric("val_iou_score", epoch_iou_score, step=epoch)

    results[epoch + 1] = {
        "val_loss": epoch_val_loss,
        "val_iou_score": epoch_iou_score,
    }

    if epoch_iou_score > best_iou:
        best_iou = epoch_iou_score

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
