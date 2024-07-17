import torch
from model import UNet
from dataset import RetinaDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from utils import load_data
import warnings
import segmentation_models_pytorch as smp
from tqdm import tqdm
import json
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import numpy as np

warnings.filterwarnings("ignore")

EPOCHS = 1
LR = 0.0005
IN_CHANNELS = 3
OUT_CHANNELS = 1
BASE_DIRECTORY = "dataset"
K_FOLD = 8
BATCH_SIZE = 2

CHECKPOINT_PATH = (
    "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/checkpoints/checkpoint.pth"
)
METRICS_PATH = (
    "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/metrics/metrics.json"
)

experiment = Experiment(
    api_key="rwyMmTQC0QDIH0oF5XaSzgmh4",
    project_name="retina-blood-vessel-segmentation",
    workspace="youssefaboelwafa",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_images, train_masks, test_images, test_masks = load_data(BASE_DIRECTORY)

train_dataset = RetinaDataset(train_images, train_masks, augment=True)

# Overfitting to a small subset of the data
subset_indices = np.random.choice(len(train_dataset), 8, replace=False)
train_dataset = Subset(train_dataset, subset_indices)

model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = smp.losses.DiceLoss(mode="binary")


kfold = KFold(n_splits=K_FOLD, shuffle=True)
best_val_loss = float("inf")
results = {}

for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    print(f"Fold {fold + 1}")
    print("-" * 30)
    train_subsampler = Subset(train_dataset, train_ids)
    val_subsampler = Subset(train_dataset, val_ids)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_iou_score": [],
        "val_iou_score": [],
    }

    for epoch in range(EPOCHS):
        model.train()
        train_loss = []
        train_iou_score = []

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
            pred = torch.sigmoid(pred)
            mask = mask.round().long()
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred, mask, mode="binary", threshold=0.5
            )
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            train_loss.append(loss.item())
            train_iou_score.append(iou_score)

            tqdm_train.set_description(
                f"Epoch {epoch+1}/{EPOCHS} [Training] Loss: {sum(train_loss)/len(train_loss):.4f}, "
                f"IOU: {sum(train_iou_score)/len(train_iou_score):.4f}, "
            )

        metrics["train_loss"].append(sum(train_loss) / len(train_loss))
        metrics["train_iou_score"].append(sum(train_iou_score) / len(train_iou_score))

        model.eval()
        val_loss = []
        val_iou_score = []
        tqdm_val = tqdm(
            val_dataloader,
            total=len(val_dataloader),
            desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]",
        )

        with torch.no_grad():
            for image, mask in tqdm_val:
                image = image.to(device=device)
                mask = mask.float().to(device=device)
                pred = model(image)
                loss = criterion(pred, mask)
                pred = torch.sigmoid(pred)
                mask = mask.round().long()
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pred, mask, mode="binary", threshold=0.5
                )
                iou_score = smp.metrics.iou_score(
                    tp, fp, fn, tn, reduction="micro"
                ).item()
                val_loss.append(loss.item())
                val_iou_score.append(iou_score)

                tqdm_val.set_description(
                    f"Epoch {epoch+1}/{EPOCHS} [Validation] Loss: {sum(val_loss)/len(val_loss):.4f}, "
                    f"IOU: {sum(val_iou_score)/len(val_iou_score):.4f}, "
                )

        epoch_val_loss = sum(val_loss) / len(val_loss)
        metrics["val_loss"].append(epoch_val_loss)
        metrics["val_iou_score"].append(sum(val_iou_score) / len(val_iou_score))

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_PATH}_fold_{fold}")
            print(
                f"Checkpoint saved at epoch {epoch+1}, fold {fold} with loss {best_val_loss:.4f}"
            )

    results[fold + 1] = metrics

with open(METRICS_PATH, "w") as f:
    json.dump(results, f)
