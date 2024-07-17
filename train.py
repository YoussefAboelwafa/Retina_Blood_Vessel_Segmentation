import torch
from model import UNet
from dataset import RetinaDataset
from torch.utils.data import DataLoader
from utils import load_data
import warnings
import segmentation_models_pytorch as smp
import json
from comet_ml import Experiment

warnings.filterwarnings("ignore")

EPOCHS = 100
LR = 0.0001
IN_CHANNELS = 3
OUT_CHANNELS = 1
BASE_DIRECTORY = "dataset"
BATCH_SIZE = 16

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
test_dataset = RetinaDataset(test_images, test_masks, augment=False)

model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = smp.losses.DiceLoss(mode="binary")


best_iou = float("-inf")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

metrics = {
    "train_loss": [],
    "val_loss": [],
    "train_iou_score": [],
    "val_iou_score": [],
}
results = {}

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

        print(
            f"Epoch {epoch+1}/{EPOCHS} [Training] Loss: {sum(train_loss)/len(train_loss):.4f}, "
            f"IOU: {sum(train_iou_score)/len(train_iou_score):.4f}, ",
            flush=True,
        )
    epoch_train_loss = sum(train_loss) / len(train_loss)
    metrics["train_loss"].append(epoch_train_loss)
    experiment.log_metric("train_loss", epoch_train_loss, step=epoch)
    epoch_train_iou_score = sum(train_iou_score) / len(train_iou_score)
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

            print(
                f"Epoch {epoch+1}/{EPOCHS} [Validation] Loss: {sum(val_loss)/len(val_loss):.4f}, "
                f"IOU: {sum(val_iou_score)/len(val_iou_score):.4f}, ",
                flush=True,
            )

    epoch_val_loss = sum(val_loss) / len(val_loss)
    metrics["val_loss"].append(epoch_val_loss)
    experiment.log_metric("val_loss", epoch_val_loss, step=epoch)
    epoch_iou_score = sum(val_iou_score) / len(val_iou_score)
    metrics["val_iou_score"].append(epoch_iou_score)
    experiment.log_metric("val_iou_score", epoch_iou_score, step=epoch)

    results[epoch + 1] = {
        "val_loss": epoch_val_loss,
        "val_iou_score": epoch_iou_score,
    }

    if epoch_iou_score > best_iou:
        best_iou = epoch_iou_score
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(
            f"Checkpoint saved at epoch {epoch+1} with loss {best_iou:.4f}",
            flush=True,
        )

with open(METRICS_PATH, "w") as f:
    json.dump(results, f)
