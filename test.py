import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from utils import *
from dataset import RetinaDataset
from model import LitUnet
import segmentation_models_pytorch as smp

set_seed(5)

EXP_ID = 18439

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIRECTORY = "dataset"
MODEL_PATH = f"/scratch/y.aboelwafa/Retina/Retina_Blood_Vessel_Segmentation/checkpoints/checkpoint_{EXP_ID}.pth"

model = LitUnet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
criterion = nn.BCEWithLogitsLoss()
model.eval()

_, _, images, masks = load_data(BASE_DIRECTORY)

test_transform = A.Compose(
    [
        A.Resize(512, 512),
    ]
)

test_dataset = RetinaDataset(images, masks, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

test_loss = []
test_iou_score = []

for i, (image, mask) in enumerate(test_dataloader):
    image = image.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        pred = model(image)
        loss = criterion(pred, mask)
        pred = torch.sigmoid(pred)
        mask = mask.round().long()
        tp, fp, fn, tn = smp.metrics.get_stats(pred, mask, mode="binary", threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
        test_loss.append(loss.item())
        test_iou_score.append(iou_score)


print(f"Test Loss: {sum(test_loss) / len(test_loss)}")
print(f"Test IoU Score: {sum(test_iou_score) / len(test_iou_score)}")
