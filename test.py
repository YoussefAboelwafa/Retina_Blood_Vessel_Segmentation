import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_data
from dataset import RetinaDataset
from model import UNet
import segmentation_models_pytorch as smp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIRECTORY = "dataset"
CHECKPOINT_PATH = "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/checkpoints/checkpoint_17_07_18_48_11.pth"


model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
criterion = nn.BCEWithLogitsLoss()
model.eval()

train_images, train_masks, test_images, test_masks = load_data(BASE_DIRECTORY)
test_dataset = RetinaDataset(test_images, test_masks, augment=False)
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
