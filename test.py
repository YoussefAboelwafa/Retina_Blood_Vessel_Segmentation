import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils import load_data
from dataset import RetinaDataset
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIRECTORY = "dataset"
CHECKPOINT_PATH = (
    "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/checkpoints/checkpoint.pth"
)

model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu')))
model.eval()

train_images, train_masks, test_images, test_masks = load_data(BASE_DIRECTORY)
test_dataset = RetinaDataset(test_images, test_masks, augment=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

