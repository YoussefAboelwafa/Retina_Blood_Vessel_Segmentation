import torch
from model import UNet
from dataset import RetinaDataset
from torch.utils.data import DataLoader
from utils import load_data
import warnings
import json

warnings.filterwarnings("ignore")

H = 512
W = 512
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
IN_CHANNELS = 3
OUT_CHANNELS = 1
CHECKPOINT_PATH = "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/checkpoints/checkpoint.pth"
METRICS_PATH = "/scratch/y.aboelwafa/Retina_Blood_Vessel_Segmentation/metrics/metrics.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_images, train_masks, test_images, test_masks = load_data("dataset")

train_dataset = RetinaDataset(train_images, train_masks, augment=True)
test_dataset = RetinaDataset(test_images, test_masks)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

