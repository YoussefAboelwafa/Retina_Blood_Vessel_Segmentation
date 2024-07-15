import torch
from model import UNet
from dataset import RetinaDataset
from utils import *

model = UNet(3, 1)
input_image = torch.rand(1, 3, 512, 512)
output = model(input_image)
print(type(output), output.size())
