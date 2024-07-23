import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 as cv
from torchvision.transforms import ToTensor


class RetinaDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.to_tensor = ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv.imread(self.images[idx], cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image / 255.0
        image = image.astype(np.float32)

        mask = cv.imread(self.masks[idx], cv.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1
        mask = mask.astype(np.float32)

        image_tensor = self.to_tensor(image)
        mask_tensor = self.to_tensor(mask)

        if self.transform:
            augmented = self.transform(image=image_tensor, mask=mask_tensor)
            image_tensor, mask_tensor = augmented["image"], augmented["mask"]

        return image_tensor, mask_tensor
