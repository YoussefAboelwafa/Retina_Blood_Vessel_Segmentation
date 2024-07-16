import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 as cv


class RetinaDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv.imread(self.images[idx], cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image / 255.0
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        mask = cv.imread(self.masks[idx], cv.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = mask.astype(np.float32)
        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)
            mask_tensor = self.transform(mask_tensor)

        return image_tensor, mask_tensor
