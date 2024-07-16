import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 as cv
import albumentations as A


class RetinaDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        self.images = images
        self.masks = masks
        self.augment = augment
        if augment:
            self.augmentations = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.5),
                ]
            )
        else:
            self.augmentations = None

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

        if self.augment:
            augmented = self.augmentations(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor
