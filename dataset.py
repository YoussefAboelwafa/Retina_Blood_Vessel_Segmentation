from utils import load_data
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


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

        mask = cv.imread(self.masks[idx], cv.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1
        mask = mask.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor


class RetinaDataModule(pl.LightningDataModule):
    def __init__(self, dataset_directory, train_transform, test_transform, batch_size):
        super().__init__()
        self.train_images, self.train_masks, self.test_images, self.test_masks = (
            load_data(dataset_directory)
        )
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_images, val_images, train_masks, val_masks = train_test_split(
                self.train_images, self.train_masks, test_size=0.25, random_state=5
            )

            self.train_dataset = RetinaDataset(
                train_images, train_masks, transform=self.train_transform
            )
            self.val_dataset = RetinaDataset(
                val_images, val_masks, transform=self.test_transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = RetinaDataset(
                self.test_images, self.test_masks, transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)
