import numpy as np
from torch.utils.data import Dataset
import cv2 as cv
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize


class RetinaDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv.imread(self.images[idx], cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mean = image.mean(axis=(0, 1))
        std = image.std(axis=(0, 1))
        image = (image - mean[None, None, :]) / std[None, None, :]
        # image = image / 255.0
        image = image.astype(np.float32)

        mask = cv.imread(self.masks[idx], cv.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1
        mask = mask.astype(np.float32) 

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        
        image_tensor = self.to_tensor(image)
        mask_tensor = self.to_tensor(mask)
        
        return image_tensor, mask_tensor
