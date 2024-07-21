import os
import torch
import numpy as np
import random
from glob import glob


def load_data(base_directory="dataset"):
    train_images = sorted(glob(os.path.join(base_directory, "train", "image", "*.png")))
    train_masks = sorted(glob(os.path.join(base_directory, "train", "mask", "*.png")))
    test_images = sorted(glob(os.path.join(base_directory, "test", "image", "*.png")))
    test_masks = sorted(glob(os.path.join(base_directory, "test", "mask", "*.png")))
    return train_images, train_masks, test_images, test_masks

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)