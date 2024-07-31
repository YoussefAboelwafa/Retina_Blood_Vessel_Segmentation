import os
import torch
import numpy as np
import random
from glob import glob
import cv2


def load_data(base_directory="dataset"):
    train_images = sorted(glob(os.path.join(base_directory, "train", "image", "*.png")))
    train_masks = sorted(glob(os.path.join(base_directory, "train", "mask", "*.png")))
    test_images = sorted(glob(os.path.join(base_directory, "test", "image", "*.png")))
    test_masks = sorted(glob(os.path.join(base_directory, "test", "mask", "*.png")))
    return train_images, train_masks, test_images, test_masks


def set_seed():
    torch.manual_seed(5)
    torch.cuda.manual_seed(5)
    torch.cuda.manual_seed_all(5)
    np.random.seed(5)
    random.seed(5)


def connected_components(img, min_threshold=0.2, max_threshold=0.4):

    _, sure_edges = cv2.threshold(img, max_threshold, 1, cv2.THRESH_BINARY)

    uncertain_edges = np.logical_and(img > min_threshold, img < max_threshold).astype(
        np.uint8
    )

    num_sure, sure_labels = cv2.connectedComponents(
        sure_edges.astype(np.uint8), connectivity=8
    )

    num_uncertain, uncertain_labels = cv2.connectedComponents(
        uncertain_edges, connectivity=8
    )

    combined = sure_labels.copy()

    for label in range(1, num_uncertain):
        uncertain_mask = uncertain_labels == label
        sure_neighbors = sure_labels[uncertain_mask]
        sure_neighbors = sure_neighbors[sure_neighbors > 0]

        if len(sure_neighbors) > 0:
            combined[uncertain_mask] = np.bincount(sure_neighbors).argmax()

    combined[combined > 0] = 1

    return combined
