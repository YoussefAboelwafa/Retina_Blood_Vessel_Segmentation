import os
from PIL import Image


def load_images_from_directory(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            img_path = os.path.join(directory_path, filename)
            try:
                with Image.open(img_path) as img:
                    images.append(img.copy())
            except Exception as e:
                print(f"Failed to load image {filename}: {e}")
    return images


def load_data(directory_path):
    train_images_path = directory_path + "/train/image"
    train_masks_path = directory_path + "/train/mask"
    test_images_path = directory_path + "/test/image"
    test_masks_path = directory_path + "/test/mask"
    return train_images_path, train_masks_path, test_images_path, test_masks_path
