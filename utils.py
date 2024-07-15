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


def load_data(base_directory="dataset"):
    train_images_path = base_directory + "/train/image"
    train_masks_path = base_directory + "/train/mask"
    test_images_path = base_directory + "/test/image"
    test_masks_path = base_directory + "/test/mask"
    tain_images = load_images_from_directory(train_images_path)
    train_masks = load_images_from_directory(train_masks_path)
    test_images = load_images_from_directory(test_images_path)
    test_masks = load_images_from_directory(test_masks_path)
    return tain_images, train_masks, test_images, test_masks
