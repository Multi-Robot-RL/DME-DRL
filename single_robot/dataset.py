from os import makedirs
from datasets import load_dataset, load_from_disk
from PIL import Image
import numpy as np
import io
from pathlib import Path


# Load the HouseExpo dataset

def process_image(image):
    # Convert to numpy array
    image_array = np.array(image)

    # Create binary mask: 0 for free space (white), 1 for obstacles (black)
    binary_mask = np.where(image_array < 128, 1, 0)  # Threshold at 128

    # Calculate features
    total_pixels = binary_mask.size
    obstacle_pixels = np.sum(binary_mask)
    free_space_pixels = total_pixels - obstacle_pixels

    obstacle_percentage = obstacle_pixels / total_pixels
    free_space_percentage = free_space_pixels / total_pixels

    return {
        'binary_mask': binary_mask,
        'obstacle_percentage': obstacle_percentage,
        'free_space_percentage': free_space_percentage
    }


data_path = Path("data")


def load_processed_dataset(split: str = "train"):
    # if data_path does not exist, mkdir
    base_path = data_path / split
    if not data_path.exists(follow_symlinks=False):
        makedirs(base_path)

    # Load the original dataset
    original_dataset_path = base_path / "house_expo_dataset"
    if (original_dataset_path.exists()):
        dataset = load_from_disk(str(original_dataset_path))
    else:
        dataset = load_dataset('cwyark/HouseExpo', split=split)
        dataset.save_to_disk(str(original_dataset_path))

    # Apply the processing function to the dataset
    mapped_dataset_path = base_path / "processed_house_expo_dataset"
    if (mapped_dataset_path.exists()):
        dataset = load_from_disk(str(mapped_dataset_path))
    else:
        dataset = dataset.map(lambda x: process_image(x['image']))
        dataset.save_to_disk(str(mapped_dataset_path))


if __name__ == "__main__":
    load_processed_dataset()

