#!/usr/bin/env python3
"""
Create Synthetic Cucumber Detection Dataset
============================================
Generates synthetic images with elongated shapes (simulating cucumbers)
for testing the Plan B-Minimal RT-DETR configuration.

This creates a small but functional dataset for verification purposes.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw

# Configuration
NUM_TRAIN_IMAGES = 100
NUM_VAL_IMAGES = 30
IMAGE_SIZE = 640
OUTPUT_DIR = "datasets/cucumber_test"
CLASSES = ["harvestable", "no_harvestable"]


def create_elongated_shape(draw, x, y, w, h, class_id, img_size):
    """Create an elongated shape (simulating cucumber) on the image."""
    # Color based on class (green variants for cucumbers)
    if class_id == 0:  # harvestable - healthy green
        color = (random.randint(50, 100), random.randint(150, 200), random.randint(50, 100))
    else:  # no_harvestable - yellowish/brownish
        color = (random.randint(150, 200), random.randint(150, 180), random.randint(50, 100))

    # Add slight rotation simulation by using ellipse
    angle = random.uniform(-30, 30)

    # Draw elongated ellipse (cucumber-like)
    bbox = [x - w/2, y - h/2, x + w/2, y + h/2]
    draw.ellipse(bbox, fill=color, outline=(0, 100, 0), width=2)

    # Add some texture lines
    for _ in range(3):
        offset = random.uniform(-h/4, h/4)
        line_color = tuple(max(0, c - 30) for c in color)
        draw.line([(x - w/3, y + offset), (x + w/3, y + offset)], fill=line_color, width=1)

    # Return normalized YOLO format: class_id, x_center, y_center, width, height
    return [
        class_id,
        x / img_size,
        y / img_size,
        w / img_size,
        h / img_size
    ]


def create_image_with_annotations(img_id, split):
    """Create a single image with multiple cucumber-like objects."""
    # Create image with random background (simulating field)
    img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
    draw = ImageDraw.Draw(img)

    # Green/brown background (field-like)
    bg_color = (
        random.randint(50, 100),
        random.randint(100, 150),
        random.randint(30, 80)
    )
    draw.rectangle([0, 0, IMAGE_SIZE, IMAGE_SIZE], fill=bg_color)

    # Add noise/texture to background
    for _ in range(200):
        px = random.randint(0, IMAGE_SIZE-1)
        py = random.randint(0, IMAGE_SIZE-1)
        noise_color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in bg_color)
        draw.point((px, py), fill=noise_color)

    # Number of objects per image
    num_objects = random.randint(2, 6)
    annotations = []

    for _ in range(num_objects):
        # Random class (weighted towards harvestable)
        class_id = 0 if random.random() < 0.7 else 1

        # Elongated aspect ratio (width < height for vertical cucumbers)
        if random.random() < 0.5:
            # Vertical cucumber
            w = random.uniform(30, 60)
            h = random.uniform(100, 200)
        else:
            # Horizontal cucumber
            w = random.uniform(100, 200)
            h = random.uniform(30, 60)

        # Random position (ensure object is within image)
        x = random.uniform(w/2 + 10, IMAGE_SIZE - w/2 - 10)
        y = random.uniform(h/2 + 10, IMAGE_SIZE - h/2 - 10)

        # Create object and get annotation
        ann = create_elongated_shape(draw, x, y, w, h, class_id, IMAGE_SIZE)
        annotations.append(ann)

    # Save image
    img_path = f"{OUTPUT_DIR}/{split}/images/{img_id:06d}.jpg"
    img.save(img_path, "JPEG", quality=90)

    # Save annotations in YOLO format
    label_path = f"{OUTPUT_DIR}/{split}/labels/{img_id:06d}.txt"
    with open(label_path, 'w') as f:
        for ann in annotations:
            line = f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n"
            f.write(line)

    return len(annotations)


def create_data_yaml():
    """Create the data.yaml configuration file."""
    yaml_content = f"""# Cucumber Detection Test Dataset
# Auto-generated for Plan B-Minimal verification

path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: val/images

# Classes
names:
  0: harvestable
  1: no_harvestable

nc: 2
"""
    with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f:
        f.write(yaml_content)
    print(f"Created data.yaml at {OUTPUT_DIR}/data.yaml")


def main():
    """Main function to generate the dataset."""
    print("Creating Synthetic Cucumber Detection Dataset")
    print("=" * 50)

    # Create directories
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            os.makedirs(f"{OUTPUT_DIR}/{split}/{subdir}", exist_ok=True)

    # Generate training images
    print(f"\nGenerating {NUM_TRAIN_IMAGES} training images...")
    total_train_objects = 0
    for i in range(NUM_TRAIN_IMAGES):
        num_objs = create_image_with_annotations(i, 'train')
        total_train_objects += num_objs
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{NUM_TRAIN_IMAGES}")

    print(f"Training set: {NUM_TRAIN_IMAGES} images, {total_train_objects} objects")

    # Generate validation images
    print(f"\nGenerating {NUM_VAL_IMAGES} validation images...")
    total_val_objects = 0
    for i in range(NUM_VAL_IMAGES):
        num_objs = create_image_with_annotations(i, 'val')
        total_val_objects += num_objs

    print(f"Validation set: {NUM_VAL_IMAGES} images, {total_val_objects} objects")

    # Create data.yaml
    create_data_yaml()

    # Summary
    print("\n" + "=" * 50)
    print("Dataset Creation Complete!")
    print(f"Total images: {NUM_TRAIN_IMAGES + NUM_VAL_IMAGES}")
    print(f"Total objects: {total_train_objects + total_val_objects}")
    print(f"Dataset location: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
