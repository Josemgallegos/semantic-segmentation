import os
import shutil
import random

dataset_path = "dataset/"
train_path = "dataset/train/"
val_path = "dataset/val/"
test_path = "dataset/test/"

for path in [train_path, val_path, test_path]:
    os.makedirs(os.path.join(path, "images"), exist_ok=True)
    os.makedirs(os.path.join(path, "masks"), exist_ok=True)

images = sorted(os.listdir(os.path.join(dataset_path, "augmented_images")))
random.shuffle(images)

train_split = int(0.7 * len(images))
val_split = int(0.9 * len(images))

for i, img in enumerate(images):
    mask = img.replace(".jpg", ".png")
    subset = train_path if i < train_split else val_path if i < val_split else test_path

    shutil.copy(os.path.join(dataset_path, "augmented_images", img), os.path.join(subset, "images", img))
    shutil.copy(os.path.join(dataset_path, "augmented_masks", mask), os.path.join(subset, "masks", mask))

print("Dataset dividido en Train, Val y Test.")