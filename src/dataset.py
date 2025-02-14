import torch
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))  # Ordenar imágenes

        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convertir numpy array a imagen PIL
            transforms.Resize((512, 512)),  # 🔹 Cambiar tamaño a 512x512
            transforms.ToTensor(),  # Convertir a tensor de PyTorch
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        # Leer la imagen y la máscara
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises

        # Convertir la máscara a valores entre 0 y 1
        mask = mask / 255.0  # Normalizar entre 0 y 1
        mask = np.expand_dims(mask, axis=0)  # Agregar dimensión de canal

        # Convertir a tensor de PyTorch
        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask