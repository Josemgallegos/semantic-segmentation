import albumentations as A
import cv2
import os
import numpy as np

# Directorios
image_dir = "dataset/images/"
mask_dir = "dataset/masks/"
output_image_dir = "dataset/augmented_images/"
output_mask_dir = "dataset/augmented_masks/"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Definir transformaciones
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.6),
    A.GaussNoise(p=0.5),
], additional_targets={"mask": "image"})  # Asegurar que la máscara también reciba las mismas transformaciones

# Aplicar transformaciones
for img_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))

    # Leer la imagen y la máscara
    image = cv2.imread(image_path)  # Imagen en BGR con 3 canales
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Mantener los valores originales de la máscara

    # Verificar que la máscara no sea convertida a 3 canales
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        print(f"Advertencia: {mask_path} ya tiene 3 canales, pero debería ser de una sola banda.")

    for i in range(5):  # Generar 5 imágenes aumentadas
        augmented = transform(image=image, mask=mask)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        # Guardar sin alterar el formato de la máscara
        cv2.imwrite(os.path.join(output_image_dir, f"{img_name[:-4]}_aug{i}.jpg"), aug_image)
        cv2.imwrite(os.path.join(output_mask_dir, f"{img_name[:-4]}_aug{i}.png"), aug_mask)

print("Aumento de datos terminado.")