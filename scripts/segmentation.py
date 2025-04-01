# Importaciones
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import albumentations as A
from albumentations.pytorch import ToTensorV2

""" Configuración """
MODEL_PATH = "../models/clasificacion4.pth" # Ruta al modelo entrenado
IMAGE_PATH = "270.jpg" # Imagen de entrada
NUM_CLASSES = 5 # Número de clases
OUTPUT_PATH = "mascara_predicha_rgb.png"
OUTPUT_SIZE = (680, 382)  # width x height

# Colores personalizados para cada clase
custom_colors = [
    "#000000",  # 0 - Fondo
    "#004fff",  # 1 - Agua
    "#f5d6c4",  # 2 - Suelo Expuesto
    "#8f9107",  # 3 - Vegetación Seca
    "#08920a",  # 4 - Vegetación Verde
]
cmap = ListedColormap(custom_colors)

""" Transformaciones """
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

""" Funciones """
def load_image(path, transform):
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    tensor = augmented['image'].unsqueeze(0)  # (1, C, H, W)
    return tensor, image

def predict_mask(model, image_tensor, device):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1)
    return pred.squeeze(0).cpu().numpy()

def mask_to_rgb(mask, color_map):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, hex_color in enumerate(color_map):
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        rgb_mask[mask == label] = rgb
    return rgb_mask

def resize_mask(mask_rgb, size=(680, 382)):
    return cv2.resize(mask_rgb, size, interpolation=cv2.INTER_NEAREST)

""" Programa principal """
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Cargando modelo desde: {MODEL_PATH}")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    image_tensor, original_image = load_image(IMAGE_PATH, transform)
    predicted_mask = predict_mask(model, image_tensor, device)

    # Convertir y redimensionar la máscara a RGB
    rgb_mask = mask_to_rgb(predicted_mask, custom_colors)
    rgb_mask_resized = resize_mask(rgb_mask, OUTPUT_SIZE)

    # Visualizar
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Imagen Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_mask_resized)
    plt.title("Máscara Predicha")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Guardar
    cv2.imwrite(str(OUTPUT_PATH), cv2.cvtColor(rgb_mask_resized, cv2.COLOR_RGB2BGR))
    print(f"Máscara guardada como: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()