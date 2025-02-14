import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet

# Cargar el modelo
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(DEVICE)
model.load_state_dict(torch.load("unet_model.pth", map_location=DEVICE))
model.eval()

# Cargar una imagen de prueba
image_path = "dataset/test/images/image1.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0  # Normalizar
image = np.transpose(image, (2, 0, 1))  # Reordenar para PyTorch
image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# Inferencia
with torch.no_grad():
    output = model(image_tensor)

# Convertir la salida a imagen
mask = output.squeeze().cpu().numpy()
mask = (mask > 0.5).astype(np.uint8)  # Umbral de segmentación

# Mostrar resultados
plt.subplot(1, 2, 1)
plt.imshow(cv2.imread(image_path)[:, :, ::-1])  # Imagen original
plt.title("Imagen Original")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")  # Máscara segmentada
plt.title("Segmentación")

plt.show()