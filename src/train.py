import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import SegmentationDataset
from unet import UNet

# Configuración
BATCH_SIZE = 2
LEARNING_RATE = 0.001
EPOCHS = 5
# DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

# Cargar dataset
train_dataset = SegmentationDataset("dataset/train/images", "dataset/train/masks")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Definir modelo, pérdida y optimizador
model = UNet().to(DEVICE)
criterion = nn.BCELoss()  # Para segmentación binaria
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Entrenamiento
train_losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))
    print(f"Época {epoch+1}/{EPOCHS}, Pérdida: {train_losses[-1]:.4f}")

# Guardar el modelo
torch.save(model.state_dict(), "unet_model.pth")

# Graficar la pérdida
plt.plot(train_losses, label="Pérdida de entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()
plt.show()