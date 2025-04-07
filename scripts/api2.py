import os
import json
import torch
import cv2
import httpx
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from fastapi import FastAPI
from pydantic import BaseModel

# Inicialización de la aplicación FastAPI
app = FastAPI()

# Esquema de entrada
class Folder(BaseModel):
    name: str

# Configuración general
FOLDER_PATH = "C:/Users/JoPaG/OneDrive/Documentos/GitHub/Endpoint" # Cambiar por la ruta deseada
MODEL_PATH = "UNetModel.pth" # Cambiar por la ruta deseada
NUM_CLASSES = 5
OUTPUT_SIZE = (680, 382) # width, height

# Nombres de las clases
CLASS_NAMES = {
    0: "Fondo",
    1: "Agua",
    2: "Suelo Expuesto",
    3: "Vegetacion Seca",
    4: "Vegetacion Verde"
}

# Colores en formato HEX para la visualización
CLASS_COLORS_HEX = [
    "#000000",  # 0 - Fondo
    "#004fff",  # 1 - Agua
    "#ffffff",  # 2 - Suelo Expuesto
    "#8f9107",  # 3 - Vegetación Seca
    "#08920a",  # 4 - Vegetación Verde
]

# Pipeline de preprocesamiento de imágenes
preprocess_pipeline = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Carga el modelo de segmentación entrenado
def load_segmentation_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model, device

# Carga las imágenes .jpg desde la carpeta especificada
def load_images(image_dir):
    image_list = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg",".JPG")):
            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = preprocess_pipeline(image=image)
            tensor = augmented['image'].unsqueeze(0)
            image_list.append((filename, tensor))
    return image_list

# Convierte una máscara de clases a imagen RGB con colores definidos
def convert_mask_to_rgb(mask, class_colors):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, hex_color in enumerate(class_colors):
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        rgb_mask[mask == class_id] = rgb
    return rgb_mask

# Redimensiona la máscara segmentada al tamaño deseado
def resize_segmentation_mask(mask_rgb, target_size=OUTPUT_SIZE):
    return cv2.resize(mask_rgb, target_size, interpolation=cv2.INTER_NEAREST)

# Calcula la distribución de clases en porcentaje, ignorando el fondo
def compute_class_distribution(mask, num_classes=NUM_CLASSES, exclude_class_id=0):
    total_pixels = np.sum(mask != exclude_class_id)
    class_distribution = {}

    for class_id in range(1, num_classes):
        class_pixel_count = np.sum(mask == class_id)
        percentage = (class_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        class_distribution[CLASS_NAMES[class_id]] = round(percentage, 2)

    return class_distribution

# Aplica la lógica de filtrado para decidir si se debe descartar una imagen
def should_discard(distribution: dict) -> bool:
    suelo = distribution.get("Suelo Expuesto", 0)
    agua = distribution.get("Agua", 0)
    seca = distribution.get("Vegetacion Seca", 0)
    verde = distribution.get("Vegetacion Verde", 0)

    # Ajustar valores
    if suelo > 50:
        return True
    if agua > 40:
        return True
    if seca == 0 and verde == 0:
        return True

    return False

# Extrae todos los metadatos EXIF de una imagen, incluyendo información GPS si está disponible
def get_exif_data(image_path):
    image = Image.open(image_path) # Abrir imagen con Pillow (conserva EXIF)
    info = image._getexif()

    if not info:
        return None  # No hay EXIF

    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            gps_data = {}
            for t in value:
                sub_decoded = GPSTAGS.get(t, t)
                gps_data[sub_decoded] = value[t]
            return {"GPSInfo": gps_data} # Devolver solo la parte GPS

    return None # No se encontró GPSInfo

# Convierte los datos GPS (en formato grados, minutos, segundos) a coordenadas decimales
def get_coordinates(gps_info):
    def convert_to_degrees(value):
        try:
            # Soporta tuplas y objetos tipo IFDRational
            def to_float(x):
                return float(x[0]) / float(x[1]) if isinstance(x, tuple) else float(x)

            d, m, s = value
            return to_float(d) + to_float(m) / 60 + to_float(s) / 3600
        except Exception as e:
            print(f"[ERROR] Al convertir coordenadas: {e}")
            return None

    try:
        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = convert_to_degrees(gps_info["GPSLatitude"])
            lon = convert_to_degrees(gps_info["GPSLongitude"])

            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat
            if gps_info.get("GPSLongitudeRef") == "W":
                lon = -lon

            return [round(lat, 6), round(lon, 6)]  # Coordenadas redondeadas
    except Exception as e:
        print(f"[ERROR] Al obtener coordenadas: {e}")

    return None  # Si no hay datos válidos

# Endpoint para segmentar las imágenes de una carpeta
@app.post("/segmentation")
async def segment_folder_images(folder: Folder):
    if not folder.name.strip():
        return {"error": "El nombre de la carpeta no fue proporcionado correctamente."}

    full_path = os.path.join(FOLDER_PATH, folder.name)

    if os.path.exists(os.path.join(full_path, "resultados.json")):
        return {"error": "Esta carpeta ya fue procesada anteriormente. El archivo 'resultados.json' ya existe."}

    if not os.path.exists(full_path):
        return {"error": f"La carpeta '{folder.name}' no fue encontrada en la ruta especificada."}

    model, device = load_segmentation_model()
    images = load_images(full_path)

    if not images:
        return {"error": f"La carpeta '{folder.name}' no contiene imágenes .jpg para procesar."}

    results = []

    for filename, tensor in images:
        tensor = tensor.to(device)
        with torch.no_grad():
            output = model(tensor)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Convertir la predicción a máscara RGB
        mask_rgb = convert_mask_to_rgb(prediction, CLASS_COLORS_HEX)
        mask_resized = resize_segmentation_mask(mask_rgb, OUTPUT_SIZE)

        # Extraer timestamp desde el nombre de la imagen (formato: image_<timestamp>.jpg)
        timestamp = os.path.splitext(filename)[0].replace("image_", "")

        # Crear nombre para la máscara
        output_filename = f"mask_{timestamp}.png"
        output_path = os.path.join(full_path, output_filename)

        # Guardar la máscara como imagen PNG
        cv2.imwrite(output_path, cv2.cvtColor(mask_resized, cv2.COLOR_RGB2BGR))

        # Calcular la distribución de clases
        class_distribution = compute_class_distribution(prediction)

        # Aplicar filtrado: eliminar imágenes irrelevantes
        if should_discard(class_distribution):
            os.remove(os.path.join(full_path, filename))
            os.remove(output_path)
            continue

        # Obtener coordenadas GPS desde los metadatos EXIF
        image_path = os.path.join(full_path, filename)
        coords = None
        try:
            exif = get_exif_data(image_path)
            if exif and "GPSInfo" in exif:
                coords = get_coordinates(exif["GPSInfo"])
        except Exception as e:
            print(f"Error leyendo EXIF de {filename}: {e}")

        # Agregar resultado válido
        results.append({
            "image": filename,
            "mask": output_filename,
            "class_distribution": class_distribution,
            "coordinates": coords
        })

    # Guardar resultados en archivo JSON
    output_data = {
        "results": results
    }

    json_output_path = os.path.join(full_path, "resultados.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    # Llamar a la otra API
    async with httpx.AsyncClient() as client:
        try:
            # response = await client.post("http://localhost:8000/routes/parse")  # Cambiar la endpoint
            print(f"Llamada a API externa realizada")
        except httpx.RequestError as e:
            print(f"Error al contactar API externa: {e}")

    return {"message": f"Se han procesado las imágenes en la carpeta: {folder.name}"}
