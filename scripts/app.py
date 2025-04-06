# Importaciones
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib.colors import ListedColormap
from PIL import Image

# Inicializar Flask
app = Flask(__name__)

# Configuración
MODEL_PATH = "../models/clasificacion4.pth"
NUM_CLASSES = 5
OUTPUT_SIZE = (680, 382)

CLASS_NAMES = {
    0: "Fondo",
    1: "Agua",
    2: "Suelo Expuesto",
    3: "Vegetacion Seca",
    4: "Vegetacion Verde"
}

CLASS_COLORS_HEX = [
    "#000000",  # 0 - Fondo
    "#004fff",  # 1 - Agua
    "#ffffff",  # 2 - Suelo Expuesto
    "#8f9107",  # 3 - Vegetación Seca
    "#08920a",  # 4 - Vegetación Verde
]

# Pipeline de preprocesamiento
preprocess_pipeline = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Cargar modelo
def load_segmentation_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model, device

# Decodificar imagen cargada
def decode_uploaded_image(file):
    image_bytes = np.frombuffer(file.read(), np.uint8)
    bgr_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image

# Preprocesar
def preprocess_image(rgb_image):
    augmented = preprocess_pipeline(image=rgb_image)
    input_tensor = augmented["image"].unsqueeze(0)
    return input_tensor

# Predicción
def generate_segmentation_mask(model, input_tensor, device):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)
    return prediction.squeeze(0).cpu().numpy()

# Convertir a RGB
def convert_mask_to_rgb(mask, class_colors):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, hex_color in enumerate(class_colors):
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        rgb_mask[mask == class_id] = rgb
    return rgb_mask

# Redimensionar
def resize_segmentation_mask(mask_rgb, target_size=OUTPUT_SIZE):
    return cv2.resize(mask_rgb, target_size, interpolation=cv2.INTER_NEAREST)

# Porcentajes
def compute_class_distribution(mask, num_classes=NUM_CLASSES, exclude_class_id=0):
    total_pixels = np.sum(mask != exclude_class_id)
    class_distribution = {}

    for class_id in range(1, num_classes):
        class_pixel_count = np.sum(mask == class_id)
        percentage = (class_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        class_distribution[CLASS_NAMES[class_id]] = round(percentage, 2)

    return class_distribution

# Global temporal para mostrar la última imagen procesada
last_mask_image = None
last_input_image = None

# Endpoint 1: Subir imagen y obtener porcentajes
@app.route("/predict-mask/", methods=["POST"])
def predict_mask():
    global last_mask_image, last_input_image

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    rgb_image = decode_uploaded_image(file)

    # Guardar la imagen redimensionada antes del modelo
    resized = cv2.resize(rgb_image, OUTPUT_SIZE)
    last_input_image = Image.fromarray(resized)

    input_tensor = preprocess_image(rgb_image)
    model, device = load_segmentation_model()

    predicted_mask = generate_segmentation_mask(model, input_tensor, device)
    class_percentages = compute_class_distribution(predicted_mask, NUM_CLASSES)
    mask_rgb = convert_mask_to_rgb(predicted_mask, CLASS_COLORS_HEX)
    mask_resized = resize_segmentation_mask(mask_rgb, OUTPUT_SIZE)

    # Guardar la máscara procesada
    last_mask_image = Image.fromarray(mask_resized)

    return jsonify({
        "Archivo": file.filename,
        "Distribucion de clases": class_percentages
    })

@app.route("/")
def say_hello():
    print("Hola")

# Endpoint 2: Ver la imagen segmentada más reciente
@app.route("/mask-image/")
def get_last_mask_image():
    if last_mask_image is None:
        return jsonify({"error": "No mask image available"}), 404

    buf = BytesIO()
    last_mask_image.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# Endpoint 3: Ver la imagen original
@app.route("/original-image/")
def get_original_image():
    if last_input_image is None:
        return jsonify({"error": "No input image available"}), 404

    buf = BytesIO()
    last_input_image.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# Iniciar la aplicación
if __name__ == "__main__":
    app.run(debug=True, port=8000)