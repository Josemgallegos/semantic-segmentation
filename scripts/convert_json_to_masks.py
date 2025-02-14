import os
import labelme
import numpy as np
from PIL import Image
import glob

# Directorios
json_dir = "dataset/labels/"
output_dir = "dataset/masks/"
os.makedirs(output_dir, exist_ok=True)

# Obtener lista de archivos JSON
json_files = glob.glob(json_dir + "*.json")

for json_path in json_files:
    # Cargar anotaciones
    with open(json_path, "r") as f:
        data = labelme.utils.load_json(f)

    # Convertir a máscara
    img = labelme.utils.shapes_to_label(
        img_shape=(data["imageHeight"], data["imageWidth"]),
        shapes=data["shapes"]
    )

    # Guardar como PNG
    mask_path = os.path.join(output_dir, os.path.basename(json_path).replace(".json", ".png"))
    Image.fromarray(img.astype(np.uint8)).save(mask_path)

print("Conversión terminada.")