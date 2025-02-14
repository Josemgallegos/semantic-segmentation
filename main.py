import os
import subprocess
import torch

# 📌 Seleccionar dispositivo para Mac M3 o CPU/GPU normal
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# 📂 Definir rutas de los scripts
SCRIPTS = {
    "convert_json": "scripts/convert_json_to_masks.py",
    "augment_data": "scripts/augment_data.py",
    "split_data": "scripts/split_dataset.py",
    "train": "src/train.py",
    "test": "src/test.py"
}

def run_script(script_name):
    """ Ejecuta un script de Python y maneja errores """
    print(f"\n🚀 Ejecutando {script_name}...\n")
    result = subprocess.run(["python", SCRIPTS[script_name]], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {script_name} completado con éxito.\n")
    else:
        print(f"Error en {script_name}:\n{result.stderr}")
        exit(1)  # Termina el programa si hay un error

def main():
    """ Ejecuta todo el pipeline del proyecto """
    print("\n**INICIANDO PIPELINE DE SEGMENTACIÓN SEMÁNTICA**\n")
    
    # Paso 1: Convertir JSON a máscaras
    # run_script("convert_json")

    # Paso 2: Aumentación de datos
    # run_script("augment_data")

    # Paso 3: Dividir dataset en train/val/test
    # run_script("split_data")

    # Paso 4: Entrenar el modelo U-Net
    run_script("train")

    # Paso 5: Probar el modelo en imágenes de test
    run_script("test")

    print("\n🎉 **PROCESO COMPLETADO!** 🚀\n")

if __name__ == "__main__":
    main()