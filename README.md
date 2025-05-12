# Segmentación Semántica

Este repositorio contiene implementaciones de varios modelos de segmentación semántica utilizados para clasificar píxeles en imágenes aéreas en diferentes categorías, tales como:

- Suelo expuesto  
- Vegetación seca  
- Vegetación verde
- Agua (Considerar)

## Modelos Implementados

El proyecto incluye implementaciones de los siguientes modelos:

- **UNet**: Arquitectura clásica en forma de U para segmentación de imágenes.
- **DeepLabV3+**: Modelo avanzado de segmentación semántica que utiliza *atrous convolutions* para capturar contexto a múltiples escalas.
- **SegFormer**: Modelo reciente basado en Transformers, diseñado para segmentación semántica eficiente y precisa.

## Requisitos del Sistema

- Python 3.8 o superior  
- CUDA 12.x (para uso con GPU)  
- GPU NVIDIA con al menos 8 GB de VRAM (recomendado para entrenamiento)  

## Estructura del Proyecto

```plaintext
jupyter/
│
├── DeepLab.ipynb       # Implementación y entrenamiento de DeepLabV3+
├── SegFormer.ipynb     # Implementación y entrenamiento de SegFormer
├── UNet.ipynb          # Implementación y entrenamiento de UNet
└── metrics.ipynb       # Evaluación y comparación de los modelos

models/
│
├── UNET.pth            # Modelo UNet entrenado
├── DEEPLAB.pth         # Modelo DeepLabV3+ entrenado
└── SEGFORMER.pth       # Modelo SegFormer entrenado

best_models/
│
└── *.pth               # Mejores checkpoints de entrenamiento
