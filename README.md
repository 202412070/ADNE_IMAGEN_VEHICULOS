# Vehículos – EDA & Pipeline ML/DL

Este notebook realiza un **flujo completo** de análisis y modelado sobre un dataset de imágenes de vehículos, organizado en subcarpetas por clase. Está estructurado en seis grandes bloques:

---

## Índice

1. [Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)  
2. [Preprocesamiento](#2-preprocesamiento)  
3. [Machine Learning Clásico](#3-machine-learning-clásico)  
4. [Deep Learning – From Scratch & Transfer Learning](#4-deep-learning--from-scratch--transfer-learning)  
5. [Object Detection con MobileNet-SSD](#5-object-detection-con-mobilenet-ssd)  
6. [Generación de Imágenes](#6-generación-de-imágenes)  

---

## 1. Exploratory Data Analysis (EDA)

- **Objetivo**: entender la distribución, el balance de clases, tamaños, histogramas de color e intensidades, outliers y clustering preliminar.
- **Secciones**:
  1. Lectura recursiva de imágenes y extracción de metadatos (ancho, alto, tamaño en bytes, clase, aspect ratio).  
  2. Guardado de un CSV `metadata.csv`.  
  3. Gráficos:  
     - Distribución de clases (barras).  
     - Histograma de anchuras/alturas y aspect ratios.  
     - Histograma de intensidades (gris y RGB).  
     - Scatter ancho vs. alto.  
  4. Mosaicos de ejemplos por clase.  
  5. Estadísticas de media/desviación por canal RGB.  
  6. Técnica extra: PCA sobre mini-imágenes (32×32) para visualizar agrupamientos de clases.

---

## 2. Preprocesamiento

- **Objetivo**: normalizar y enriquecer las imágenes antes de entrenar cualquier modelo.
- **Pasos**:
  1. **Redimensionado** a `IMG_SIZE×IMG_SIZE`.  
  2. **Equalización global** del canal Y (espacio YUV).  
  3. **CLAHE** en canal L (espacio LAB).  
  4. **Auto-crop por bordes** (opcional, detectando contornos con Canny).  
  5. **División estratificada** en train/val/test (60/20/20).  
  6. **Data Augmentation** usando `ImageDataGenerator`: rotaciones, shifts, shear, zoom, flips, plus función de preprocessing que incluye CLAHE + normalización.  
  7. Visualización de un batch aumentado y ejemplos de auto-crop.

---

## 3. Machine Learning Clásico

- **Objetivo**: entrenar un clasificador tradicional (SVM) con features “hand-crafted”.
- **Features extraídas**:
  1. Histograma de color RGB (16 bins por canal, normalizado).  
  2. HOG (Histogram of Oriented Gradients).  
  3. LBP (Local Binary Patterns, uniforme).  
- **Pipeline**:
  - `StandardScaler` → `SVC(kernel='rbf')`.  
  - Validación en set de validación:  
    - Accuracy, classification report (precision, recall=sensitivity, f1).  
    - Matriz de confusión (heatmap).  
  - Evaluación final en test set con métricas y matriz.

---

## 4. Deep Learning – From Scratch & Transfer Learning

- **4.1 CNN From Scratch**  
  - Arquitectura sencilla de 3 bloques Conv→Pool → Dense → Dropout → Softmax.  
  - Entrenamiento 10 epochs, curvas de `train/val` accuracy & loss.

- **4.2 Transfer Learning (EfficientNetB0)**  
  - Feature-extraction (base congelado) + capa GAP + Dropout + Softmax.  
  - Entrenamiento 5 epochs.  
  - Fine-tuning: descongelar últimas 20 capas, tasa de learning más baja, 10 epochs.

---

## 5. Object Detection con MobileNet-SSD

- **Modelo**: MobileNet-SSD entrenado en VOC.  
- **Descarga** automática de `deploy.prototxt` y `mobilenet_iter_73000.caffemodel`.  
- **OpenCV DNN** para crear el blob, forward pass y dibujar bounding boxes + etiquetas con confianza > 0.5.  
- Se visualizan ejemplos sobre el set de test.

---

## 6. Generación de Imágenes

- **6.1 Neural Style Transfer (Keras)**  
  - Basado en VGG16: losses de contenido, estilo y TV.  
  - Optimización L-BFGS para combinar una imagen de contenido (vehículo) y una de estilo externa.

- **6.2 DCGAN (PyTorch)**  
  - Generator: series de `ConvTranspose2d` con BatchNorm y ReLU → Tanh.  
  - Discriminator: series de `Conv2d` con LeakyReLU → Sigmoid.  
  - Entrenamiento adversarial: BCELoss, Adam.  
  - Visualización de muestras generadas tras cada epoch (grid 64 muestras).

---

### Cómo arrancar

1. Crea un entorno Conda con todas las dependencias (OpenCV, TensorFlow, PyTorch, scikit-learn, etc.).  
2. Coloca tus imágenes en `./data/vehicle/Vehicles/<clase>/*.jpg`.  
3. Abre el notebook, ajusta `ROOT_DIR`, `IMG_SIZE`, `BATCH_SIZE` si lo deseas.  
4. Ejecuta celda a celda en orden.  

