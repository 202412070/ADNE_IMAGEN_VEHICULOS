# Vehículos – EDA & Pipeline ML/DL

Dataset obtenido de [Kaggle](https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification).

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

### 4.1 CNN From Scratch

- Implementación manual de una red convolucional simple (`SimpleCNN`) en PyTorch.  
- Arquitectura:
  - **2 capas convolucionales** con activaciones `ReLU` y operaciones de `MaxPool2d`.
  - **Capa densa final** compuesta por `Flatten`, `Linear` y `ReLU`.
- Función de pérdida utilizada: `CrossEntropyLoss`.
- Visualización de curvas de entrenamiento y validación para `accuracy` y `loss`.


## 4.2 Transfer Learning + Fine-tuning (ResNet50)
- Se utiliza un modelo ResNet50 preentrenado en ImageNet.
- Se congelan las capas base y se reemplaza la capa final para ajustarla al número de clases del dataset.
- Se entrena únicamente la nueva capa final utilizando Adam como optimizador.
- Se muestran curvas de entrenamiento y validación (accuracy y loss) para analizar el rendimiento del modelo.


---

## 5. Object Detection con MobileNet-SSD

- **Modelo**: MobileNet-SSD entrenado en VOC.  
- **Descarga** automática de `deploy.prototxt` y `mobilenet_iter_73000.caffemodel`.  
- **OpenCV DNN** para crear el blob, forward pass y dibujar bounding boxes + etiquetas con confianza > 0.5.  
- Se visualizan ejemplos sobre el set de test.

---

## 6. Generación de Imágenes

- Este script utiliza el modelo preentrenado **Stable Diffusion v1.5** para generar imágenes sintéticas representativas de distintas clases de vehículos. La generación se realiza mediante *text-to-image* condicional: a partir de una descripción textual (prompt) basada en el nombre de la clase, se produce una imagen realista que refleja esa categoría.
- Usa el modelo `runwayml/stable-diffusion-v1-5` de Hugging Face.
- Guarda las imágenes generadas en el directorio local `outputs/generated_images`.

---

## 7. OPCIONAL:

### 7.1. Image Captioning
  - Generación automática de descripciones para imágenes usando el modelo BLIP de Salesforce (`blip-image-captioning-base`).
  - El modelo interpreta el contenido visual y devuelve una frase descriptiva.
 
### 7.2. Text Extraction (Image-to-Text)
  - En esta celda se utiliza la librería `pytesseract`, un wrapper de Python para Tesseract OCR, con el objetivo de extraer texto desde una imagen. Este tipo de técnica es útil dentro del análisis de datos no estructurados cuando se trabaja con imágenes que contienen texto (por ejemplo, carteles, matrículas, documentos escaneados, etc.).
  - **NOTA**: Será necesaria primero la descarga del programa Tesseract y se recomienda instalarlo en la misma ruta que se muestra en el código.

### Cómo arrancar

1. Crea un entorno a partir del fichero `requirements_imagen.txt`, dentro de la carpeta **requirements**.
2. Instala Tesseract (ejecutable en la carpeta **requirements**, obtenido de [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki))
3. Descargar el dataset desde [Kaggle](https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification).
4. Coloca tus imágenes en `./data/vehicle/Vehicles/<clase>/*.jpg`.  
5. Abre el notebook, ajusta `ROOT_DIR`, `IMG_SIZE`, `BATCH_SIZE` si lo deseas.  
6. Ejecuta celda a celda en orden.  

