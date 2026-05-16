# Rough Idea

## Proyecto: Análisis de Sentimientos con Redes Neuronales Recurrentes Apiladas

### Objetivo
Construir, entrenar y evaluar un sistema de Análisis de Sentimientos basado en PLN utilizando PyTorch. El modelo debe clasificar reseñas de películas del dataset IMDB como positivas o negativas usando una arquitectura de Red Neuronal Recurrente Apilada (Stacked RNN).

### Dataset
- IMDB Movie Reviews (vía Hugging Face `datasets` library o archivos tabulares locales)
- Clasificación binaria: positiva (1) / negativa (0)
- Subconjuntos requeridos: entrenamiento, validación y prueba

### Pipeline Completo Requerido

#### Paso 1: Adquisición y Limpieza de Datos
- Descargar dataset IMDB (Hugging Face `datasets` o formato tabular)
- Función de limpieza de texto que incluya:
  - Eliminación de etiquetas HTML (regex para `<br />`, etc.)
  - Conversión a minúsculas
  - Eliminación de puntuación y caracteres no alfanuméricos
  - Tokenización (basada en espacios o con NLTK)

#### Paso 2: Procesamiento y Preparación de Secuencias
- Construcción de vocabulario bidireccional (`word2idx` / `idx2word`):
  - Limitar tamaño máximo del vocabulario
  - Filtrar palabras con frecuencia menor a N
  - Tokens especiales: `<PAD>` (índice 0), `<UNK>` (índice 1)
  - Vocabulario construido SOLO con datos de entrenamiento (evitar data leakage)
- Padding y Truncating:
  - Definir longitud máxima de secuencia
  - Truncar reseñas más largas
  - Rellenar reseñas más cortas con `<PAD>`
- Dataset y DataLoaders:
  - Clase personalizada heredando de `torch.utils.data.Dataset`
  - DataLoaders para train, val y test con batching y shuffling

#### Paso 3: Definición del Modelo
- Clase PyTorch heredando de `nn.Module` con:
  - `nn.Embedding`: representaciones densas de tokens
  - Capa recurrente apilada (`nn.LSTM` o `nn.GRU`) con `num_layers >= 2`
  - `nn.Dropout` después del embedding y entre capas recurrentes
  - `nn.Linear` final para clasificación binaria (salida única)
- Desplegar estructura del modelo con `print(model)`
- Imprimir desglose de parámetros entrenables por capa (`model.named_parameters()`)

#### Paso 4: Configuración, Entrenamiento y Validación
- Función de pérdida: `nn.BCEWithLogitsLoss` (estabilidad numérica)
- Optimizador: Adam
- Ciclo de entrenamiento por época:
  - `optimizer.zero_grad()`
  - Forward pass
  - `loss.backward()`
  - Gradient Clipping (`nn.utils.clip_grad_norm_`)
  - `optimizer.step()`
- Evaluación en validación al final de cada época
- Guardar automáticamente el mejor modelo según validación (`torch.save()`)

#### Paso 5: Pruebas y Métricas
- Cargar mejores pesos y poner modelo en `.eval()`
- Desactivar gráfica computacional (`torch.no_grad()`)
- Generar predicciones sobre test set completo
- Reporte de métricas: Precision, Recall, F1-Score, Accuracy

#### Paso 6: Visualizaciones
1. **Curvas de entrenamiento**: Train vs Val Loss y Train vs Val Accuracy por época
2. **Matriz de confusión**: Heatmap con TP, TN, FP, FN
3. **Monitoreo de gradientes**: Norma L2 de gradientes por capa (embedding, LSTM capa 1, LSTM capa 2, Linear) durante primeros pasos de entrenamiento

### Preguntas de Reflexión (a responder en documentación)
- Impacto del filtrado de vocabulario en generalización y tokens OOV
- Efectos del padding sin masking en gradientes y estado oculto
- LSTM vs GRU: diferencias en compuertas, tiempos de entrenamiento y desempeño
- Qué aprende la segunda capa recurrente vs la primera
- Ventaja de `BCEWithLogitsLoss` sobre Sigmoid + `BCELoss`
- Cómo identificar vanishing/exploding gradients en la gráfica de norma L2
- Análisis de falsos positivos en matriz de confusión y su impacto en Precision/Recall
- Identificación del punto de overfitting en curvas de entrenamiento y rol del Dropout

### Tecnologías y Dependencias
- Python 3.8+
- PyTorch
- Hugging Face `datasets`
- NumPy
- Matplotlib / Seaborn (visualizaciones)
- scikit-learn (métricas: classification_report, confusion_matrix)
- NLTK (opcional, para tokenización avanzada)
