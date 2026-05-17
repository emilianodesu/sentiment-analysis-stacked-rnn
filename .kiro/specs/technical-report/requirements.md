# Requirements Document

## Introduction

Este documento especifica los requisitos para la generación del reporte técnico en LaTeX del proyecto de Análisis de Sentimientos con Redes Neuronales Recurrentes Apiladas (Stacked RNN). El reporte documenta un experimento comparativo entre arquitecturas LSTM y GRU aplicadas a clasificación binaria de sentimiento en reseñas de IMDB, como parte del curso de Aprendizaje Profundo (8o semestre) en el IIMAS, UNAM.

El reporte se integra a la estructura LaTeX existente en `doc/`, utilizando el preámbulo, configuración y formato ya definidos, y debe redactarse en español siguiendo las convenciones académicas de la universidad.

## Glossary

- **Generador_Reporte**: Sistema de generación del documento LaTeX que produce las secciones del reporte técnico a partir de los resultados experimentales y el código fuente del proyecto.
- **Sección_LaTeX**: Archivo `.tex` individual dentro de `doc/sections/` que contiene una sección del reporte.
- **LSTM**: Long Short-Term Memory, arquitectura de red neuronal recurrente con compuertas de entrada, olvido y salida.
- **GRU**: Gated Recurrent Unit, arquitectura de red neuronal recurrente con compuertas de reset y actualización.
- **Stacked_RNN**: Arquitectura de RNN con múltiples capas recurrentes apiladas.
- **REFLEXIONES.md**: Documento generado automáticamente con el análisis experimental de 8 preguntas sobre el comportamiento de los modelos.
- **Estructura_Doc**: Estructura de directorios y archivos LaTeX existente en `doc/` (main.tex, preamble.tex, config.tex, format.tex).

## Requirements

### Requirement 1: Sección de Introducción y Marco Teórico

**User Story:** Como estudiante, quiero una sección de introducción que presente el problema de análisis de sentimientos y el marco teórico de las RNN apiladas, para que el lector comprenda el contexto y la motivación del experimento.

#### Acceptance Criteria

1. WHEN el Generador_Reporte produce la sección de introducción, THE Sección_LaTeX SHALL contener una descripción del problema de clasificación binaria de sentimiento que incluya: la definición de la tarea (clasificar reseñas como positivas o negativas), el dominio de aplicación (reseñas de películas IMDB), y la motivación para usar redes recurrentes en texto secuencial.
2. WHEN el Generador_Reporte produce el marco teórico, THE Sección_LaTeX SHALL incluir las 5 ecuaciones de la celda LSTM: compuerta de olvido (f_t), compuerta de entrada (i_t), candidato de celda (c̃_t), estado de celda (c_t), compuerta de salida (o_t) y estado oculto (h_t).
3. WHEN el Generador_Reporte produce el marco teórico, THE Sección_LaTeX SHALL incluir las 4 ecuaciones de la celda GRU: compuerta de reset (r_t), compuerta de actualización (z_t), candidato oculto (h̃_t) y estado oculto (h_t).
4. WHEN el Generador_Reporte produce el marco teórico, THE Sección_LaTeX SHALL definir la notación matemática utilizada en las ecuaciones (matrices de pesos W y U, vectores de bias b, funciones de activación σ y tanh) antes de presentar las ecuaciones de LSTM y GRU.
5. WHEN el Generador_Reporte produce el marco teórico, THE Sección_LaTeX SHALL explicar el concepto de apilamiento de capas recurrentes (Stacked RNN) describiendo cómo la salida de una capa alimenta la entrada de la siguiente y su ventaja para aprender representaciones jerárquicas de mayor nivel de abstracción.
6. WHEN el Generador_Reporte produce el marco teórico, THE Sección_LaTeX SHALL compilar sin errores con `pdflatex` utilizando los paquetes `amsmath` y `amssymb` del preámbulo existente, verificando que todos los entornos de ecuación (`equation`, `aligned`) y símbolos matemáticos estén correctamente cerrados.

### Requirement 2: Sección de Arquitectura del Modelo

**User Story:** Como estudiante, quiero una sección que describa la arquitectura implementada del modelo SentimentRNN, para que el lector comprenda la estructura exacta utilizada en el experimento.

#### Acceptance Criteria

1. WHEN el Generador_Reporte produce la sección de arquitectura, THE Sección_LaTeX SHALL describir la capa de embedding con vocabulario de 25,000 tokens y dimensión 128.
2. WHEN el Generador_Reporte produce la sección de arquitectura, THE Sección_LaTeX SHALL describir la configuración de las capas recurrentes apiladas (2 capas, 256 unidades ocultas por capa) indicando que el modelo soporta tanto LSTM como GRU como tipo de capa recurrente.
3. WHEN el Generador_Reporte produce la sección de arquitectura, THE Sección_LaTeX SHALL documentar la estrategia de regularización (dropout de 0.5 aplicado después de la capa de embedding, dropout de 0.3 entre capas RNN apiladas).
4. WHEN el Generador_Reporte produce la sección de arquitectura, THE Sección_LaTeX SHALL describir la capa de clasificación lineal (256 → 1) que recibe el estado oculto de la última capa recurrente en el último paso temporal, produciendo un logit para clasificación binaria.
5. WHEN el Generador_Reporte produce la sección de arquitectura, THE Sección_LaTeX SHALL mencionar la inicialización del bias de la compuerta de olvido del LSTM a 1.0 (Jozefowicz et al., 2015).
6. WHEN el Generador_Reporte produce la sección de arquitectura, THE Sección_LaTeX SHALL incluir una tabla que resuma la arquitectura completa del modelo listando cada capa con su tipo, dimensiones de entrada y salida, y número de parámetros entrenables.

### Requirement 3: Sección de Metodología de Entrenamiento

**User Story:** Como estudiante, quiero una sección que detalle la metodología de entrenamiento utilizada, para que el experimento sea reproducible.

#### Acceptance Criteria

1. WHEN el Generador_Reporte produce la sección de metodología, THE Sección_LaTeX SHALL documentar el preprocesamiento de datos incluyendo: dataset IMDB (25,000 muestras originales de entrenamiento divididas en 20,000 de entrenamiento y 5,000 de validación mediante permutación aleatoria con seed=42), limpieza de texto (remoción de etiquetas HTML, conversión a minúsculas, eliminación de caracteres no alfabéticos), tokenización por espacios, construcción de vocabulario (máximo 25,000 palabras, frecuencia mínima de 2), y truncamiento/padding a longitud máxima de 200 tokens.
2. WHEN el Generador_Reporte produce la sección de metodología, THE Sección_LaTeX SHALL especificar el optimizador Adam con learning rate de 0.001 y parámetros por defecto (β1=0.9, β2=0.999).
3. WHEN el Generador_Reporte produce la sección de metodología, THE Sección_LaTeX SHALL documentar el mecanismo de early stopping que monitorea la pérdida de validación, con patience de 5 épocas sin mejora y un máximo de 10 épocas de entrenamiento.
4. WHEN el Generador_Reporte produce la sección de metodología, THE Sección_LaTeX SHALL especificar el gradient clipping con norma L2 máxima de 5.0 aplicado después del cálculo de gradientes y antes del paso del optimizador.
5. WHEN el Generador_Reporte produce la sección de metodología, THE Sección_LaTeX SHALL indicar la semilla de reproducibilidad (seed=42 aplicada a Python random, NumPy, PyTorch CPU y CUDA, con cuDNN determinístico) y el tamaño de batch (64).
6. WHEN el Generador_Reporte produce la sección de metodología, THE Sección_LaTeX SHALL describir la función de pérdida utilizada (Binary Cross-Entropy con logits) y el umbral de clasificación de 0.5 aplicado a la salida sigmoide.
7. WHEN el Generador_Reporte produce la sección de metodología, THE Sección_LaTeX SHALL documentar el conjunto de prueba (25,000 muestras del split de test de IMDB) y el registro de normas L2 de gradientes por capa durante los primeros 100 pasos de entrenamiento.

### Requirement 4: Sección de Resultados Experimentales

**User Story:** Como estudiante, quiero una sección que presente los resultados comparativos entre LSTM y GRU con tablas y figuras, para que el lector pueda evaluar el rendimiento de cada arquitectura.

#### Acceptance Criteria

1. WHEN el Generador_Reporte produce la sección de resultados, THE Sección_LaTeX SHALL incluir una tabla comparativa con accuracy, precision, recall y F1-score para ambos modelos, mostrando los valores con 2 decimales de precisión en formato porcentual.
2. WHEN el Generador_Reporte produce la sección de resultados, THE Sección_LaTeX SHALL incluir la figura `training_curves.png` con las curvas de entrenamiento de ambos modelos, referenciada mediante `\includegraphics` con `\caption` y `\label{fig:training_curves}`.
3. WHEN el Generador_Reporte produce la sección de resultados, THE Sección_LaTeX SHALL incluir la figura `confusion_matrices.png` con las matrices de confusión, referenciada mediante `\includegraphics` con `\caption` y `\label{fig:confusion_matrices}`.
4. WHEN el Generador_Reporte produce la sección de resultados, THE Sección_LaTeX SHALL incluir la figura `gradient_norms.png` con el análisis de flujo de gradientes, referenciada mediante `\includegraphics` con `\caption` y `\label{fig:gradient_norms}`.
5. WHEN el Generador_Reporte produce la sección de resultados, THE Sección_LaTeX SHALL reportar los tiempos de entrenamiento (LSTM: 71.9s en 4 épocas, GRU: 994.0s en 10 épocas) y el número de épocas entrenadas antes del early stopping.
6. WHEN el Generador_Reporte produce la sección de resultados, THE Sección_LaTeX SHALL utilizar el paquete `booktabs` para las tablas (con `\toprule`, `\midrule`, `\bottomrule`) y el entorno `figure` con posicionamiento `[H]` (requiere paquete `float`) para las figuras.

### Requirement 5: Sección de Análisis de Gradientes

**User Story:** Como estudiante, quiero una sección dedicada al análisis del flujo de gradientes, para demostrar la estabilidad del entrenamiento y la efectividad del gradient clipping.

#### Acceptance Criteria

1. WHEN el Generador_Reporte produce la sección de gradientes, THE Sección_LaTeX SHALL reportar las normas promedio de gradientes (LSTM: 0.018461, GRU: 0.061758) calculadas durante los primeros 100 pasos de entrenamiento, e indicar que ambas normas se mantienen por debajo del umbral de clipping configurado (max_norm=5.0).
2. WHEN el Generador_Reporte produce la sección de gradientes, THE Sección_LaTeX SHALL incluir una comparación cuantitativa que demuestre la ausencia de vanishing gradients (normas promedio superiores a 1×10⁻⁷) y de exploding gradients (normas promedio inferiores al umbral de clipping de 5.0) en ambos modelos, referenciando el gradient clipping como mecanismo de prevención.
3. WHEN el Generador_Reporte produce la sección de gradientes, THE Sección_LaTeX SHALL explicar el rol de las compuertas internas (3 en LSTM: entrada, olvido, salida; 2 en GRU: reset, actualización) en la estabilidad del flujo de gradientes, describiendo cómo cada tipo de compuerta contribuye a mantener la propagación del gradiente a través del tiempo.
4. WHEN el Generador_Reporte produce la sección de gradientes, THE Sección_LaTeX SHALL referenciar la figura `gradient_norms.png` mediante el entorno `figure` con `\label` y `\caption` descriptivo, como evidencia visual de las normas L2 por capa durante los primeros 100 pasos.
5. WHEN el Generador_Reporte produce la sección de gradientes, THE Sección_LaTeX SHALL incluir un resumen de las normas de gradiente desglosadas por capa (weight_ih, weight_hh, bias) para ambos modelos, presentado en formato de tabla o lista con valores de media y máximo.

### Requirement 6: Sección de Reflexiones y Discusión

**User Story:** Como estudiante, quiero una sección de reflexiones que integre el análisis detallado del archivo REFLEXIONES.md, para cumplir con el requisito del curso de responder las 8 preguntas de análisis.

#### Acceptance Criteria

1. WHEN el Generador_Reporte produce la sección de reflexiones, THE Sección_LaTeX SHALL incluir las 8 preguntas de reflexión como subsecciones individuales (`\subsection`), cada una con su pregunta como título y su respuesta correspondiente extraída de REFLEXIONES.md.
2. WHEN el Generador_Reporte produce la sección de reflexiones, THE Sección_LaTeX SHALL incluir para cada uno de los 8 temas (efecto del número de capas, impacto del dropout, comparación de convergencia, flujo de gradientes, longitud de secuencia, capa de embedding, early stopping, y trade-offs LSTM vs GRU) al menos una comparación explícita entre los resultados de LSTM y GRU.
3. WHEN el Generador_Reporte produce la sección de reflexiones, THE Sección_LaTeX SHALL incluir al menos un valor numérico proveniente de los resultados experimentales (accuracy, F1-score, tiempo de entrenamiento, norma de gradientes, o épocas entrenadas) en cada una de las 8 respuestas.
4. WHEN el Generador_Reporte produce la sección de reflexiones, THE Sección_LaTeX SHALL presentar una conclusión general que sintetice: el modelo con mejor rendimiento según F1-score, la efectividad de la regularización (dropout y early stopping), y la estabilidad del flujo de gradientes.
5. IF el archivo REFLEXIONES.md no existe o no contiene las 8 secciones esperadas, THEN THE Generador_Reporte SHALL emitir un mensaje de error indicando las secciones faltantes y no generar la sección de reflexiones de forma parcial.

### Requirement 7: Sección de Código Fuente

**User Story:** Como estudiante, quiero incluir fragmentos relevantes del código fuente en el reporte, para demostrar la implementación y facilitar la reproducibilidad.

#### Acceptance Criteria

1. WHEN el Generador_Reporte produce la sección de código, THE Sección_LaTeX SHALL incluir únicamente fragmentos relevantes del código fuente que apoyen la explicación de cada sección, seleccionando las porciones que ilustren los conceptos clave (definición del modelo, bucle de entrenamiento, gradient clipping, early stopping) en lugar de archivos completos.
2. WHEN el Generador_Reporte selecciona fragmentos de código, THE Sección_LaTeX SHALL limitar cada fragmento a entre 15 y 40 líneas, enfocándose en una sola responsabilidad funcional por fragmento (e.g., definición de la clase del modelo, configuración del optimizador, lógica de gradient clipping).
3. WHEN el Generador_Reporte produce cualquier sección del reporte, THE Sección_LaTeX SHALL incluir fragmentos de código inline donde estos ayuden a explicar el concepto presentado en esa sección, utilizando el entorno `minted` con sintaxis de Python.
4. WHEN el Generador_Reporte produce la sección de código, THE Sección_LaTeX SHALL utilizar la configuración de `minted` definida en el preámbulo (linenos, breaklines, frame=single, fontsize=\small).
5. IF un bloque de código fuente excede 40 líneas de longitud, THEN THE Generador_Reporte SHALL dividir el código en fragmentos más cortos, separados por un párrafo de texto de entre 1 y 3 oraciones que describa la funcionalidad del fragmento siguiente.
6. WHEN el Generador_Reporte incluye fragmentos de código, THE Sección_LaTeX SHALL presentar cada fragmento con un comentario de encabezado que identifique la sección funcional (e.g., `% --- Configuración del optimizador ---`) y una breve descripción textual que explique por qué ese fragmento es relevante para la discusión.

### Requirement 8: Integración con la Estructura LaTeX Existente

**User Story:** Como estudiante, quiero que las nuevas secciones se integren correctamente con la estructura LaTeX existente del proyecto, para que el documento compile sin errores y mantenga el formato institucional.

#### Acceptance Criteria

1. WHEN el Generador_Reporte crea archivos de sección, THE Sección_LaTeX SHALL producir los siguientes archivos en `doc/sections/`, cada uno correspondiente a una sección del reporte:
   - `doc/sections/introduccion.tex` — Introducción y marco teórico
   - `doc/sections/arquitectura.tex` — Arquitectura del modelo
   - `doc/sections/metodologia.tex` — Metodología de entrenamiento
   - `doc/sections/resultados.tex` — Resultados experimentales
   - `doc/sections/gradientes.tex` — Análisis de gradientes
   - `doc/sections/reflexiones.tex` — Reflexiones y discusión (8 preguntas)
   - `doc/sections/codigo.tex` — Fragmentos de código fuente
   - `doc/sections/referencias.tex` — Bibliografía/referencias
2. WHEN el Generador_Reporte actualiza el documento principal, THE Estructura_Doc SHALL sobrescribir `doc/main.tex` con los `\input{sections/...}` correspondientes entre los comentarios `% -------------------- SECCIONES --------------------` y `\end{document}`, ordenados según la secuencia: introduccion, arquitectura, metodologia, resultados, gradientes, reflexiones, codigo, referencias.
3. WHEN el documento completo se compila con `pdflatex -shell-escape main.tex` ejecutado desde el directorio `doc/`, THE Estructura_Doc SHALL generar un PDF con código de salida 0 y sin líneas que contengan el prefijo `!` en la salida del compilador (los warnings son aceptables).
4. WHEN el Generador_Reporte produce cualquier sección, THE Sección_LaTeX SHALL utilizar los comandos `\titulo`, `\materia`, `\profesor`, `\semestre` y `\autorA` definidos en `config.tex` en lugar de escribir esos valores como texto literal.
5. WHEN el Generador_Reporte incluye figuras generadas por el código Python del proyecto, THE Sección_LaTeX SHALL copiar las imágenes (`training_curves.png`, `confusion_matrices.png`, `gradient_norms.png`) desde `outputs/` al directorio `doc/img/` y referenciarlas con la ruta relativa `img/<nombre_archivo>` dentro de los comandos `\includegraphics`.
6. WHEN el Generador_Reporte realiza la limpieza de la plantilla anterior, THE Estructura_Doc SHALL eliminar el archivo `doc/sections/ejercicio_01.tex` que pertenece a un proyecto anterior y no es relevante para el reporte de Stacked RNN.
7. WHEN el Generador_Reporte realiza la limpieza de la plantilla anterior, THE Estructura_Doc SHALL eliminar los directorios de artefactos de compilación (`doc/build/` y `doc/_minted/`) para partir de un estado limpio.
8. WHEN el Generador_Reporte realiza la limpieza de la plantilla anterior, THE Estructura_Doc SHALL preservar sin modificación los archivos: `doc/preamble.tex` (paquetes amsmath, amssymb, graphicx, float, booktabs, minted, fancyvrb, babel spanish), `doc/format.tex` (geometry, hyperref, fancyhdr), `doc/config.tex` (metadatos del curso), y `doc/img/logo.png` (logo institucional para la portada).

### Requirement 9: Formato y Estilo Académico

**User Story:** Como estudiante, quiero que el reporte siga las convenciones de formato académico de la UNAM, para cumplir con los estándares del curso.

#### Acceptance Criteria

1. THE Generador_Reporte SHALL producir todo el contenido textual en idioma español.
2. WHEN el Generador_Reporte produce ecuaciones, THE Sección_LaTeX SHALL utilizar los entornos `equation`, `aligned` o `gather` de `amsmath`, numerando todas las ecuaciones principales (una numeración por entorno `equation` o `gather`) y dejando sin numeración únicamente las líneas auxiliares dentro de entornos `aligned`.
3. WHEN el Generador_Reporte produce tablas, THE Sección_LaTeX SHALL incluir un `\caption` que identifique el contenido comparado y las variables presentadas, una etiqueta `\label` con prefijo `tab:`, y al menos una referencia cruzada `\ref` en el texto que precede a la tabla.
4. WHEN el Generador_Reporte produce figuras, THE Sección_LaTeX SHALL incluir un `\caption` que identifique el contenido visualizado y los modelos representados, una etiqueta `\label` con prefijo `fig:`, y al menos una referencia cruzada `\ref` en el texto que precede a la figura.
5. WHEN el Generador_Reporte produce la sección de referencias, THE Sección_LaTeX SHALL incluir como mínimo las siguientes fuentes: Hochreiter & Schmidhuber (1997) para LSTM, Cho et al. (2014) para GRU, y Jozefowicz et al. (2015) para inicialización de bias, utilizando el entorno `thebibliography` con entradas `\bibitem` y citadas en el texto mediante `\cite`.
6. WHEN el Generador_Reporte produce ecuaciones referenciadas en el texto, THE Sección_LaTeX SHALL utilizar `\label` con prefijo `eq:` y referenciarlas mediante `\eqref` en el cuerpo del documento.
