# Implementation Plan: Technical Report (Reporte Técnico LaTeX)

## Overview

Este plan implementa la generación del reporte técnico en LaTeX siguiendo el pipeline de 5 pasos definido en el diseño: (1) limpieza de plantilla anterior, (2) copia de imágenes, (3) generación de 8 secciones `.tex`, (4) reestructuración de `main.tex`, y (5) compilación y verificación. Cada sección `.tex` es una tarea independiente, pero todas dependen de que la limpieza e imágenes estén completas primero.

## Tasks

- [x] 1. Limpieza de plantilla anterior y copia de imágenes
  - [x] 1.1 Eliminar artefactos de la plantilla anterior
    - Eliminar `doc/sections/ejercicio_01.tex`
    - Eliminar recursivamente `doc/build/`
    - Eliminar recursivamente `doc/_minted/`
    - Verificar que se preservan sin modificación: `doc/preamble.tex`, `doc/format.tex`, `doc/config.tex`, `doc/img/logo.png`
    - _Requirements: 8.6, 8.7, 8.8_

  - [x] 1.2 Copiar imágenes de resultados a `doc/img/`
    - Copiar `outputs/training_curves.png` → `doc/img/training_curves.png`
    - Copiar `outputs/confusion_matrices.png` → `doc/img/confusion_matrices.png`
    - Copiar `outputs/gradient_norms.png` → `doc/img/gradient_norms.png`
    - Verificar que las 3 imágenes existen en `doc/img/`
    - _Requirements: 8.5_

- [x] 2. Generar secciones de marco teórico y arquitectura
  - [x] 2.1 Crear `doc/sections/introduccion.tex` — Introducción y Marco Teórico
    - Escribir introducción al problema de clasificación binaria de sentimiento en IMDB
    - Definir notación matemática (matrices W, U, bias b, activaciones σ, tanh)
    - Incluir 5 ecuaciones LSTM en entorno `equation`+`aligned` con `\label{eq:...}`
    - Incluir 4 ecuaciones GRU con `\label{eq:...}`
    - Explicar concepto de Stacked RNN (apilamiento de capas)
    - Usar `\eqref` para referenciar ecuaciones en el texto
    - Todo el contenido en español
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 9.1, 9.2, 9.6_

  - [x] 2.2 Crear `doc/sections/arquitectura.tex` — Arquitectura del Modelo
    - Describir capa de embedding (25,000 tokens, dimensión 128)
    - Describir capas recurrentes apiladas (2 capas, 256 unidades ocultas, LSTM/GRU)
    - Documentar estrategia de regularización (dropout 0.5 post-embedding, 0.3 entre capas)
    - Describir capa de clasificación lineal (256 → 1)
    - Mencionar inicialización bias forget gate a 1.0 (Jozefowicz et al., 2015) con `\cite{jozefowicz2015}`
    - Incluir tabla de arquitectura con `booktabs` (`\toprule`, `\midrule`, `\bottomrule`), `\caption`, `\label{tab:...}` y `\ref` en texto
    - Incluir fragmentos de código de `src/model.py` con `minted{python}` (clase `__init__`, forward, bias init — 15-40 líneas cada uno)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 7.1, 7.2, 7.3, 7.4, 7.6, 9.3_

- [x] 3. Generar secciones de metodología y resultados
  - [x] 3.1 Crear `doc/sections/metodologia.tex` — Metodología de Entrenamiento
    - Documentar preprocesamiento (IMDB 25K→20K+5K, limpieza HTML, tokenización, vocabulario, padding 200)
    - Especificar optimizador Adam (lr=0.001, β1=0.9, β2=0.999)
    - Documentar función de pérdida BCEWithLogitsLoss y umbral 0.5
    - Documentar early stopping (patience=5, max 10 épocas)
    - Especificar gradient clipping (norma L2 max 5.0)
    - Indicar semilla de reproducibilidad (seed=42) y batch_size=64
    - Documentar conjunto de prueba (25K muestras) y monitoreo de gradientes (100 pasos)
    - Incluir fragmentos de código de `src/train.py` y `src/data.py` con `minted{python}` (optimizer, training loop, gradient clipping, early stopping, clean_text, pad_sequence)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 9.1_

  - [x] 3.2 Crear `doc/sections/resultados.tex` — Resultados Experimentales
    - Incluir tabla comparativa (accuracy, precision, recall, F1) con `booktabs`, `\caption`, `\label{tab:...}` y `\ref`
    - Reportar tiempos de entrenamiento (LSTM: 71.9s/4 épocas, GRU: 994.0s/10 épocas)
    - Incluir figura `training_curves.png` con `\includegraphics{img/training_curves}`, `\caption`, `\label{fig:training_curves}`, posicionamiento `[H]`
    - Incluir figura `confusion_matrices.png` con `\label{fig:confusion_matrices}`
    - Incluir figura `gradient_norms.png` con `\label{fig:gradient_norms}`
    - Incluir al menos una `\ref{fig:...}` en texto previo a cada figura
    - Incluir fragmento de código de `src/evaluate.py` (inferencia) con `minted{python}`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 7.1, 7.3, 9.1, 9.3, 9.4_

- [x] 4. Generar secciones de análisis y reflexiones
  - [x] 4.1 Crear `doc/sections/gradientes.tex` — Análisis de Gradientes
    - Reportar normas promedio (LSTM: 0.018461, GRU: 0.061758)
    - Demostrar ausencia de vanishing (>1×10⁻⁷) y exploding (<5.0) gradients
    - Explicar rol de compuertas (LSTM: 3, GRU: 2) en estabilidad de gradientes
    - Incluir tabla/lista de normas por capa (weight_ih, weight_hh, bias) con media y máximo
    - Referenciar figura `gradient_norms.png` con `\label` y `\caption`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 9.1, 9.3, 9.4_

  - [x] 4.2 Crear `doc/sections/reflexiones.tex` — Reflexiones y Discusión
    - Verificar que `outputs/REFLEXIONES.md` contiene las 8 secciones esperadas
    - Crear 8 `\subsection` con las preguntas como título y respuestas del archivo
    - Incluir comparación explícita LSTM vs GRU en cada subsección
    - Incluir al menos un valor numérico experimental en cada respuesta
    - Incluir conclusión general (mejor F1, efectividad regularización, estabilidad gradientes)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 9.1_

- [x] 5. Generar secciones de código y referencias
  - [x] 5.1 Crear `doc/sections/codigo.tex` — Fragmentos de Código Fuente
    - Seleccionar fragmentos relevantes de `src/model.py`, `src/train.py`, `src/data.py`, `src/evaluate.py` (15-40 líneas cada uno)
    - Usar entorno `minted{python}` con configuración del preámbulo
    - Incluir comentario de encabezado (`% --- Descripción ---`) antes de cada bloque
    - Incluir párrafo descriptivo de 1-3 oraciones antes de cada fragmento
    - Dividir fragmentos >40 líneas con párrafo explicativo intermedio
    - _Requirements: 7.1, 7.2, 7.4, 7.5, 7.6, 9.1_

  - [x] 5.2 Crear `doc/sections/referencias.tex` — Bibliografía
    - Usar entorno `thebibliography` con `\bibitem`
    - Incluir obligatoriamente: Hochreiter & Schmidhuber (1997), Cho et al. (2014), Jozefowicz et al. (2015)
    - Agregar fuentes adicionales relevantes (Kingma & Ba 2015 para Adam, dataset IMDB)
    - Verificar que todas las `\cite` en las secciones tienen su `\bibitem` correspondiente
    - _Requirements: 9.5_

- [x] 6. Reestructurar main.tex e integración final
  - [x] 6.1 Actualizar `doc/main.tex` con los nuevos `\input{sections/...}`
    - Reemplazar la sección entre `% -------------------- SECCIONES --------------------` y `\end{document}`
    - Incluir en orden: introduccion, arquitectura, metodologia, resultados, gradientes, reflexiones, codigo, referencias
    - Separar cada sección con `\newpage`
    - Preservar portada y `\tableofcontents` sin cambios
    - _Requirements: 8.2, 8.4_

- [x] 7. Checkpoint — Compilación y verificación
  - [x] 7.1 Compilar el documento con `pdflatex -shell-escape main.tex` (dos pasadas)
    - Ejecutar primera pasada desde `doc/` para generar TOC y referencias
    - Ejecutar segunda pasada para resolver referencias cruzadas
    - Verificar código de salida 0 y ausencia de líneas con prefijo `!`
    - Verificar ausencia de warnings "undefined reference" en el log
    - Corregir cualquier error de compilación encontrado
    - _Requirements: 8.3, 9.6_

- [x] 8. Checkpoint final
  - Ensure all compilation passes, ask the user if questions arise.

## Notes

- Este proyecto genera documentos LaTeX estáticos — no aplica property-based testing ni pytest
- La verificación se realiza mediante compilación exitosa con `pdflatex -shell-escape`
- Todos los fragmentos de código usan `minted{python}` que requiere `-shell-escape` y Pygments instalado
- Las imágenes se referencian con ruta relativa `img/<nombre>` desde `doc/`
- Se requieren dos pasadas de `pdflatex` para resolver todas las referencias cruzadas
- Los valores numéricos del reporte provienen de `outputs/REFLEXIONES.md` y los resultados experimentales del proyecto

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["2.1", "2.2", "3.1", "3.2", "4.1", "4.2", "5.1", "5.2"] },
    { "id": 3, "tasks": ["6.1"] },
    { "id": 4, "tasks": ["7.1"] }
  ]
}
```
