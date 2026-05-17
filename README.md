# Stacked RNN — Análisis de Sentimiento en IMDB

Estudio comparativo entre arquitecturas **LSTM** y **GRU** apiladas para clasificación binaria de sentimiento en reseñas de películas (IMDB). Proyecto del curso de Aprendizaje Profundo, 8° semestre — IIMAS, UNAM.

## Resultados

| Métrica | LSTM | GRU |
|---------|------|-----|
| Accuracy | 85.12% | **86.46%** |
| Precision | 85.29% | 85.34% |
| Recall | 84.88% | **88.03%** |
| F1-Score | 85.08% | **86.67%** |
| Tiempo total | 184.1s | 1042.0s |
| Épocas (early stop) | 10/10 | 10/10 |

Ambos modelos mantienen gradientes estables (normas promedio: LSTM 0.079, GRU 0.052) sin evidencia de vanishing ni exploding gradients.

## Estructura del Proyecto

```
StackedRNN/
├── src/                    # Código fuente
│   ├── config.py           # Hiperparámetros centralizados
│   ├── data.py             # Descarga IMDB, limpieza, vocabulario, DataLoaders
│   ├── model.py            # SentimentRNN (LSTM/GRU configurable)
│   ├── train.py            # Loop de entrenamiento con early stopping
│   ├── evaluate.py         # Evaluación y métricas
│   ├── visualize.py        # Generación de plots
│   ├── reflexiones.py      # Generación de REFLEXIONES.md
│   └── main.py             # Pipeline completo (punto de entrada)
├── tests/                  # Tests unitarios y property-based (hypothesis)
├── doc/                    # Reporte técnico en LaTeX
│   ├── sections/           # 8 secciones .tex
│   └── build/              # PDF compilado (gitignored)
├── outputs/                # Figuras generadas (gitignored)
├── checkpoints/            # Mejores modelos .pt (gitignored)
├── conftest.py             # Configuración pytest
└── requirements.txt        # Dependencias pinned
```

## Arquitectura del Modelo

```
Embedding(25000, 128) → Dropout(0.5) → RNN×2(256, dropout=0.3) → Linear(256, 1)
```

- **Embedding**: 25,000 tokens → 128 dimensiones
- **RNN apilada**: 2 capas, 256 unidades ocultas (LSTM o GRU)
- **Regularización**: dropout 0.5 post-embedding, 0.3 entre capas
- **Clasificación**: capa lineal 256→1, BCEWithLogitsLoss
- **Inicialización**: bias forget gate LSTM a 1.0 (Jozefowicz et al., 2015)

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/emilianodesu/sentiment-analysis-stacked-rnn.git
cd sentiment-analysis-stacked-rnn

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -e ".[dev]"
```

## Uso

### Ejecutar el pipeline completo

```bash
python -m src.main
```

Esto ejecuta secuencialmente:
1. Descarga y preprocesamiento del dataset IMDB
2. Entrenamiento del modelo LSTM
3. Entrenamiento del modelo GRU
4. Evaluación comparativa en el conjunto de test
5. Generación de visualizaciones (`outputs/`)
6. Generación del documento de reflexiones

### Ejecutar tests

```bash
python -m pytest tests/ -q
```

219 tests incluyendo property-based testing con Hypothesis.

### Compilar el reporte LaTeX

```bash
cd doc
mkdir -p build
pdflatex -shell-escape -interaction=nonstopmode -output-directory=build main.tex
pdflatex -shell-escape -interaction=nonstopmode -output-directory=build main.tex
```

Requiere una distribución TeX (TeX Live / MacTeX) y Pygments (`pip install Pygments`) para el paquete `minted`.

## Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Seed | 42 |
| Vocabulario | 25,000 (freq min 2) |
| Longitud máxima | 200 tokens |
| Embedding dim | 128 |
| Hidden size | 256 |
| Capas RNN | 2 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam (β1=0.9, β2=0.999) |
| Gradient clipping | L2 norm max 5.0 |
| Early stopping | patience=5, max 10 épocas |
| Train/Val split | 20,000 / 5,000 |
| Test set | 25,000 (IMDB test split) |

## Visualizaciones

El pipeline genera tres figuras en `outputs/`:

- **training_curves.png** — Pérdida y accuracy por época (train/val)
- **confusion_matrices.png** — Matrices de confusión LSTM vs GRU
- **gradient_norms.png** — Normas L2 de gradientes por capa (primeros 100 pasos)

## Tecnologías

- Python 3.9+
- PyTorch 2.2
- Hugging Face Datasets (descarga IMDB)
- Matplotlib / Seaborn (visualización)
- scikit-learn (métricas)
- Hypothesis (property-based testing)
- LaTeX + minted (reporte técnico)

## Licencia

Proyecto académico — IIMAS, UNAM. Curso de Aprendizaje Profundo, semestre 2025-2.
