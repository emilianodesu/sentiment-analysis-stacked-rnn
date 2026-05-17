# Requirements Document

## Introduction
Sistema de Análisis de Sentimientos que clasifica reseñas de películas del dataset IMDB como positivas o negativas, utilizando arquitecturas de Redes Neuronales Recurrentes Apiladas (Stacked RNN). El proyecto implementa y compara dos variantes — LSTM y GRU — bajo condiciones idénticas para evaluar sus diferencias en rendimiento, velocidad de convergencia y comportamiento de gradientes.

## Glossary
- **Stacked RNN**: Red recurrente con múltiples capas apiladas (num_layers >= 2)
- **LSTM**: Long Short-Term Memory — celda recurrente con compuertas de entrada, olvido y salida
- **GRU**: Gated Recurrent Unit — celda recurrente simplificada con compuertas de reset y update
- **Embedding**: Representación densa de tokens en un espacio vectorial continuo
- **Padding**: Relleno de secuencias cortas con tokens especiales para igualar longitudes
- **Truncating**: Corte de secuencias largas a una longitud máxima fija
- **Early Stopping**: Detención del entrenamiento cuando la métrica de validación deja de mejorar
- **Gradient Clipping**: Limitación de la norma de gradientes para evitar explosión de gradientes
- **OOV (Out-of-Vocabulary)**: Palabras no presentes en el vocabulario construido

## Requirements

### Requirement 1: Adquisición y Limpieza de Datos
**User Story:** As a data scientist, I want to download and clean the IMDB dataset, so that I have text normalizado y listo para procesamiento.

#### Acceptance Criteria
1. WHEN the data acquisition module is executed, THE system SHALL download the IMDB dataset via Hugging Face `datasets` library containing 50,000 reviews (25K train, 25K test)
2. WHEN a review contains HTML tags (e.g., `<br />`), THE system SHALL remove all HTML tags using regex
3. WHEN a review is processed, THE system SHALL convert all text to lowercase
4. WHEN a review contains punctuation or non-alphanumeric characters, THE system SHALL remove them preserving only letters and spaces
5. WHEN a review is cleaned, THE system SHALL tokenize the text using space-based splitting
6. IF the dataset download fails AND no previously downloaded dataset is available, THEN THE system SHALL raise an informative error with the failure reason

### Requirement 2: Construcción de Vocabulario
**User Story:** As a data scientist, I want to build a vocabulary from training data only, so that I avoid data leakage and have consistent token-to-index mappings.

#### Acceptance Criteria
1. WHEN the vocabulary is built, THE system SHALL use ONLY the training split (20K samples) to count word frequencies
2. WHEN a word appears fewer than 2 times (min_freq=2) in the training data, THE system SHALL exclude it from the vocabulary
3. WHEN the vocabulary exceeds 25,000 words after frequency filtering, THE system SHALL keep only the top 25,000 most frequent words
4. WHEN the vocabulary is constructed, THE system SHALL enforce index 0 for `<PAD>` and index 1 for `<UNK>` (this order is mandatory and SHALL NOT be swapped)
5. WHEN the vocabulary is built, THE system SHALL provide bidirectional mappings (`word2idx` and `idx2word`)
6. IF a word from validation or test data is not in the vocabulary, THEN THE system SHALL map it to the `<UNK>` token (index 1)

### Requirement 3: Preparación de Secuencias y DataLoaders
**User Story:** As a data scientist, I want padded/truncated sequences and efficient DataLoaders, so that I can feed batches to the model.

#### Acceptance Criteria
1. WHEN a review has more than 200 tokens, THE system SHALL truncate it to exactly 200 tokens
2. WHEN a review has fewer than 200 tokens, THE system SHALL pad it with `<PAD>` (index 0) to reach exactly 200 tokens
3. WHEN the dataset is split, THE system SHALL create 20,000 training samples, 5,000 validation samples, and 25,000 test samples
4. WHEN the train/val split is performed, THE system SHALL use a fixed random seed of 42 for reproducibility
5. WHEN DataLoaders are created, THE system SHALL use batch_size=64 with shuffling enabled for training and explicitly disabled for both validation and test DataLoaders
6. WHEN a custom Dataset class is created, THE system SHALL inherit from `torch.utils.data.Dataset` and implement `__len__` and `__getitem__`

### Requirement 4: Definición del Modelo
**User Story:** As a data scientist, I want a configurable Stacked RNN model that supports both LSTM and GRU, so that I can compare architectures fairly.

#### Acceptance Criteria
1. WHEN the model is instantiated, THE system SHALL accept a `rnn_type` parameter ("LSTM" or "GRU") to select the recurrent layer type
2. WHEN the model is constructed, THE system SHALL include: nn.Embedding (vocab_size=25000, embedding_dim=128), stacked recurrent layer (num_layers=2, hidden_size=256, dropout=0.3), and nn.Linear (output_size=1)
3. WHEN the model processes input, THE system SHALL apply dropout (p=0.5) after the embedding layer
4. WHEN the model produces output, THE system SHALL use the hidden state of the last time step from the final recurrent layer
5. WHEN the model is printed, THE system SHALL display both the full architecture AND a breakdown of trainable parameters per layer together (both are required and SHALL NOT be shown independently)
6. IF an invalid `rnn_type` is provided, THEN THE system SHALL raise a ValueError with supported options

### Requirement 5: Entrenamiento y Validación
**User Story:** As a data scientist, I want to train both models with identical configuration, so that the comparison is fair and reproducible.

#### Acceptance Criteria
1. WHEN training begins, THE system SHALL set the random seed to 42 for PyTorch, NumPy, and Python's random module
2. WHEN the loss function is configured, THE system SHALL use `nn.BCEWithLogitsLoss`
3. WHEN the optimizer is configured, THE system SHALL use Adam with learning_rate=0.001
4. WHEN a training step completes backpropagation, THE system SHALL apply gradient clipping with max_norm=5.0
5. WHEN an epoch completes, THE system SHALL evaluate on the validation set and report train/val loss and accuracy; IF validation evaluation fails, THEN THE system SHALL stop training immediately
6. WHEN validation loss does not improve for 3 consecutive epochs (patience=3), THE system SHALL stop training early
7. WHEN validation loss improves, THE system SHALL save the model weights to disk as the best checkpoint
8. WHEN a GPU (CUDA or MPS) is available, THE system SHALL automatically move the model and data to that device (GPU usage is enforced when hardware is available)
9. IF no GPU is available, THEN THE system SHALL fall back to CPU execution without errors

### Requirement 6: Evaluación y Métricas
**User Story:** As a data scientist, I want comprehensive metrics on the test set for both models, so that I can quantify their performance differences.

#### Acceptance Criteria
1. WHEN evaluation is triggered, THE system SHALL load the best checkpoint weights and set the model to eval mode
2. WHEN generating predictions, THE system SHALL disable gradient computation (`torch.no_grad()`)
3. WHEN metrics are explicitly computed after evaluation, THE system SHALL report Accuracy, Precision, Recall, and F1-Score for each model (metrics SHALL only be reported after explicit computation, not cached or default values)
4. WHEN both models are evaluated, THE system SHALL present a side-by-side comparison table (LSTM vs GRU)
5. WHEN training time is measured, THE system SHALL report total training time and time per epoch for each model

### Requirement 7: Visualizaciones
**User Story:** As a data scientist, I want publication-quality plots comparing both models, so that I can visually analyze their behavior.

#### Acceptance Criteria
1. WHEN training completes, THE system SHALL generate training curves (loss and accuracy) with both LSTM and GRU overlaid on the same plot
2. WHEN evaluation completes, THE system SHALL generate a confusion matrix heatmap for each model (side by side)
3. WHEN training runs, THE system SHALL record gradient L2 norms per layer (embedding, RNN layer 1, RNN layer 2, Linear) during training steps 1 through 100 (counting from 1)
4. WHEN gradient monitoring data is available, THE system SHALL generate a gradient norm plot per layer comparing LSTM vs GRU
5. WHEN plots are generated, THE system SHALL save them as PNG files in an `outputs/` directory

### Requirement 8: Documento de Reflexiones
**User Story:** As a student, I want a markdown document answering theoretical questions based on experimental results, so that I can deepen my understanding.

#### Acceptance Criteria
1. WHEN all experiments complete, THE system SHALL generate a `REFLEXIONES.md` file addressing all 8 reflection questions from the project specification
2. WHEN answering each question, THE system SHALL reference specific experimental results (metrics, plots, observations) as evidence
3. WHEN discussing LSTM vs GRU, THE system SHALL include quantitative comparisons (accuracy difference, training time ratio, parameter count)
4. WHEN discussing gradients, THE system SHALL reference the gradient norm plots and identify vanishing or exploding gradient patterns only if such patterns are actually detected in the data

### Requirement 9: Reproducibilidad y Estructura del Proyecto
**User Story:** As a developer, I want a well-organized, reproducible project, so that anyone can replicate the results.

#### Acceptance Criteria
1. WHEN the project is structured, THE system SHALL organize code into separate modules within `src/`: `data.py`, `model.py`, `train.py`, `evaluate.py`, `visualize.py`, `reflexiones.py`, and a `main.py` entry point
2. WHEN the project is set up, THE system SHALL include a `requirements.txt` with pinned dependency versions as the sole dependency specification
3. WHEN `src/main.py` is executed, THE system SHALL verify that project setup is complete (all modules and config exist) before running the pipeline (data → train LSTM → train GRU → evaluate → visualize → generate reflexiones)
4. WHEN any random operation occurs, THE system SHALL use seed=42 to ensure identical results across runs
5. WHEN the project is structured, THE system SHALL include a `src/config.py` or similar with all hyperparameters centralized in one place
6. WHEN the project environment is set up, THE system SHALL use a Python virtual environment (`.venv/`) to isolate dependencies from the global Python installation
7. WHEN the project is structured, THE system SHALL include a `.gitignore` file that excludes the virtual environment directory, `__pycache__/`, IDE files, OS files, and generated outputs
