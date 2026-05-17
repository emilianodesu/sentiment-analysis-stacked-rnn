"""Centralized hyperparameters and configuration for the Stacked RNN project."""

# Reproducibility
SEED = 42

# Vocabulary
VOCAB_SIZE = 25_000
MIN_FREQ = 2

# Sequence
MAX_LEN = 200

# Model architecture
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
RNN_DROPOUT = 0.3
EMBED_DROPOUT = 0.5

# Training
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LSTM_LEARNING_RATE = 0.001
MAX_EPOCHS = 10
PATIENCE = 5
GRAD_CLIP_NORM = 5.0

# Data splits
TRAIN_SIZE = 20_000
VAL_SIZE = 5_000

# Monitoring
GRADIENT_MONITOR_STEPS = 100

# Output paths
OUTPUT_DIR = "outputs/"
CHECKPOINT_DIR = "checkpoints/"
