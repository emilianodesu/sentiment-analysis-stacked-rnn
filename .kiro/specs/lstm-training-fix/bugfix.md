# Bugfix Requirements Document

## Introduction

The LSTM model in the Stacked RNN sentiment analysis pipeline fails to learn, achieving only ~50% accuracy (random chance) while the GRU model converges correctly to ~87% accuracy under identical hyperparameters. The root cause is a model initialization ordering issue in `main.py`: both models are instantiated after a single `set_seed(42)` call, but `set_seed(42)` is called again before each model's training without re-initializing the model weights. The LSTM, created first, gets a particular weight initialization that leads to a degenerate local minimum (predicting all-positive), while the GRU benefits from different random state consumed after the LSTM's instantiation.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the LSTM model is instantiated before `set_seed(42)` is called for training THEN the system trains the LSTM with stale weight initialization that was determined by the random state at model creation time, leading to ~50% accuracy (no learning)

1.2 WHEN the LSTM model trains with its initial weights from the first `set_seed(42)` call THEN the system produces a model that predicts almost all samples as positive (91% recall, 50% precision) with no discriminative ability

1.3 WHEN `set_seed(42)` is called before LSTM training THEN the system does not re-initialize the LSTM model weights, only resetting random state for data iteration and dropout masks

### Expected Behavior (Correct)

2.1 WHEN the LSTM model is trained THEN the system SHALL ensure the model's weights are initialized from a deterministic seed state immediately before training begins, so that the initialization and training share the same reproducible random sequence

2.2 WHEN the LSTM model trains with properly seeded weight initialization THEN the system SHALL produce a model that achieves discriminative accuracy comparable to the GRU (above 80% on the validation set)

2.3 WHEN `set_seed(42)` is called before a model's training THEN the system SHALL also re-initialize (or freshly instantiate) that model's weights so that the seed controls both initialization and training stochasticity

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the GRU model is trained THEN the system SHALL CONTINUE TO achieve approximately 87% accuracy on the test set

3.2 WHEN `set_seed(42)` is used for reproducibility THEN the system SHALL CONTINUE TO produce deterministic, reproducible results across runs for both models

3.3 WHEN the pipeline runs end-to-end THEN the system SHALL CONTINUE TO train both models, evaluate them, generate comparison plots, and produce the REFLEXIONES.md document without errors

3.4 WHEN model hyperparameters (hidden_size=256, num_layers=2, embed_dropout=0.5, rnn_dropout=0.3, lr=0.001) are used THEN the system SHALL CONTINUE TO use these same values for both LSTM and GRU models
