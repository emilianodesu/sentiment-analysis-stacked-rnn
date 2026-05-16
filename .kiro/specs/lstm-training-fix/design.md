# LSTM Training Fix Bugfix Design

## Overview

The LSTM model in the Stacked RNN sentiment analysis pipeline fails to learn (~50% accuracy, equivalent to random chance) because its weights are initialized before `set_seed(42)` is called for training. In `main.py`, both models are instantiated at Step 5 after a single `set_seed(42)` call (Step 2), but when `set_seed(42)` is called again before each model's training (Step 6), only the random state for data iteration and dropout is reset — the model weights remain unchanged. The LSTM's initial weights (determined by the random state consumed during its creation) lead it into a degenerate local minimum where it predicts all-positive. The fix moves model instantiation to immediately before each model's training, after the per-model `set_seed(42)` call, ensuring deterministic and effective weight initialization.

## Glossary

- **Bug_Condition (C)**: The condition where the LSTM model is instantiated before the per-training `set_seed(42)` call, causing its weights to be initialized from a non-deterministic random state relative to training
- **Property (P)**: The desired behavior where the LSTM model achieves discriminative accuracy (>80%) comparable to the GRU model (~87%)
- **Preservation**: The GRU model's ~87% accuracy, pipeline reproducibility, and all existing test behavior must remain unchanged
- **set_seed(42)**: Function in `train.py` that fixes random seeds for Python, NumPy, PyTorch CPU/CUDA, and cuDNN determinism
- **SentimentRNN**: The model class in `model.py` supporting both LSTM and GRU architectures via `rnn_type` parameter
- **train_model**: The training function in `train.py` that runs the training loop with early stopping and gradient monitoring
- **Degenerate local minimum**: A state where the LSTM predicts all samples as positive (91% recall, 50% precision), yielding ~50% accuracy

## Bug Details

### Bug Condition

The bug manifests when the LSTM model is instantiated (Step 5 in `main.py`) after the initial `set_seed(42)` (Step 2), but before the per-training `set_seed(42)` (Step 6). The model's weight initialization is determined by the random state at creation time, not at training time. When `set_seed(42)` is called again before training, it resets the RNG but does NOT re-initialize the model weights, leaving the LSTM with weights that lead to a degenerate local minimum.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type PipelineExecution
  OUTPUT: boolean
  
  RETURN input.model.rnn_type == "LSTM"
         AND input.model.instantiated_before_training_seed == True
         AND input.model.weights_not_reinitialized_after_seed == True
END FUNCTION
```

### Examples

- **LSTM training with current code**: `set_seed(42)` → create LSTM → create GRU → `set_seed(42)` → train LSTM → Result: ~50% accuracy (no learning, predicts all-positive)
- **GRU training with current code**: `set_seed(42)` → create LSTM → create GRU → `set_seed(42)` → train LSTM → `set_seed(42)` → train GRU → Result: ~87% accuracy (GRU benefits from different random state at creation)
- **LSTM training with fix**: `set_seed(42)` → create LSTM → train LSTM → Result: >80% accuracy (weights initialized from deterministic seed state)
- **Edge case — single model**: If only one model were created and trained immediately after `set_seed(42)`, the bug would not manifest

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- GRU model must continue to achieve approximately 87% accuracy on the test set
- Both models must produce deterministic, reproducible results across runs when using `set_seed(42)`
- The full pipeline (train → evaluate → visualize → reflexiones) must complete without errors
- Model hyperparameters (hidden_size=256, num_layers=2, embed_dropout=0.5, rnn_dropout=0.3, lr=0.001) must remain the same for both models
- All 213 existing tests must continue to pass

**Scope:**
All inputs that do NOT involve the LSTM model's weight initialization ordering should be completely unaffected by this fix. This includes:
- GRU model training and evaluation
- Data loading and preprocessing
- Visualization and report generation
- Evaluation metrics computation
- Checkpoint saving and loading

## Hypothesized Root Cause

Based on the bug description, the most likely issue is:

1. **Model Instantiation Ordering**: In `main.py` lines 85-101, both models are created at Step 5, after the initial `set_seed(42)` at Step 2 (line 67). The random state consumed during LSTM creation (embedding init, RNN weight init, linear layer init) determines its initial weights. When `set_seed(42)` is called again at line 109 before LSTM training, it resets the RNG but the LSTM's weights are already fixed from creation time.

2. **Weight Initialization Sensitivity**: The LSTM architecture with 2 stacked layers (num_layers=2) and hidden_size=256 is more sensitive to initial weight values than the GRU. The particular weight initialization from the post-Step-2 random state leads the LSTM into a degenerate local minimum where it predicts all-positive.

3. **GRU Accidental Success**: The GRU model is created second (after the LSTM consumes some random state), so it gets a different initialization. Combined with GRU's simpler gate structure, this happens to produce weights that allow successful training.

4. **Seed Reset Without Weight Reset**: The `set_seed(42)` call before training only resets the RNG for data shuffling, dropout masks, and optimizer initialization — it does NOT re-initialize model parameters. This is the core defect.

## Correctness Properties

Property 1: Bug Condition - LSTM Achieves Discriminative Accuracy

_For any_ pipeline execution where the LSTM model is trained with weights initialized from a deterministic seed state immediately before training begins, the fixed pipeline SHALL produce an LSTM model that achieves validation accuracy above 80%, demonstrating discriminative ability (not predicting all-positive).

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - GRU Accuracy and Pipeline Integrity

_For any_ pipeline execution after the fix is applied, the GRU model SHALL continue to achieve approximately 87% test accuracy, both models SHALL produce deterministic reproducible results across runs, and all 213 existing tests SHALL continue to pass without modification.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `main.py`

**Function**: `main()`

**Specific Changes**:

1. **Move LSTM instantiation after per-training seed**: Move the LSTM `SentimentRNN(...)` creation from Step 5 to immediately after `set_seed(config.SEED)` in the LSTM training block (before `train_model` is called for LSTM).

2. **Move GRU instantiation after per-training seed**: Move the GRU `SentimentRNN(...)` creation from Step 5 to immediately after `set_seed(config.SEED)` in the GRU training block (before `train_model` is called for GRU).

3. **Preserve model summary printing**: Keep `print_model_summary` calls, but move them to after each model is created in its respective training block.

4. **Maintain evaluation flow**: The evaluation section (Step 7) references `lstm_model` and `gru_model` — these variables must still be in scope after training. Since models are created before training in each block, they remain accessible.

5. **Keep checkpoint paths unchanged**: The checkpoint paths (`LSTM_best.pt`, `GRU_best.pt`) are derived from `model.rnn_type` inside `train_model`, so no changes needed there.

**Resulting code structure:**
```python
# Step 5+6 combined: Create and train LSTM
set_seed(config.SEED)
lstm_model = SentimentRNN(...)  # Weights initialized from seed=42 state
print_model_summary(lstm_model)
lstm_history = train_model(lstm_model, ...)

# Step 5+6 combined: Create and train GRU
set_seed(config.SEED)
gru_model = SentimentRNN(...)  # Weights initialized from seed=42 state
print_model_summary(gru_model)
gru_history = train_model(gru_model, ...)
```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that verify the LSTM model's weight initialization is deterministic relative to the seed call, and that training with the current ordering produces poor accuracy. Run these tests on the UNFIXED code to observe failures and understand the root cause.

**Test Cases**:
1. **Seed-Weight Determinism Test**: Create a model after `set_seed(42)` and verify its weights are identical across runs (will pass on both fixed and unfixed code — baseline)
2. **Initialization Ordering Test**: Create model BEFORE `set_seed(42)`, then call `set_seed(42)` — verify weights are NOT reset (demonstrates the bug mechanism)
3. **LSTM Accuracy Test**: Train LSTM with current `main.py` ordering and assert accuracy > 80% (will FAIL on unfixed code, demonstrating the bug)
4. **Weight Divergence Test**: Create two models with different seed orderings and compare weights (will show they differ, confirming root cause)

**Expected Counterexamples**:
- LSTM trained with pre-seed initialization achieves ~50% accuracy
- Model weights after `set_seed(42)` are unchanged from creation-time values
- Possible causes: weight initialization not coupled to training seed

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL execution WHERE isBugCondition(execution) DO
  result := main_fixed(execution)
  ASSERT result.lstm_accuracy > 0.80
  ASSERT result.lstm_predictions != all_positive
  ASSERT result.lstm_loss_decreasing == True
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL execution WHERE NOT isBugCondition(execution) DO
  ASSERT main_original(execution).gru_accuracy ≈ main_fixed(execution).gru_accuracy
  ASSERT main_original(execution).pipeline_completes == main_fixed(execution).pipeline_completes
  ASSERT all_existing_tests_pass(main_fixed)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many random seed values and model configurations to verify GRU behavior is unchanged
- It catches edge cases where the fix might accidentally affect non-LSTM code paths
- It provides strong guarantees that the GRU's training dynamics are preserved

**Test Plan**: Observe behavior on UNFIXED code first for GRU training and pipeline completion, then write property-based tests capturing that behavior.

**Test Cases**:
1. **GRU Accuracy Preservation**: Verify GRU achieves ~87% accuracy after the fix (same as before)
2. **Reproducibility Preservation**: Run the fixed pipeline twice with same seed and verify identical results
3. **Pipeline Completion Preservation**: Verify all pipeline steps (train, evaluate, visualize, reflexiones) complete without errors
4. **Existing Test Suite Preservation**: Run all 213 existing tests and verify they pass

### Unit Tests

- Test that `SentimentRNN` created immediately after `set_seed(42)` produces deterministic weights
- Test that model weights are NOT affected by a subsequent `set_seed(42)` call (confirming the bug mechanism)
- Test that model created after `set_seed(42)` has different weights than model created before `set_seed(42)`
- Test that the fixed `main()` function creates models in the correct order relative to seed calls

### Property-Based Tests

- Generate random seed values and verify that model instantiation after `set_seed(seed)` always produces the same weights for that seed (determinism property)
- Generate random hyperparameter configurations and verify that the fix pattern (seed → create → train) always produces better-than-chance accuracy for LSTM
- Generate random pipeline configurations and verify GRU accuracy is preserved regardless of LSTM fix

### Integration Tests

- Test full pipeline execution with the fix and verify both models achieve expected accuracy
- Test that checkpoints saved by the fixed pipeline can be loaded and produce correct evaluation results
- Test that visualization and reflexiones generation work correctly with the fixed training histories
