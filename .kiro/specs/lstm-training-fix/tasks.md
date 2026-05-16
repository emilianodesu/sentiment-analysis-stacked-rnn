# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - LSTM Fails to Learn with Pre-Seed Instantiation
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to the concrete failing case: LSTM model instantiated before per-training `set_seed(42)`, then trained — accuracy should be >80% but will be ~50%
  - Write a property-based test (using Hypothesis) that:
    - Calls `set_seed(42)` once (simulating Step 2 in current main.py)
    - Creates an LSTM `SentimentRNN` model (simulating Step 5 — before per-training seed)
    - Calls `set_seed(42)` again (simulating Step 6 — per-training seed reset)
    - Trains the LSTM model for a few epochs on the IMDB data
    - Asserts that the LSTM achieves validation accuracy > 0.80 (from Expected Behavior in design)
  - The bug condition from design: `isBugCondition(input)` where `input.model.rnn_type == "LSTM" AND input.model.instantiated_before_training_seed == True AND input.model.weights_not_reinitialized_after_seed == True`
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (LSTM achieves ~50% accuracy, not >80%) — this proves the bug exists
  - Document counterexamples found (e.g., "LSTM trained with pre-seed instantiation achieves ~50% accuracy instead of >80%")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - GRU Accuracy and Pipeline Integrity
  - **IMPORTANT**: Follow observation-first methodology
  - Observe: GRU model trained with current `main.py` ordering achieves ~87% test accuracy on UNFIXED code
  - Observe: Pipeline completes without errors on UNFIXED code (train → evaluate → visualize → reflexiones)
  - Observe: All 213 existing tests pass on UNFIXED code
  - Write property-based tests (using Hypothesis) that:
    - Verify GRU model achieves test accuracy ≈ 87% (within tolerance, e.g., >0.84) after training with `set_seed(42)` → create GRU → train GRU pattern
    - Verify GRU training produces deterministic results across two runs with same seed
    - Verify model hyperparameters (hidden_size=256, num_layers=2, embed_dropout=0.5, rnn_dropout=0.3, lr=0.001) are unchanged
  - The non-bug condition from design: inputs where `isBugCondition` returns false (GRU model, or any model instantiated immediately after its per-training seed call)
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline GRU behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 3. Fix for LSTM training failure due to model instantiation ordering

  - [x] 3.1 Implement the fix in main.py
    - Move LSTM `SentimentRNN(...)` instantiation from Step 5 to immediately after `set_seed(config.SEED)` in the LSTM training block
    - Move GRU `SentimentRNN(...)` instantiation from Step 5 to immediately after `set_seed(config.SEED)` in the GRU training block
    - Move `print_model_summary(lstm_model)` call to after LSTM instantiation in its training block
    - Move `print_model_summary(gru_model)` call to after GRU instantiation in its training block
    - Remove the old Step 5 block that created both models together
    - Ensure `lstm_model` and `gru_model` variables remain in scope for Step 7 (evaluation)
    - Resulting structure: `set_seed(42)` → create model → `print_model_summary` → `train_model` for each model
    - _Bug_Condition: isBugCondition(input) where input.model.instantiated_before_training_seed == True_
    - _Expected_Behavior: LSTM achieves >80% accuracy with post-seed instantiation_
    - _Preservation: GRU ~87% accuracy unchanged, all 213 tests pass, pipeline completes_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_

  - [x] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - LSTM Achieves Discriminative Accuracy
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior (LSTM accuracy > 80%)
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms LSTM now achieves >80% accuracy with the fix)
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - GRU Accuracy and Pipeline Integrity
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms GRU still achieves ~87% and no regressions)
    - Confirm all tests still pass after fix (no regressions)
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 3.4 Verify all 213 existing tests still pass
    - Run `python -m pytest tests/ --tb=short` to execute the full test suite
    - **EXPECTED OUTCOME**: All 213 tests PASS
    - If any test fails, investigate whether the fix introduced a regression and correct it
    - _Requirements: 3.4_

- [x] 4. Checkpoint - Ensure all tests pass
  - Run the full test suite: `python -m pytest tests/ --tb=short`
  - Verify LSTM achieves >80% accuracy (bug condition test passes)
  - Verify GRU achieves ~87% accuracy (preservation test passes)
  - Verify all 213 existing tests pass
  - Ensure all tests pass, ask the user if questions arise.
