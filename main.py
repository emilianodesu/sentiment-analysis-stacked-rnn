"""Main pipeline orchestrator for the Stacked RNN sentiment analysis experiment."""

import os

import config
from data import get_dataloaders
from evaluate import EvaluationResults, evaluate_model, print_comparison_table
from model import SentimentRNN, print_model_summary
from reflexiones import generate_reflexiones
from train import TrainingHistory, get_device, set_seed, train_model
from visualize import save_all_plots


REQUIRED_MODULES = [
    "config.py",
    "data.py",
    "model.py",
    "train.py",
    "evaluate.py",
    "visualize.py",
    "reflexiones.py",
]


def verify_setup() -> None:
    """Verify that all required modules exist in the project directory.

    Checks for the presence of each module file listed in REQUIRED_MODULES.

    Raises:
        RuntimeError: If any required module file is missing, with a message
            listing all missing files.
    """
    missing = []
    for module_file in REQUIRED_MODULES:
        if not os.path.isfile(module_file):
            missing.append(module_file)

    if missing:
        raise RuntimeError(
            f"Missing required module files: {', '.join(missing)}. "
            "Ensure all modules are present before running the pipeline."
        )


def main() -> None:
    """Run the full sentiment analysis pipeline.

    Steps:
    1. Verify project setup
    2. Set global seed for reproducibility
    3. Get compute device
    4. Load data (download, clean, vocab, encode, pad, split)
    5. Create LSTM and GRU models
    6. Train both models with gradient monitoring
    7. Evaluate both models on the test set
    8. Print comparison table
    9. Generate and save all plots
    10. Generate REFLEXIONES.md document
    """
    # Step 1: Verify setup
    verify_setup()
    print("✓ All modules verified.\n")

    # Step 2: Set seed before any stochastic operation
    set_seed(config.SEED)
    print(f"✓ Seed set to {config.SEED}.\n")

    # Step 3: Get device
    device = get_device()
    print(f"✓ Using device: {device}\n")

    # Step 4: Load data
    print("Loading IMDB dataset...")
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        batch_size=config.BATCH_SIZE, seed=config.SEED
    )
    print(
        f"✓ Data loaded: "
        f"{len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val, "
        f"{len(test_loader.dataset)} test samples.\n"
    )

    # Step 5: Create and train LSTM
    print("=" * 60)
    print("TRAINING LSTM")
    print("=" * 60)
    set_seed(config.SEED)
    lstm_model = SentimentRNN(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        rnn_type="LSTM",
        rnn_dropout=config.RNN_DROPOUT,
        embed_dropout=config.EMBED_DROPOUT,
    )
    print("=== LSTM Model ===")
    print_model_summary(lstm_model)
    print()
    lstm_history = train_model(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config.LSTM_LEARNING_RATE,
    )

    # Step 6: Create and train GRU
    print("\n" + "=" * 60)
    print("TRAINING GRU")
    print("=" * 60)
    set_seed(config.SEED)
    gru_model = SentimentRNN(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        rnn_type="GRU",
        rnn_dropout=config.RNN_DROPOUT,
        embed_dropout=config.EMBED_DROPOUT,
    )
    print("\n=== GRU Model ===")
    print_model_summary(gru_model)
    print()
    gru_history = train_model(
        model=gru_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Step 7: Evaluate both models
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    lstm_checkpoint = os.path.join(config.CHECKPOINT_DIR, "LSTM_best.pt")
    gru_checkpoint = os.path.join(config.CHECKPOINT_DIR, "GRU_best.pt")

    lstm_results = evaluate_model(
        model=lstm_model,
        test_loader=test_loader,
        device=device,
        checkpoint_path=lstm_checkpoint,
        rnn_type="LSTM",
    )
    gru_results = evaluate_model(
        model=gru_model,
        test_loader=test_loader,
        device=device,
        checkpoint_path=gru_checkpoint,
        rnn_type="GRU",
    )

    # Step 8: Print comparison
    print()
    print_comparison_table(lstm_results, gru_results)

    # Step 9: Generate and save plots
    print("\nGenerating visualizations...")
    save_all_plots(
        lstm_history, gru_history, lstm_results, gru_results, config.OUTPUT_DIR
    )
    print(f"✓ Plots saved to {config.OUTPUT_DIR}\n")

    # Step 10: Generate REFLEXIONES.md
    print("Generating REFLEXIONES.md...")
    reflexiones_path = generate_reflexiones(
        lstm_history, gru_history, lstm_results, gru_results, config.OUTPUT_DIR
    )
    print(f"✓ Reflexiones document saved to {reflexiones_path}\n")

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
